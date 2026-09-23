import numpy as np
import cv2
from typing import Dict, List
from collections import OrderedDict
import sys
import os
import os.path as osp

from utils.registry import Registry
from utils.textblock_mask import extract_ballon_mask, _is_gradient_bg
from utils.imgproc_utils import enlarge_window, smart_resize
from utils.logger import logger as LOGGER

from ..base import BaseModule, DEFAULT_DEVICE, soft_empty_cache, DEVICE_SELECTOR, GPUINTENSIVE_SET, TORCH_DTYPE_MAP, BF16_SUPPORTED
from ..textdetector import TextBlock

INPAINTERS = Registry('inpainters')
register_inpainter = INPAINTERS.register_module


def _laplacian_pyramid_blend(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
    """Blend img_a (inpainted) into img_b (original) at mask boundary via Laplacian pyramid.
    Eliminates hard rectangular seams on gradient/screentone backgrounds.
    """
    # Soft mask: blur boundary for feathering, float [0,1]
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (31, 31), 0)

    def build_laplacian(img):
        gp = [img.astype(np.float32)]
        for _ in range(levels - 1):
            gp.append(cv2.pyrDown(gp[-1]))
        lp = []
        for i in range(levels - 1):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1].astype(np.float32))
        # lp[0] is finest, lp[-1] is coarsest
        return lp

    def build_gauss_pyramid(img, n):
        gp = [img]
        for _ in range(n - 1):
            gp.append(cv2.pyrDown(gp[-1]))
        # gp[0] is finest, gp[-1] is coarsest — same ordering as Laplacian pyramid
        return gp

    lp_a = build_laplacian(img_a)
    lp_b = build_laplacian(img_b)
    gp_mask = build_gauss_pyramid(mask_f, levels)

    # Blend each level (both pyramids have same ordering: finest→coarsest)
    blended_lp = []
    for la, lb, m in zip(lp_a, lp_b, gp_mask):
        m3 = m[:, :, None] if la.ndim == 3 else m
        blended_lp.append(la * m3 + lb * (1.0 - m3))

    # Reconstruct from coarsest to finest
    result = blended_lp[-1]
    for layer in reversed(blended_lp[:-1]):
        result = cv2.pyrUp(result, dstsize=(layer.shape[1], layer.shape[0])) + layer
    return np.clip(result, 0, 255).astype(np.uint8)


def inpaint_handle_alpha_channel(original_alpha, mask):
    '''
    perhaps a better idea is to feed the alpha into inpainting model, but it'll double the cost  
    for now it just return the original alpha
    '''

    result_alpha = original_alpha.copy()

    # Analyze the alpha values around the original mask to determine appropriate transparency
    mask_dilated = cv2.dilate((mask > 127).astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    surrounding_mask = mask_dilated - (mask > 127).astype(np.uint8)

    if np.any(surrounding_mask > 0):
        surrounding_alpha = original_alpha[surrounding_mask > 0]
        if len(surrounding_alpha) > 0:
            median_surrounding_alpha = np.median(surrounding_alpha)
            # If surrounding area is mostly transparent (median alpha < 128),
            # make inpainted areas transparent too
            if median_surrounding_alpha < 128:
                inpainted_mask = (mask > 127)
                result_alpha[inpainted_mask] = median_surrounding_alpha

    return result_alpha

class InpainterBase(BaseModule):

    inpaint_by_block = True
    check_need_inpaint = True

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in INPAINTERS.module_dict:
            if INPAINTERS.module_dict[key] == self.__class__:
                self.name = key
                break
    
    def memory_safe_inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        '''
        handle cuda out of memory
        '''
        try:
            return self._inpaint(img, mask, textblock_list)
        except Exception as e:
            is_oom = (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(e).lower()
                or "cuda error: out of memory" in str(e).lower()
            )
            if is_oom:
                soft_empty_cache()
                try:
                    return self._inpaint(img, mask, textblock_list)
                except Exception as ee:
                    if isinstance(ee, torch.cuda.OutOfMemoryError) or "out of memory" in str(ee).lower():
                        self.logger.warning(f'CUDA out of memory while calling {self.name}, fall back to cpu...\n'
                                            f'if running into it frequently, consider lowering the inpaint_size')
                        self.moveToDevice('cpu')
                        inpainted = self._inpaint(img, mask, textblock_list)
                        precision = getattr(self, 'precision', None)
                        self.moveToDevice('cuda' if torch.cuda.is_available() else 'cpu', precision)

                        return inpainted
            raise e

    def inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None, check_need_inpaint: bool = False) -> np.ndarray:
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return img
        if mask is None or mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
            return img
        
        if not self.all_model_loaded():
            self.load_model()
        
        # Handle RGBA images by preserving alpha channel
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3:4]  # Keep alpha channel
            img_rgb = img[:, :, :3]  # Use only RGB for inpainting
        else:
            img_rgb = img
        
        if not self.inpaint_by_block or textblock_list is None:
            # Full-page neural inpainting
            result_rgb = self.memory_safe_inpaint(img_rgb, mask, textblock_list)
            if result_rgb is None or result_rgb.size == 0 or result_rgb.shape[:2] != img_rgb.shape[:2]:
                raise RuntimeError(f"[{self.name}] Inpainting produced invalid or empty output tensor.")
            
            # Secondary Pass for Un-inpainted High-Frequency Zones (Optimized Local ROI + Screentone Baseline)
            if textblock_list and len(textblock_list) > 0:
                im_h, im_w = result_rgb.shape[:2]
                residual_blocks = []
                for blk in textblock_list:
                    bx1, by1, bx2, by2 = [int(v) for v in blk.xyxy]
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(im_w, bx2), min(im_h, by2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    
                    box_mask = mask[by1:by2, bx1:bx2]
                    if np.count_nonzero(box_mask > 0) == 0:
                        continue
                    
                    box_crop = result_rgb[by1:by2, bx1:bx2]
                    gray_crop = cv2.cvtColor(box_crop, cv2.COLOR_RGB2GRAY)
                    
                    # Screentone Baseline Subtraction:
                    # Measure local background's baseline Laplacian variance on a 10px strip outside the mask perimeter
                    sx1 = max(0, bx1 - 10)
                    sy1 = max(0, by1 - 10)
                    sx2 = min(im_w, bx2 + 10)
                    sy2 = min(im_h, by2 + 10)
                    
                    strip_crop = result_rgb[sy1:sy2, sx1:sx2]
                    strip_gray = cv2.cvtColor(strip_crop, cv2.COLOR_RGB2GRAY)
                    strip_lap = cv2.Laplacian(strip_gray, cv2.CV_64F)
                    strip_mask = mask[sy1:sy2, sx1:sx2]
                    
                    dilated_strip_m = cv2.dilate((strip_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                    outside_strip = (dilated_strip_m > 0) & (strip_mask == 0)
                    bg_lap_var = float(np.var(strip_lap[outside_strip])) if np.count_nonzero(outside_strip) >= 15 else 0.0
                    
                    inner_m = (strip_mask > 0)
                    inner_lap_var = float(np.var(strip_lap[inner_m])) if np.count_nonzero(inner_m) >= 10 else float(np.var(strip_lap))
                    
                    # Net Laplacian variance after subtracting background screentone baseline
                    net_lap_var = max(0.0, inner_lap_var - bg_lap_var)
                    
                    # Measure edge density inside the inpainted mask region
                    canny = cv2.Canny(gray_crop, 50, 150)
                    masked_edges = canny[box_mask > 0]
                    edge_ratio = float(np.count_nonzero(masked_edges > 0)) / max(1, len(masked_edges))
                    
                    is_balloon = getattr(blk, 'is_balloon', True)
                    var_thresh = 120.0 if is_balloon else 300.0
                    
                    if net_lap_var > var_thresh and edge_ratio > 0.06:
                        residual_blocks.append((blk, net_lap_var, edge_ratio))
                
                if residual_blocks:
                    LOGGER.warning(
                        f"[{self.name}] High-frequency residual text detected in {len(residual_blocks)} blocks "
                        f"(Max Net Laplacian Var: {max(b[1] for b in residual_blocks):.1f}). Triggering optimized local ROI 2nd-pass..."
                    )
                    
                    for blk, lvar, eratio in residual_blocks:
                        bx1, by1, bx2, by2 = [int(v) for v in blk.xyxy]
                        bw = bx2 - bx1
                        bh = by2 - by1
                        if bw <= 0 or bh <= 0:
                            continue

                        # Detect gradient background: 2D Sobel magnitude > 8.0 or non-balloon
                        blk_gray = cv2.cvtColor(result_rgb[by1:by2, bx1:bx2], cv2.COLOR_RGB2GRAY)
                        is_gradient_zone = _is_gradient_bg(blk_gray, mask=mask[by1:by2, bx1:bx2]) or not getattr(blk, 'is_balloon', True)
                        
                        # Crop strictly to affected bounding box expanded by 1.3x (or 2.2x vertical for gradients)
                        pad_x = int(round(bw * 0.15)) + 16
                        pad_y = int(round(bh * 0.15)) + 16
                        if is_gradient_zone:
                            # Minimum 2.2× vertical expansion so LaMa sees full gradient span
                            pad_y = max(pad_y, int(round(bh * 0.60)) + 16)
                        rx1 = max(0, bx1 - pad_x)
                        ry1 = max(0, by1 - pad_y)
                        rx2 = min(im_w, bx2 + pad_x)
                        ry2 = min(im_h, by2 + pad_y)
                        
                        roi_h = ry2 - ry1
                        roi_w = rx2 - rx1
                        if roi_h <= 0 or roi_w <= 0:
                            continue
                        
                        roi_img = result_rgb[ry1:ry2, rx1:rx2].copy()
                        roi_orig = img_rgb[ry1:ry2, rx1:rx2].copy()
                        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                        
                        # Expanded mask within local ROI
                        ebx1 = max(0, int(round(bx1 - bw * 0.15)) - rx1)
                        eby1 = max(0, int(round(by1 - bh * 0.15)) - ry1)
                        ebx2 = min(roi_w, int(round(bx2 + bw * 0.15)) - rx1)
                        eby2 = min(roi_h, int(round(by2 + bh * 0.15)) - ry1)
                        
                        roi_mask[eby1:eby2, ebx1:ebx2] = 255
                        roi_mask = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                        
                        # Execute local CUDA LaMa inference on cropped ROI
                        roi_inpainted = self.memory_safe_inpaint(roi_img, roi_mask, None)
                        if roi_inpainted is not None and roi_inpainted.shape == roi_img.shape:
                            if is_gradient_zone:
                                # Seamless Gradient Integration: Laplacian Pyramid Blending eliminates hard seam
                                blended = _laplacian_pyramid_blend(roi_inpainted, roi_orig, roi_mask, levels=4)
                                result_rgb[ry1:ry2, rx1:rx2] = blended
                            else:
                                roi_mask_3d = (roi_mask > 0)[:, :, None]
                                result_rgb[ry1:ry2, rx1:rx2] = np.where(roi_mask_3d, roi_inpainted, result_rgb[ry1:ry2, rx1:rx2])
                            mask[ry1:ry2, rx1:rx2] = np.maximum(mask[ry1:ry2, rx1:rx2], roi_mask)
                            
                    LOGGER.info(f"✓ [{self.name}] Optimized local ROI 2nd-pass completed on {len(residual_blocks)} residual zones.")

            # Strict pixel preservation invariant: Pixels outside mask must remain 100% original
            mask_3d = mask[:, :, None] if mask.ndim == 2 else mask
            mask_bin = (mask_3d > 0)
            result_rgb = np.where(mask_bin, result_rgb, img_rgb)

            # Measure actual Unmasked Pixel MAE dynamically
            unmasked_mask = (~mask_bin[:, :, 0]) if mask_bin.ndim == 3 else (~mask_bin)
            if np.any(unmasked_mask):
                unmasked_mae = float(np.mean(np.abs(result_rgb[unmasked_mask].astype(np.float32) - img_rgb[unmasked_mask].astype(np.float32))))
            else:
                unmasked_mae = 0.0
            
            # Measure screentone texture standard deviation
            mask_bool = (mask > 0)
            if np.any(mask_bool):
                inpainted_std = float(np.std(result_rgb[mask_bool]))
                surround_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                dilated_surround = cv2.dilate(mask.astype(np.uint8), surround_elem) > 0
                surround_only = dilated_surround & (~mask_bool)
                surround_std = float(np.std(img_rgb[surround_only])) if np.any(surround_only) else inpainted_std
                LOGGER.info(f"[{self.name}] Inpainting Complete | Unmasked MAE: {unmasked_mae:.6f} | Texture Std Dev (Inpainted: {inpainted_std:.2f}, Background: {surround_std:.2f})")
            else:
                LOGGER.info(f"[{self.name}] Inpainting Complete | Unmasked MAE: {unmasked_mae:.6f} | No active mask pixels.")

            # Recombine with alpha if original was RGBA
            if original_alpha is not None:
                result_alpha = inpaint_handle_alpha_channel(original_alpha, mask)
                return np.concatenate([result_rgb, result_alpha], axis=2)
            return result_rgb
        else:
            im_h, im_w = img_rgb.shape[:2]
            inpainted = np.copy(img_rgb)
            
            # Preserve original mask for transparency analysis
            original_mask = mask.copy()
            
            for blk in textblock_list:
                xyxy = blk.xyxy
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                ratio = 1.7
                if max(bw, bh) < 60:
                    ratio = max(2.5, min(4.0, 150.0 / max(1, max(bw, bh))))
                xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=ratio)
                if xyxy_e[2] <= xyxy_e[0] or xyxy_e[3] <= xyxy_e[1]:
                    continue
                im = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                msk = mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                if im.size == 0 or msk.size == 0 or im.shape[0] == 0 or im.shape[1] == 0:
                    continue
                
                # Execute neural inpaint for every non-empty mask (NO flat color bypass)
                if np.any(msk > 0):
                    inpaint_crop = self.memory_safe_inpaint(im, msk)
                    if inpaint_crop is not None and inpaint_crop.shape == im.shape:
                        msk_3d = msk[:, :, None] if msk.ndim == 2 else msk
                        inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = np.where(msk_3d > 0, inpaint_crop, im)

                mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0
            
            # Enforce 100% pixel preservation outside total mask
            orig_mask_3d = original_mask[:, :, None] if original_mask.ndim == 2 else original_mask
            inpainted = np.where(orig_mask_3d > 0, inpainted, img_rgb)

            # Recombine with alpha if original was RGBA
            if original_alpha is not None:
                result_alpha = inpaint_handle_alpha_channel(original_alpha, original_mask)
                return np.concatenate([inpainted, result_alpha], axis=2)
            return inpainted

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        raise NotImplementedError
    
    def moveToDevice(self, device: str, precision: str = None):
        raise not NotImplementedError


@register_inpainter('opencv-tela')
class OpenCVInpainter(InpainterBase):

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.inpaint_method = lambda img, mask, *args, **kwargs: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


@register_inpainter('patchmatch')
class PatchmatchInpainter(InpainterBase):

    if sys.platform == 'darwin':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/macos_arm64_patchmatch_libs.7z',
                'sha256_pre_calculated': ['843704ab096d3afd8709abe2a2c525ce3a836bb0a629ed1ee9b8f5cee9938310', '849ca84759385d410c9587d69690e668822a3fc376ce2219e583e7e0be5b5e9a'],
                'files': ['macos_libopencv_world.4.8.0.dylib', 'macos_libpatchmatch_inpaint.dylib'],
                'save_dir': 'data/libs',
                'archived_files': 'macos_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': '9f332c888be0f160dbe9f6d6887eb698a302e62f4c102a0f24359c540d5858ea'
        }]
    elif sys.platform == 'win32':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/windows_patchmatch_libs.7z',
                'sha256_pre_calculated': ['3b7619caa29dc3352b939de4e9981217a9585a13a756e1101a50c90c100acd8d', '0ba60cfe664c97629daa7e4d05c0888ebfe3edcb3feaf1ed5a14544079c6d7af'],
                'files': ['opencv_world455.dll', 'patchmatch_inpaint.dll'],
                'save_dir': 'data/libs',
                'archived_files': 'windows_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': 'c991ff61f7cb3efaf8e75d957e62d56ba646083bc25535f913ac65775c16ca65'
        }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        from . import patch_match
        self.inpaint_method = lambda img, mask, *args, **kwargs: patch_match.inpaint(img, mask, patch_size=3)
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


import torch
from utils.imgproc_utils import resize_keepasp
from .aot import AOTGenerator, load_aot_model


@register_inpainter('aot')
class AOTInpainter(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'value': 2048
        }, 
        'device': DEVICE_SELECTOR(),
        'description': 'manga-image-translator inpainter'
    }

    device = DEFAULT_DEVICE
    inpaint_size = 2048
    model: AOTGenerator = None
    _load_model_keys = {'model'}

    download_file_list = [{
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt',
            'sha256_pre_calculated': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'files': 'data/models/aot_inpainter.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.model: AOTGenerator = None
        
    def _load_model(self):
        AOTMODEL_PATH = 'data/models/aot_inpainter.ckpt'
        self.model = load_aot_model(AOTMODEL_PATH, self.device)

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None

        img = resize_keepasp(img, new_shape, stride=None)
        mask = resize_keepasp(mask, new_shape, stride=None)

        im_h, im_w = img.shape[:2]
        pad_bottom = 128 - im_h if im_h < 128 else 0
        pad_right = 128 - im_w if im_w < 128 else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])


from .lama import LamaFourier, load_lama_mpe

@register_inpainter('lama_mpe')
class LamaInpainterMPE(InpainterBase):

    inpaint_by_block = False
    check_need_inpaint = False

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048,
                2560
            ], 
            'value': 2048
        },
        'inpaint_passes': {
            'type': 'selector',
            'options': [
                1,
                2,
                3
            ], 
            'value': 2
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone'])
    }

    download_file_list = [{
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt',
            'sha256_pre_calculated': 'd625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc',
            'files': 'data/models/lama_mpe.ckpt',
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.inpaint_passes = int(self.params.get('inpaint_passes', {}).get('value', 2))
        self.precision = 'fp32'
        self.model: LamaFourier = None

    def _load_model(self):
        self.model = load_lama_mpe(r'data/models/lama_mpe.ckpt', self.device)

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
        # Apply slight Gaussian blur (ksize=(5, 5), sigma=1.0) on binary mask to ensure seamless edge feathering
        mask_feathered = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
        mask = np.clip(mask_feathered, 0, 255).astype(np.uint8)

        # high resolution input could produce cloudy artifacts
        img = resize_keepasp(img, new_shape, stride=64)
        mask = resize_keepasp(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch > 0.05] = 1.0
        mask_torch[mask_torch <= 0.05] = 0.0
        if self.model is not None and getattr(self.model, 'mpe', None) is not None:
            rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
            rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
            direct = torch.LongTensor(direct).unsqueeze_(0)
            if self.device != 'cpu':
                rel_pos = rel_pos.to(self.device)
                direct = direct.to(self.device)
        else:
            rel_pos = None
            direct = None

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return img
        if mask is None or mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
            return img

        im_h, im_w = img.shape[:2]
        passes = max(1, getattr(self, 'inpaint_passes', 2))
        curr_img = img

        for p_idx in range(passes):
            active_pixels = int(np.count_nonzero(mask > 0))
            LOGGER.info(f"[{self.name}] Neural Inpaint Pass {p_idx + 1}/{passes} executing on {curr_img.shape[:2]} (Active Mask Pixels: {active_pixels})...")
            img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(curr_img, mask)
            
            precision = TORCH_DTYPE_MAP[self.precision]
            if self.device in {'cuda'}:
                try:
                    with torch.autocast(device_type=self.device, dtype=precision):
                        img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
                except Exception as e:
                    self.logger.error(e)
                    self.logger.error(f'{precision} inference is not supported for this device, use fp32 instead.')
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            else:
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

            img_inpainted = (img_inpainted_torch.to(device='cpu', dtype=torch.float32).squeeze_(0).permute(1, 2, 0).numpy() * 255)
            img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
            if pad_bottom > 0:
                img_inpainted = img_inpainted[:-pad_bottom]
            if pad_right > 0:
                img_inpainted = img_inpainted[:, :-pad_right]
            new_shape = img_inpainted.shape[:2]
            if new_shape[0] != im_h or new_shape[1] != im_w:
                img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            curr_img = img_inpainted * mask_original + img_original * (1 - mask_original)
            LOGGER.info(f"[{self.name}] Pass {p_idx + 1}/{passes} completed successfully.")

        return curr_img

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])

        elif param_key == 'inpaint_passes':
            self.inpaint_passes = int(self.params['inpaint_passes']['value'])

        elif param_key == 'precision':
            precision = self.params['precision']['value']
            self.precision = precision

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device
        if precision is not None:
            self.precision = precision

@register_inpainter('inpaint-anything')
@register_inpainter('big-lama')
@register_inpainter('lama_large_512px')
class LamaLarge(LamaInpainterMPE):

    inpaint_by_block = False
    check_need_inpaint = False

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                512,
                768,
                1024,
                1536, 
                2048,
                2560
            ], 
            'value': 2048,
        },
        'inpaint_passes': {
            'type': 'selector',
            'options': [
                1,
                2,
                3
            ], 
            'value': 2,
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
        'precision': {
            'type': 'selector',
            'options': [
                'fp32',
                'bf16'
            ], 
            'value': 'bf16' if BF16_SUPPORTED == 'cuda' else 'fp32'
        }, 
    }

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
            'sha256_pre_calculated': '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935',
            'files': 'data/models/lama_large_512px.ckpt',
    }]

    def _load_model(self):
        device = self.params['device']['value']
        precision = self.params['precision']['value']
        ckpt_path = r'data/models/lama_large_512px.ckpt'

        if not os.path.exists(ckpt_path):
            LOGGER.info(f"📥 [Auto-Download] Chưa tìm thấy file trọng số LaMa, đang tải tự động một lần duy nhất từ HuggingFace...")
            from utils.download_util import download_and_check_files
            for download_kwargs in self.download_file_list:
                download_and_check_files(**download_kwargs)

        self.model = load_lama_mpe(ckpt_path, device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(device, precision=precision)



FLUX_MODEL_MAPPER = {
    '4b-Q4_K_M': 'black-forest-labs/FLUX.2-klein-4B'
}

@register_inpainter('flux2-klein')
class Flux2Klein(InpainterBase):

    params = {
        'model': {
            'type': 'selector',
            'options': [
                '4b-Q4_K_M', 
            ], 
            'value': '4b-Q4_K_M'
        },
        'max_resolution': {
            'type': 'selector',
            'options': [
                512,
                768,
                1024,
                1280,
                1536,
                2048
            ], 
            'value': 1536
        }, 
        'device': DEVICE_SELECTOR(),
        'step': 16
    }
    check_need_inpaint = False
    inpaint_by_block = False
    download_file_on_load = True

    download_file_list = [
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/transformer/config.json',
                'files': 'data/models/flux-2-klein-4b/transformer/config.json',
            },
            # {
            #     'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/transformer/diffusion_pytorch_model.safetensors',
            #     'files': 'data/models/flux-2-klein-4b/transformer/diffusion_pytorch_model.safetensors',
            #     'sha256_pre_calculated': '9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e'
            # },
            {
                'url': 'https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf',
                'files': 'data/models/flux-2-klein-4b-Q4_K_M.gguf',
                'sha256_pre_calculated': '0b25d143c8469b342bc5af3bce92b783bf6b0636d285f7b2f75e38af63af9a15'
            },
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/config.json',
                'files': 'data/models/flux-2-vae/config.json',
            },
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors',
                'files': 'data/models/flux-2-vae/diffusion_pytorch_model.safetensors',
                'sha256_pre_calculated': 'ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04'
            },
            {
                'url': 'https://huggingface.co/dreMaz/flux2-klein-inpaint/resolve/main/flux2_inpaint_prompt.safetensors',
                'files': 'data/models/flux2_inpaint_prompt.safetensors',
                'sha256_pre_calculated': '7d7b19ec266581cb1faa51ad92f49a302932b0c589feae633f97da2d925cb6a4'
            }
        ]

    _load_model_keys = {'pipeline'}

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _load_model(self):
        
        from modules.inpaint.flux_inpaint_pipeline import Flux2KleinInpaintPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2
        from safetensors.torch import load_file
        from diffusers import GGUFQuantizationConfig

        model_type = self.get_param_value('model')
        source = FLUX_MODEL_MAPPER[model_type]

        # transformer = Flux2Transformer2DModel.from_pretrained(f'data/models/flux-2-klein-{model_type}/transformer')

        transformer = Flux2Transformer2DModel.from_single_file(
            "data/models/flux-2-klein-4b-Q4_K_M.gguf",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config='data/models/flux-2-klein-4b/transformer/config.json'
        )
        self.prompt_embeds = load_file('data/models/flux2_inpaint_prompt.safetensors')['prompt_embeds'].to(dtype=torch.bfloat16, device=self.get_param_value('device'))

        vae = AutoencoderKLFlux2.from_pretrained(f'data/models/flux-2-vae').to(device=self.get_param_value('device'), dtype=torch.bfloat16)
        pipeline = Flux2KleinInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=source,
            text_encoder=None,
            tokenizer=None,
            vae=vae,
            transformer=transformer
        )
        self.pipeline = pipeline.to(device=self.get_param_value('device'), )


    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        max_resolution = self.get_param_value('max_resolution')
        div = 16
        mask_original = (mask > 127)[..., None].astype(np.uint8)
        img_original = img.copy()

        input_sz = img.shape[:2]

        h, w = input_sz
        th, tw = h, w
        resize_ratio = max_resolution / max(th, tw)
        if resize_ratio < 1:
            th, tw = int(round(resize_ratio * th)), int(round(resize_ratio * tw))
        th = int(round(th / div)) * div
        tw = int(round(tw / div)) * div
        img = smart_resize(img, (th, tw))
        mask = smart_resize(mask, (th, tw))

        rst = self.pipeline(
            image=img,
            mask=mask,
            prompt_embeds=self.prompt_embeds,
            height=img.shape[0],
            width=img.shape[1],
            num_inference_steps=self.get_param_value('step'),
            guidance_scale=1, return_dict=False, output_type='numpy'
        )
        img_inpainted = (np.round(rst[0] * 255)).astype(np.uint8)
        img_inpainted = smart_resize(img_inpainted, img_original.shape[:2])
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted
    

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if hasattr(self, 'pipeline'):
            if param_key == 'device':
                param_device = self.get_param_value('device')
                self.pipeline.to(device=param_device)