import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from utils.textblock_mask import _is_gradient_bg

CTD_ONNX_PATH = 'data/models/comictextdetector.pt.onnx'
CTD_TORCH_PATH = 'data/models/comictextdetector.pt'

def load_ctd_model(model_path, device, detect_size=1024, conf_thresh=0.20, text_thresh=0.25, link_thresh=0.20, low_text=0.15, min_area=16):
    from .ctd import CTDModel
    model = CTDModel(model_path, detect_size=detect_size, device=device, conf_thresh=conf_thresh, text_thresh=text_thresh, link_thresh=link_thresh, low_text=low_text, min_area=min_area)
    return model

@register_textdetectors('ctd')
class ComicTextDetector(TextDetectorBase):

    params = {
        'detect_size': {
            'type': 'selector',
            'options': [896, 1024, 1152, 1280, 1536, 1792, 2048], 
            'value': 1536
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'value': 4
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector',
        'font size multiplier': 1.,
        'font size max': -1,
        'font size min': -1,
        'mask dilate size': 4,
        'conf_thresh': 0.20,
        'text_threshold': 0.25,
        'link_threshold': 0.20,
        'low_text': 0.15,
        'min_area': 16,
        'text_thresh': 0.25,
        'link_thresh': 0.20,
    }
    _load_model_keys = {'model'}
    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/',
        'files': ['data/models/comictextdetector.pt', 'data/models/comictextdetector.pt.onnx'],
        'sha256_pre_calculated': ['1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb', '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'],
        'concatenate_url_filename': 2,
    }]

    device = DEFAULT_DEVICE
    detect_size = 1536
    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model: CTDModel = None

    @property
    def device(self):
        return self.params['device']['value']
    
    @property
    def detect_size(self):
        return int(self.params['detect_size']['value'])

    def _load_model(self):
        conf_val = self.get_param_value('conf_thresh')
        conf_thresh = float(conf_val) if conf_val is not None else 0.20
        text_val = self.get_param_value('text_threshold') or self.get_param_value('text_thresh')
        text_thresh = float(text_val) if text_val is not None else 0.25
        link_val = self.get_param_value('link_threshold') or self.get_param_value('link_thresh')
        link_thresh = float(link_val) if link_val is not None else 0.20
        low_val = self.get_param_value('low_text')
        low_text = float(low_val) if low_val is not None else 0.15
        min_area_val = self.get_param_value('min_area')
        min_area = int(min_area_val) if min_area_val is not None else 16
        if self.device != 'cpu':
            self.model = load_ctd_model(CTD_TORCH_PATH, self.device, self.detect_size, conf_thresh=conf_thresh, text_thresh=text_thresh, link_thresh=link_thresh, low_text=low_text, min_area=min_area)
        else:
            self.model = load_ctd_model(CTD_ONNX_PATH, self.device, self.detect_size, conf_thresh=conf_thresh, text_thresh=text_thresh, link_thresh=link_thresh, low_text=low_text, min_area=min_area)

    def _detect(self, img: np.ndarray, proj: ProjImgTrans) -> Tuple[np.ndarray, List[TextBlock]]:
        im_h, im_w = img.shape[:2]
        max_dim = max(im_h, im_w)
        if max_dim >= 1440:
            # 2K High-DPI optimization: maintain detect_size = 1536 without aspect-ratio downscaling
            if self.model is not None and getattr(self.model, 'detect_size', 0) < 1536:
                self.model.detect_size = 1536

        _, mask, blk_list = self.model(img)

        # Classify balloon vs free-floating/narration based on surrounding background
        for blk in blk_list:
            if getattr(blk, 'is_balloon', None) is None:
                bx1, by1, bx2, by2 = [int(v) for v in blk.xyxy]
                rx1, ry1 = max(0, bx1 - 10), max(0, by1 - 10)
                rx2, ry2 = min(im_w, bx2 + 10), min(im_h, by2 + 10)
                crop_border = img[ry1:ry2, rx1:rx2]
                if crop_border.size > 0:
                    if crop_border.ndim == 2:
                        gray_b = crop_border
                    elif crop_border.shape[-1] == 4:
                        gray_b = cv2.cvtColor(crop_border, cv2.COLOR_RGBA2GRAY)
                    else:
                        gray_b = cv2.cvtColor(crop_border, cv2.COLOR_RGB2GRAY)
                    border_px = np.concatenate([gray_b[0, :], gray_b[-1, :], gray_b[:, 0], gray_b[:, -1]])
                    blk.is_balloon = bool(np.mean(border_px) > 200 and np.std(border_px) < 30)
        
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        base_ksize = self.get_param_value('mask dilate size')
        base_ksize = max(1, base_ksize) if base_ksize is not None else 4

        # 1. Base morphological cleanup
        close_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_elem)

        # 2. Adaptive Block-Level Dilation, Convex Hull Bypass & Stroke Halo Enhancement
        h_img, w_img = mask.shape[:2]
        final_mask = np.zeros_like(mask)

        for blk in blk_list:
            bx1, by1, bx2, by2 = [int(v) for v in blk.xyxy]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_img, bx2), min(h_img, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue

            bw = bx2 - bx1
            bh = by2 - by1
            fsize = getattr(blk, 'font_size', 16)
            is_balloon = getattr(blk, 'is_balloon', True)
            is_sfx_or_stroke = (
                not is_balloon 
                or getattr(blk, 'is_sfx', False) 
                or getattr(blk, 'stroke', False) 
                or fsize > 28 
                or max(bw, bh) > 150
            )

            # Dynamic Morphological Dilation Formula:
            # kernel_size = max(7, 2 * floor((box_height * 0.12) / 2) + 1)
            base_dynamic_k = max(7, 2 * int((float(bh) * 0.12) // 2) + 1)
            if is_sfx_or_stroke:
                font_h = max(float(fsize), float(bh) / max(1, len(getattr(blk, 'lines', [1]))))
                blk_ksize = max(base_dynamic_k, max(8, int(font_h * 0.25)))
                iterations = 2
            else:
                blk_ksize = max(base_dynamic_k, max(3, base_ksize))
                iterations = 2

            # Expand sub-window for local dilation to prevent border clipping
            pad = blk_ksize * iterations + 12
            px1, py1 = max(0, bx1 - pad), max(0, by1 - pad)
            px2, py2 = min(w_img, bx2 + pad), min(h_img, by2 + pad)

            sub_mask = mask[py1:py2, px1:px2].copy()

            # Task A.1: Lớp Lọc Otsu/Adaptive Thresholding Tăng Cường Viền (Outer-stroke detection with 12px padding)
            gray_roi = None
            rx1, ry1 = max(0, bx1 - 12), max(0, by1 - 12)
            rx2, ry2 = min(w_img, bx2 + 12), min(h_img, by2 + 12)
            roi_crop = img[ry1:ry2, rx1:rx2]
            if roi_crop.size > 0:
                gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_RGB2GRAY) if roi_crop.ndim == 3 else roi_crop.copy()
                grad_x = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = cv2.magnitude(grad_x, grad_y)
                border_std = float(np.std(grad_mag))

                # If text has high outer-stroke edge amplitude, merge outer contour into mask
                if border_std > 25.0 or getattr(blk, 'stroke', False) or getattr(blk, 'is_sfx', False):
                    thresh_adaptive = cv2.adaptiveThreshold(
                        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                    )
                    oy1, oy2 = max(py1, ry1), min(py2, ry2)
                    ox1, ox2 = max(px1, rx1), min(px2, rx2)
                    if oy2 > oy1 and ox2 > ox1:
                        target_roi = thresh_adaptive[oy1-ry1:oy2-ry1, ox1-rx1:ox2-rx1]
                        current_sub = sub_mask[oy1-py1:oy2-py1, ox1-px1:ox2-px1]
                        # Retain outer strokes connected or adjacent to current mask
                        dilated_core = cv2.dilate(current_sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                        stroke_part = cv2.bitwise_and(target_roi, dilated_core)
                        sub_mask[oy1-py1:oy2-py1, ox1-px1:ox2-px1] = cv2.bitwise_or(current_sub, stroke_part)

            # Task A.3: Chế Độ Balloon Solid Fill Dự Phòng (Convex Hull Bypass)
            # If is_balloon == True: if character area > 35% of bubble interior, use Convex Hull
            if getattr(blk, 'is_balloon', True) and not _is_gradient_bg(gray_roi):
                interior_mask = sub_mask[by1-py1:by2-py1, bx1-px1:bx2-px1]
                interior_area = float(bw * bh)
                if interior_area > 0 and interior_mask.size > 0:
                    text_density = float(np.count_nonzero(interior_mask > 0)) / interior_area
                    if text_density > 0.35:
                        pts = cv2.findNonZero(interior_mask)
                        if pts is not None and len(pts) >= 3:
                            hull = cv2.convexHull(pts)
                            cv2.fillConvexPoly(interior_mask, hull, 255)
                            sub_mask[by1-py1:by2-py1, bx1-px1:bx2-px1] = interior_mask

            # Solid mask fill for small text, whispers, and furigana lines
            if fsize < 16 or min(bw, bh) < 24:
                if hasattr(blk, 'lines') and blk.lines is not None and len(blk.lines) > 0:
                    for line in blk.lines:
                        try:
                            pts = (np.array(line, dtype=np.int32).reshape((-1, 2)) - np.array([px1, py1])).astype(np.int32)
                            if len(pts) >= 3:
                                cv2.fillPoly(sub_mask, [pts], 255)
                            elif len(pts) == 2:
                                cv2.rectangle(sub_mask, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), 255, -1)
                        except Exception:
                            pass
                else:
                    cv2.rectangle(sub_mask, (int(bx1 - px1), int(by1 - py1)), (int(bx2 - px1), int(by2 - py1)), 255, -1)

            # Apply Task A.2 Dynamic Morphological Dilation with cv2.MORPH_ELLIPSE and iterations=2
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blk_ksize, blk_ksize))
            sub_dilated = cv2.dilate(sub_mask, element, iterations=iterations)

            # Merge into final mask
            final_mask[py1:py2, px1:px2] = np.maximum(final_mask[py1:py2, px1:px2], sub_dilated)

        # Include any remaining global mask areas with standard dilation
        if base_ksize > 0:
            element_global = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * base_ksize + 1, 2 * base_ksize + 1))
            dilated_global = cv2.dilate(mask, element_global, iterations=2)
            final_mask = np.maximum(final_mask, dilated_global)

        return final_mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.device
        if self.model is not None:
            if self.model.device != device:
                self.model.device = device
                if device != 'cpu':
                    self.model.load_model(CTD_TORCH_PATH)
                else:
                    self.model.load_model(CTD_ONNX_PATH)
            self.model.detect_size = self.detect_size