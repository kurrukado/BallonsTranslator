import time
from typing import Union, List, Dict, Callable
import os.path as osp

import numpy as np
from qtpy.QtCore import QThread, Signal, QObject, QLocale, QTimer
from qtpy.QtWidgets import QFileDialog
from sympy import true

from .funcmaps import get_maskseg_method
from utils.logger import logger as LOGGER
from utils.registry import Registry
from utils.imgproc_utils import enlarge_window, get_block_mask
from utils.io_utils import imread, text_is_empty
from modules.translators import MissingTranslatorParams
from modules.base import BaseModule, soft_empty_cache
from modules import INPAINTERS, TRANSLATORS, TEXTDETECTORS, OCR, \
    GET_VALID_TRANSLATORS, GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_OCR, \
    BaseTranslator, InpainterBase, TextDetectorBase, OCRBase, merge_config_module_params
import modules
modules.translators.SYSTEM_LANG = QLocale.system().name()
from utils.textblock import TextBlock, sort_regions
from utils import shared
from utils.message import create_error_dialog, create_info_dialog
from .custom_widget import ImgtransProgressMessageBox, ParamComboBox
from .configpanel import ConfigPanel
from utils.proj_imgtrans import ProjImgTrans
from utils.config import pcfg, RunStatus
cfg_module = pcfg.module


class ModuleThread(QThread):

    finish_set_module = Signal()
    _failed_set_module_msg = 'Failed to set module.'
    module_thread_stopped = Signal()

    def __init__(self, module_key: str, MODULE_REGISTER: Registry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job = None
        self.module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = None
        self.module_register = MODULE_REGISTER
        self.module_key = module_key

        self.pipeline_pagekey_queue = []
        self.finished_counter = 0
        self.num_process_pages = 0
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_requested = False

    def _set_module(self, module_name: str):
        old_module = self.module
        try:
            module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] \
                = self.module_register.module_dict[module_name]
            params = cfg_module.get_params(self.module_key)[module_name]
            if params is not None:
                self.module = module(**params)
            else:
                self.module = module()
            if not pcfg.module.load_model_on_demand:
                self.module.load_model()
            if old_module is not None:
                del old_module
        except Exception as e:
            self.module = old_module
            create_error_dialog(e, self._failed_set_module_msg)

        self.finish_set_module.emit()

    def pipeline_finished(self):
        if self.imgtrans_proj is None:
            return True
        elif self.finished_counter >= self.num_process_pages:
            return True
        return False

    def initImgtransPipeline(self, proj: ProjImgTrans):
        if self.isRunning():
            self.terminate()
        self.imgtrans_proj = proj
        self.finished_counter = 0
        self.pipeline_pagekey_queue.clear()

    def requestStop(self):
        self.stop_requested = True

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None


class InpaintThread(ModuleThread):

    finish_inpaint = Signal(dict)
    inpainting = False    
    inpaint_failed = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('inpainter', INPAINTERS, *args, **kwargs)

    @property
    def inpainter(self) -> InpainterBase:
        return self.module

    def setInpainter(self, inpainter: str):
        self.job = lambda : self._set_module(inpainter)
        self.start()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        self.job = lambda : self._inpaint(img, mask, img_key, inpaint_rect)
        self.start()
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        inpaint_dict = {}
        self.inpainting = True
        try:
            inpainted = self.inpainter.inpaint(img, mask)
            inpaint_dict = {
                'inpainted': inpainted,
                'img': img,
                'mask': mask,
                'img_key': img_key,
                'inpaint_rect': inpaint_rect
            }
            self.finish_inpaint.emit(inpaint_dict)
        except Exception as e:
            create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
            self.inpainting = False
            self.inpaint_failed.emit()
        self.inpainting = False


class TextDetectThread(ModuleThread):
    
    finish_detect_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('textdetector', TEXTDETECTORS, *args, **kwargs)

    def setTextDetector(self, textdetector: str):
        self.job = lambda : self._set_module(textdetector)
        self.start()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.module


class OCRThread(ModuleThread):

    finish_ocr_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('ocr', OCR, *args, **kwargs)

    def setOCR(self, ocr: str):
        self.job = lambda : self._set_module(ocr)
        self.start()
    
    @property
    def ocr(self) -> OCRBase:
        return self.module


class TranslateThread(ModuleThread):

    finish_translate_page = Signal(str)
    progress_changed = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('translator', TRANSLATORS, *args, **kwargs)
        self.translator: BaseTranslator = self.module

    def _set_translator(self, translator: str):
        
        old_translator = self.translator
        source, target = cfg_module.translate_source, cfg_module.translate_target
        if self.translator is not None:
            if self.translator.name == translator:
                return
        
        try:
            params = cfg_module.translator_params[translator]
            translator_module: BaseTranslator = TRANSLATORS.module_dict[translator]
            if params is not None:
                self.translator = translator_module(source, target, raise_unsupported_lang=False, **params)
            else:
                self.translator = translator_module(source, target, raise_unsupported_lang=False)
            cfg_module.translate_source = self.translator.lang_source
            cfg_module.translate_target = self.translator.lang_target
            cfg_module.translator = self.translator.name
        except Exception as e:
            if old_translator is None:
                default_cls = TRANSLATORS.module_dict.get('LLM_API_Translator') or TRANSLATORS.module_dict.get('Copy Source') or next(iter(TRANSLATORS.module_dict.values()), None)
                if default_cls is not None:
                    old_translator = default_cls('English', 'Tiếng Việt', raise_unsupported_lang=False)
            self.translator = old_translator
            msg = self.tr('Failed to set translator ') + translator
            create_error_dialog(e, msg, 'FailedSetTranslator')

        self.module = self.translator
        self.finish_set_module.emit()

    def setTranslator(self, translator: str):
        self.job = lambda : self._set_translator(translator)
        self.start()

    def _extract_page_context(self, page_dict, page_key: str) -> dict:
        context = {}
        if hasattr(self, "imgtrans_proj") and self.imgtrans_proj is not None:
            try:
                curr_idx = self.imgtrans_proj.pagename2idx(page_key)
                if curr_idx > 0:
                    prev_page_name = self.imgtrans_proj.idx2pagename(curr_idx - 1)
                    if prev_page_name in self.imgtrans_proj.pages:
                        prev_blks = self.imgtrans_proj.pages[prev_page_name]
                        from modules.translators.context_engine import DialogueItem, CharacterProfile, CharacterMemory, GlossaryManager
                        history = []
                        for blk in prev_blks:
                            txt = blk.get_text() if hasattr(blk, "get_text") else str(blk)
                            if txt.strip():
                                history.append(
                                    DialogueItem(
                                        id=len(history) + 1,
                                        source=txt,
                                        translated=getattr(blk, "translation", None),
                                        speaker=getattr(blk, "speaker", None),
                                    )
                                )
                        if history:
                            context["dialogue_history"] = history
                if hasattr(self.imgtrans_proj, "img_array") and self.imgtrans_proj.img_array is not None:
                    context["img_shape"] = self.imgtrans_proj.img_array.shape[:2]
            except Exception:
                pass
        return context

    def _translate_page(self, page_dict, page_key: str, emit_finished=True):
        page = page_dict[page_key]
        page_context = self._extract_page_context(page_dict, page_key)
        try:
            self.translator.translate_textblk_lst(page, page_context=page_context)
        except Exception as e:
            create_error_dialog(e, self.tr('Translation Failed.'), 'TranslationFailed')
        if emit_finished:
            self.finish_translate_page.emit(page_key)

    def translatePage(self, page_dict, page_key: str):
        self.job = lambda: self._translate_page(page_dict, page_key)
        self.start()

    def push_pagekey_queue(self, page_key: str):
        self.pipeline_pagekey_queue.append(page_key)

    def runTranslatePipeline(self, imgtrans_proj: ProjImgTrans):
        self.initImgtransPipeline(imgtrans_proj)
        self.job = self._run_translate_pipeline
        self.start()


    def _run_translate_pipeline(self):
        delay = self.translator.delay()

        while not self.pipeline_finished():
            if self.stop_requested:
                self.module_thread_stopped.emit()
                self.stop_requested = False
                break

            if len(self.pipeline_pagekey_queue) == 0:
                time.sleep(0.1)
                continue
            
            page_key = self.pipeline_pagekey_queue.pop(0)
            LOGGER.info(f"🌐 [Translate] Bắt đầu tiến trình dịch trang '{page_key}'...")
            trans_success = True
            try:
                self._translate_page(self.imgtrans_proj.pages, page_key, emit_finished=False)
                LOGGER.info(f"✓ [Translate] Hoàn tất dịch xong toàn bộ khối thoại trên trang '{page_key}'.")
            except Exception as e:
                trans_success = False
                LOGGER.error(f"❌ [Translate] Lỗi dịch thuật trên trang '{page_key}': {e}")
                msg = self.tr('Translation Failed.')
                if isinstance(e, MissingTranslatorParams):
                    msg = msg + '\n' + str(e) + self.tr(' is required for ' + self.translator.name)
                create_error_dialog(e, msg, 'TranslationFailed')
            self.finished_counter += 1
            if trans_success:
                self.imgtrans_proj.update_page_progress(page_key, RunStatus.FIN_TRANSLATE)
            self.progress_changed.emit(self.finished_counter)

            if not self.pipeline_finished() and delay > 0:
                time.sleep(delay)


class ImgtransThread(QThread):

    pipeline_stopped = Signal()
    pipeline_finished = Signal()
    page_trans_finished = Signal(int)
    update_detect_progress = Signal(int)
    update_ocr_progress = Signal(int)
    update_translate_progress = Signal(int)
    update_inpaint_progress = Signal(int)

    finish_blktrans_stage = Signal(str, int)
    finish_blktrans = Signal(int, list)
    unload_modules = Signal(list)

    detect_counter = 0
    ocr_counter = 0
    translate_counter = 0
    inpaint_counter = 0

    def __init__(self, 
                 textdetect_thread: TextDetectThread,
                 ocr_thread: OCRThread,
                 translate_thread: TranslateThread,
                 inpaint_thread: InpaintThread,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.textdetect_thread = textdetect_thread
        self.ocr_thread = ocr_thread
        self.translate_thread = translate_thread
        self.translate_thread.module_thread_stopped.connect(self.on_module_thread_stopped)
        self.inpaint_thread = inpaint_thread
        self.job = None
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_requested = False
        self.active_job_id: str = None
        self.pages_to_process = None  # 需要处理的页面列表（用于继续运行模式）
        self.process_idx_to_page_idx = {}

    def on_module_thread_stopped(self):
        while True:
            # might freeze UI
            if self.translate_thread.isRunning() or self.inpaint_thread.isRunning() or self.ocr_thread.isRunning() or self.textdetect_thread.isRunning():
                time.sleep(0.05)
                continue
            break

        self.pipeline_stopped.emit()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr
    
    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    def runImgtransPipeline(self, imgtrans_proj: ProjImgTrans, pages_to_process=None, job_id: str = None):
        import uuid
        self.imgtrans_proj = imgtrans_proj
        self.pages_to_process = pages_to_process  # 保存需要处理的页面列表
        self.num_pages = len(self.imgtrans_proj.pages)
        self.stop_requested = False
        self.active_job_id = job_id or uuid.uuid4().hex
        # 创建处理索引到实际页面索引的映射
        self.process_idx_to_page_idx = {}
        self.job = self._imgtrans_pipeline
        self.start()
    
    def requestStop(self):
        """请求停止当前任务"""
        if self.isRunning():
            self.stop_requested = True
        # 同时停止翻译线程
        if self.translate_thread.isRunning():
            self.translate_thread.requestStop()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.job = lambda : self._blktrans_pipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)
        self.start()

    def _blktrans_pipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        if mode >= 0 and mode < 3:
            try:
                self.ocr_thread.module.run_ocr(tgt_img, blk_list, split_textblk=True)
            except Exception as e:
                create_error_dialog(e, self.tr('OCR Failed.'), 'OCRFailed')
            self.finish_blktrans.emit(mode, blk_ids)

        if mode != 0 and mode < 3:
            # Luồng B (Mode 2): Direct single bubble passthrough for manual UI edits
            if len(blk_list) == 1 and hasattr(self.translate_thread.module, "translate_single"):
                blk = blk_list[0]
                src_txt = blk.get_text() if hasattr(blk, "get_text") else str(blk)
                if src_txt and src_txt.strip():
                    context_hints = None
                    try:
                        all_page_blks = self.imgtrans_proj.current_block_list() if self.imgtrans_proj else []
                        if blk_ids and len(blk_ids) > 0:
                            target_idx = blk_ids[0]
                            prev_txt = all_page_blks[target_idx - 1].get_text() if (0 < target_idx < len(all_page_blks)) else ""
                            next_txt = all_page_blks[target_idx + 1].get_text() if (target_idx + 1 < len(all_page_blks)) else ""
                            context_hints = {
                                "previous": prev_txt,
                                "next": next_txt,
                                "block_type": "DIALOGUE" if getattr(blk, "is_balloon", True) else "NARRATION",
                            }
                    except Exception:
                        pass
                    blk.translation = self.translate_thread.module.translate_single(
                        src_txt, cfg_module.translate_source, cfg_module.translate_target, context_hints=context_hints
                    )
            else:
                self.translate_thread.module.translate_textblk_lst(blk_list)
            self.finish_blktrans.emit(mode, blk_ids)
        if mode > 1:
            im_h, im_w = tgt_img.shape[:2]
            progress_prod = 100. / len(blk_list) if len(blk_list) > 0 else 0
            for ii, blk in enumerate(blk_list):
                xyxy = enlarge_window(blk.xyxy, im_w, im_h)
                xyxy = np.array(xyxy)
                x1, y1, x2, y2 = xyxy.astype(np.int64)
                blk.region_inpaint_dict = None
                if y2 - y1 > 2 and x2 - x1 > 2:
                    im = np.copy(tgt_img[y1: y2, x1: x2])
                    maskseg_method = get_maskseg_method()
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=tgt_mask[y1: y2, x1: x2])
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask.sum() > 0:
                        inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                        blk.region_inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2], 'inpainted': inpainted}
                    self.finish_blktrans_stage.emit('inpaint', int((ii+1) * progress_prod))
        self.finish_blktrans.emit(mode, blk_ids)

    def _imgtrans_pipeline(self):
        self.detect_counter = 0
        self.ocr_counter = 0
        self.translate_counter = 0
        self.inpaint_counter = 0
        
        # 如果指定了pages_to_process，只处理这些页面
        all_pages = list(self.imgtrans_proj.pages.keys())
        if self.pages_to_process is not None and len(self.pages_to_process) > 0:
            pages_to_iterate = self.pages_to_process
            self.num_pages = num_pages = len(self.pages_to_process)
            # 建立处理索引到实际页面索引的映射
            for process_idx, page_name in enumerate(pages_to_iterate):
                if page_name in all_pages:
                    self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
            LOGGER.info(f'Processing specific pages: {len(pages_to_iterate)} pages')
        else:
            pages_to_iterate = all_pages
            self.num_pages = num_pages = len(self.imgtrans_proj.pages)
            # 处理索引等于实际页面索引
            for i in range(num_pages):
                self.process_idx_to_page_idx[i] = i
            LOGGER.info(f'Processing all {num_pages} pages')
        self.textdetect_thread.num_process_pages = self.num_pages
        self.ocr_thread.num_process_pages = self.num_pages
        self.inpaint_thread.num_process_pages = self.num_pages
        self.translate_thread.num_process_pages = self.num_pages

        low_vram_trans = False
        use_chapter_batch = False
        translation_proxy = None
        if self.translator is not None:
            low_vram_trans = self.translator.low_vram_mode
            use_chapter_batch = cfg_module.enable_translate and hasattr(self.translator, "translate_chapter_batch")
            if use_chapter_batch:
                self.parallel_trans = False
                from modules.translators.translation_proxy import TranslationProxy
                translation_proxy = TranslationProxy()
                session_page_indices = [self.imgtrans_proj.pagename2idx(p) for p in pages_to_iterate]
                session_page_names = {self.imgtrans_proj.pagename2idx(p): p for p in pages_to_iterate}
                translation_proxy.start_session(session_page_indices, page_names=session_page_names, timeout_seconds=120.0)
            else:
                self.parallel_trans = not self.translator.is_computational_intensive() and not low_vram_trans
        else:
            self.parallel_trans = False
        if self.parallel_trans and cfg_module.enable_translate:
            self.translate_thread.runTranslatePipeline(self.imgtrans_proj)

        current_job_id = self.active_job_id or "job_default"
        LOGGER.info(f"🚀 [JOB {current_job_id[:8]}] CREATED -> Processing {len(pages_to_iterate)} pages")

        for imgname in pages_to_iterate:
            
            # Check for cancellation or stale job ID
            if self.stop_requested or (self.active_job_id != current_job_id):
                LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded by new job (Active: {self.active_job_id[:8] if self.active_job_id else 'None'})")
                if translation_proxy:
                    translation_proxy.cancel_session()
                self.pipeline_stopped.emit()
                return
                
            LOGGER.info(f"🚀 [JOB {current_job_id[:8]}] [Bắt Đầu] Đang xử lý trang truyện '{imgname}'...")
            img = self.imgtrans_proj.read_img(imgname)
            mask = blk_list = None
            need_save_mask = False
            blk_removed: List[TextBlock] = []
            if cfg_module.enable_detect:
                LOGGER.info(f"🔍 [JOB {current_job_id[:8]}] DETECTING: Đang quét phát hiện bong bóng thoại & khối chữ...")
                try:
                    mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                    need_save_mask = True
                    LOGGER.info(f"✓ [JOB {current_job_id[:8]}] DETECTED: Phát hiện thành công {len(blk_list)} khối văn bản.")
                except Exception as e:
                    create_error_dialog(e, self.tr('Text Detection Failed.'), 'TextDetectFailed')
                    blk_list = []
                self.detect_counter += 1
                if pcfg.module.keep_exist_textlines:
                    blk_list = self.imgtrans_proj.pages[imgname] + blk_list
                    blk_list = sort_regions(blk_list)
                    existed_mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    if existed_mask is not None:
                        mask = np.bitwise_or(mask, existed_mask)
                self.imgtrans_proj.pages[imgname] = list(blk_list)

                if mask is not None and not cfg_module.enable_ocr:
                    self.imgtrans_proj.save_mask(imgname, mask)
                    need_save_mask = False
                    
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_DET)
                self.update_detect_progress.emit(self.detect_counter)

            if blk_list is None:
                blk_list = list(self.imgtrans_proj.pages[imgname]) if imgname in self.imgtrans_proj.pages else []

            detected_count = len(blk_list) if blk_list is not None else (len(self.imgtrans_proj.pages[imgname]) if imgname in self.imgtrans_proj.pages else 0)

            # Check for cancellation before OCR
            if self.stop_requested or (self.active_job_id != current_job_id):
                LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded before OCR on '{imgname}'")
                if translation_proxy:
                    translation_proxy.cancel_session()
                self.pipeline_stopped.emit()
                return

            if cfg_module.enable_ocr:
                LOGGER.info(f"📖 [JOB {current_job_id[:8]}] OCR_RUNNING: Đang nhận diện ký tự ({len(blk_list)} khối)...")
                try:
                    self.ocr.run_ocr(img, blk_list)

                    # Check for cancellation during/after OCR
                    if self.stop_requested or (self.active_job_id != current_job_id):
                        LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded during OCR on '{imgname}'")
                        if translation_proxy:
                            translation_proxy.cancel_session()
                        self.pipeline_stopped.emit()
                        return

                    ocr_valid_count = sum(1 for b in blk_list if b.get_text() and b.get_text().strip())
                    LOGGER.info(f"✓ [JOB {current_job_id[:8]}] OCR_VALIDATED: Nhận diện hoàn tất {len(blk_list)}/{detected_count} khối thoại (Không rỗng: {ocr_valid_count}/{detected_count}).")
                    
                    # Hard assertion: All detected boxes must be retained through OCR
                    if len(blk_list) != detected_count:
                        LOGGER.error(f"❌ [OCR INTEGRITY ERROR] Detected {detected_count} boxes but OCR yielded {len(blk_list)} blocks on '{imgname}'!")
                        raise RuntimeError(f"OCR Integrity Error: Expected {detected_count} blocks, got {len(blk_list)} on '{imgname}'")
                except Exception as e:
                    if self.stop_requested or (self.active_job_id != current_job_id):
                        LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded on '{imgname}' (suppressing error dialog)")
                        if translation_proxy:
                            translation_proxy.cancel_session()
                        self.pipeline_stopped.emit()
                        return
                    create_error_dialog(e, self.tr('OCR Failed.'), 'OCRFailed')
                    if translation_proxy:
                        p_idx = self.imgtrans_proj.pagename2idx(imgname)
                        translation_proxy.mark_page_failed(p_idx, str(e))
                self.ocr_counter += 1

                if pcfg.restore_ocr_empty:
                    blk_list_updated = []
                    for blk in blk_list:
                        text = blk.get_text()
                        if text_is_empty(text):
                            blk_removed.append(blk)
                        else:
                            blk_list_updated.append(blk)

                    if len(blk_removed) > 0:
                        blk_list.clear()
                        blk_list += blk_list_updated
                        
                        if mask is None:
                            mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                        if mask is not None:
                            inpainted = None
                            if not cfg_module.enable_inpaint:
                                inpainted = self.imgtrans_proj.load_inpainted_by_imgname(imgname)
                            for blk in blk_removed:
                                xywh = blk.bounding_rect()
                                blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                x1, y1, x2, y2 = xyxy
                                if blk_mask is not None:
                                    mask[y1: y2, x1: x2] = 0
                                    if inpainted is not None:
                                        mskpnt = np.where(blk_mask)
                                        inpainted[y1: y2, x1: x2][mskpnt] = img[y1: y2, x1: x2][mskpnt]
                                    need_save_mask = True
                            if inpainted is not None and need_save_mask:
                                self.imgtrans_proj.save_inpainted(imgname, inpainted)
                            if need_save_mask:
                                self.imgtrans_proj.save_mask(imgname, mask)
                                need_save_mask = False

                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                self.update_ocr_progress.emit(self.ocr_counter)

            if need_save_mask and mask is not None:
                self.imgtrans_proj.save_mask(imgname, mask)
                need_save_mask = False

            # Check for cancellation before inpainting
            if self.stop_requested or (self.active_job_id != current_job_id):
                LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded before inpaint on '{imgname}'")
                if translation_proxy:
                    translation_proxy.cancel_session()
                self.pipeline_stopped.emit()
                return

            if cfg_module.enable_inpaint:
                LOGGER.info(f"🎨 [JOB {current_job_id[:8]}] INPAINTING: Đang xóa chữ và khôi phục nền ảnh...")
                if mask is None:
                    mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    
                if mask is not None:
                    try:
                        inpainted = self.inpainter.inpaint(img, mask, blk_list)
                        self.imgtrans_proj.save_inpainted(imgname, inpainted)
                        LOGGER.info(f"✓ [JOB {current_job_id[:8]}] INPAINT_VALIDATED: Khôi phục nền hoàn tất.")
                    except Exception as e:
                        if self.stop_requested or (self.active_job_id != current_job_id):
                            LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded during inpaint on '{imgname}'")
                            if translation_proxy:
                                translation_proxy.cancel_session()
                            self.pipeline_stopped.emit()
                            return
                        create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
                        if translation_proxy:
                            p_idx = self.imgtrans_proj.pagename2idx(imgname)
                            translation_proxy.mark_page_failed(p_idx, str(e))
                    
                self.inpaint_counter += 1
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
                self.update_inpaint_progress.emit(self.inpaint_counter)

            # Check for cancellation before translation
            if self.stop_requested or (self.active_job_id != current_job_id):
                LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Pipeline stopped or superseded before translation on '{imgname}'")
                if translation_proxy:
                    translation_proxy.cancel_session()
                self.pipeline_stopped.emit()
                return

            if cfg_module.enable_translate:
                if use_chapter_batch and translation_proxy is not None:
                    # Page Completion Barrier: Enqueue ONLY after OCR & Inpaint are complete
                    p_idx = self.imgtrans_proj.pagename2idx(imgname)
                    page_dialogues = []
                    for d_id, blk in enumerate(blk_list, start=1):
                        txt = blk.get_text() if hasattr(blk, "get_text") else str(blk)
                        if txt and txt.strip():
                            page_dialogues.append({
                                "id": d_id,
                                "text": txt.strip(),
                                "reading_order": d_id,
                                "block_type": "DIALOGUE" if getattr(blk, "is_balloon", True) else "NARRATION",
                            })
                    translation_proxy.mark_page_completed(p_idx, page_dialogues, page_name=imgname)
                    LOGGER.info(f"⏳ [JOB {current_job_id[:8]}] Đã đệm {len(page_dialogues)} câu thoại của trang '{imgname}' (chờ gộp chương).")
                else:
                    if self.parallel_trans:
                        self.translate_thread.push_pagekey_queue(imgname)
                    elif not low_vram_trans:
                        page_ctx = self.translate_thread._extract_page_context(self.imgtrans_proj.pages, imgname) if hasattr(self, "translate_thread") else None
                        self.translator.translate_textblk_lst(blk_list, page_context=page_ctx)
                        self.translate_counter += 1
                        self.update_translate_progress.emit(self.translate_counter)
                        LOGGER.info(f"✓ [JOB {current_job_id[:8]}] Đã dịch xong trang '{imgname}'.")
        
        # PHASE 2: CHAPTER BATCH TRANSLATION DISPATCH (Luồng A - Mode 1)
        if cfg_module.enable_translate and translation_proxy is not None and not translation_proxy.is_empty():
            if not self.stop_requested and self.active_job_id == current_job_id and not translation_proxy.is_cancelled():
                LOGGER.info(f"🌐 [JOB {current_job_id[:8]}] TRANSLATING: Đang gửi toàn bộ {translation_proxy.total_dialogues_count()} câu thoại của {len(pages_to_iterate)} trang tới Gemini...")
                if hasattr(self.translator, "translate_chapter_batch"):
                    try:
                        result_map = self.translator.translate_chapter_batch(
                            translation_proxy,
                            src_lang=cfg_module.translate_source,
                            tgt_lang=cfg_module.translate_target
                        )
                        # Unpack results into each page
                        for imgname in pages_to_iterate:
                            p_idx = self.imgtrans_proj.pagename2idx(imgname)
                            blk_list = self.imgtrans_proj.pages.get(imgname, [])
                            page_trans_map = result_map.get(p_idx, {})
                            for d_id, blk in enumerate(blk_list, start=1):
                                res_item = page_trans_map.get(d_id) or page_trans_map.get(str(d_id))
                                if res_item:
                                    blk.translation = res_item.get("translation", "")
                                    if hasattr(blk, "emotion_tag"):
                                        blk.emotion_tag = res_item.get("emotion_tag", "normal")
                            
                            self.translate_counter += 1
                            self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                            self.update_translate_progress.emit(self.translate_counter)
                            LOGGER.info(f"✓ [JOB {current_job_id[:8]}] TRANSLATED: Đã gán bản dịch nhất quán cho trang '{imgname}'.")
                    except Exception as e:
                        LOGGER.error(f"❌ [JOB {current_job_id[:8]}] [Chapter Batch Translation Error] {e}. Fallback to per-page translation...")
                        for imgname in pages_to_iterate:
                            blk_list = self.imgtrans_proj.pages.get(imgname, [])
                            page_ctx = self.translate_thread._extract_page_context(self.imgtrans_proj.pages, imgname) if hasattr(self, "translate_thread") else None
                            self.translator.translate_textblk_lst(blk_list, page_context=page_ctx)
                            self.translate_counter += 1
                            self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                            self.update_translate_progress.emit(self.translate_counter)
                else:
                    for imgname in pages_to_iterate:
                        blk_list = self.imgtrans_proj.pages.get(imgname, [])
                        page_ctx = self.translate_thread._extract_page_context(self.imgtrans_proj.pages, imgname) if hasattr(self, "translate_thread") else None
                        self.translator.translate_textblk_lst(blk_list, page_context=page_ctx)
                        self.translate_counter += 1
                        self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                        self.update_translate_progress.emit(self.translate_counter)

        # PHASE 3: FINAL ATOMIC VALIDATION & SYNCHRONOUS UI NOTIFICATION
        if not self.stop_requested and (self.active_job_id == current_job_id):
            LOGGER.info(f"✨ [JOB {current_job_id[:8]}] ATOMIC_COMMIT: Toàn bộ {len(pages_to_iterate)} trang đã hoàn tất mọi công đoạn và kiểm định hợp lệ.")
            for imgname in pages_to_iterate:
                if self.stop_requested or (self.active_job_id != current_job_id):
                    LOGGER.warning(f"🛑 [JOB {current_job_id[:8]}] Commit aborted due to cancellation or stale state.")
                    self.pipeline_stopped.emit()
                    return
                p_idx = self.imgtrans_proj.pagename2idx(imgname)
                self.page_trans_finished.emit(p_idx)
            if not self.stop_requested and (self.active_job_id == current_job_id):
                LOGGER.info(f"🎉 [JOB {current_job_id[:8]}] COMPLETED: Pipeline finished successfully.")
                self.pipeline_finished.emit()
        else:
            self.pipeline_stopped.emit()

    def detect_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.detect_counter == self.num_pages or not cfg_module.enable_detect

    def ocr_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.ocr_counter == self.num_pages or not cfg_module.enable_ocr

    def translate_finished(self) -> bool:
        if self.imgtrans_proj is None \
            or not cfg_module.enable_ocr \
            or not cfg_module.enable_translate:
            return True
        if self.parallel_trans:
            # 检查翻译计数器是否达到需要处理的页面数
            return self.translate_thread.finished_counter >= self.num_pages
        return self.translate_counter == self.num_pages or not cfg_module.enable_translate

    def inpaint_finished(self) -> bool:
        if self.imgtrans_proj is None or not cfg_module.enable_inpaint:
            return True
        return self.inpaint_counter == self.num_pages or not cfg_module.enable_inpaint

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None

    def recent_finished_index(self, ref_counter: int) -> int:
        if cfg_module.enable_detect:
            ref_counter = min(ref_counter, self.detect_counter)
        if cfg_module.enable_ocr:
            ref_counter = min(ref_counter, self.ocr_counter)
        if cfg_module.enable_inpaint:
            ref_counter = min(ref_counter, self.inpaint_counter)
        if cfg_module.enable_translate:
            if self.parallel_trans:
                ref_counter = min(ref_counter, self.translate_thread.finished_counter)
            else:
                ref_counter = min(ref_counter, self.translate_counter)

        process_idx = ref_counter - 1
        # 将处理索引转换为实际页面索引
        if hasattr(self, 'process_idx_to_page_idx') and process_idx in self.process_idx_to_page_idx:
            return self.process_idx_to_page_idx[process_idx]
        return process_idx


def unload_modules(self, module_names):
    model_deleted = False
    for module in module_names:
        module: BaseModule = getattr(self, module)
        model_deleted = model_deleted or module.unload_model()
    if model_deleted:
        soft_empty_cache()


class ModuleManager(QObject):
    imgtrans_proj: ProjImgTrans = None

    finish_translate_page = Signal(str)
    canvas_inpaint_finished = Signal(dict)
    inpaint_th_finished = Signal()

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int, list)
    page_trans_finished = Signal(int)

    run_canvas_inpaint = False
    is_waiting_th = False
    block_set_inpainter = False

    def __init__(self, 
                 imgtrans_proj: ProjImgTrans,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imgtrans_proj = imgtrans_proj
        self.check_inpaint_fin_timer = QTimer(self)
        self.check_inpaint_fin_timer.timeout.connect(self.check_inpaint_th_finished)

    def setupThread(self, config_panel: ConfigPanel, imgtrans_progress_msgbox: ImgtransProgressMessageBox, ocr_postprocess: Callable = None, translate_preprocess: Callable = None, translate_postprocess: Callable = None):
        self.textdetect_thread = TextDetectThread()

        self.ocr_thread = OCRThread()
        
        self.translate_thread = TranslateThread()
        self.translate_thread.progress_changed.connect(self.on_update_translate_progress)
        self.translate_thread.finish_translate_page.connect(self.on_finish_translate_page)  

        self.inpaint_thread = InpaintThread()
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)

        self.progress_msgbox = imgtrans_progress_msgbox
        self.progress_msgbox.stop_clicked.connect(self.stopImgtransPipeline)

        self.imgtrans_thread = ImgtransThread(self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread)
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_translate_progress.connect(self.on_update_translate_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)
        self.imgtrans_thread.finish_blktrans.connect(self.on_finish_blktrans)
        self.imgtrans_thread.page_trans_finished.connect(self.page_trans_finished)
        self.imgtrans_thread.pipeline_finished.connect(self.on_pipeline_finished)
        self.imgtrans_thread.pipeline_stopped.connect(self.on_imgtrans_thread_stopped)

        self.translator_panel = translator_panel = config_panel.trans_config_panel        
        translator_params = merge_config_module_params(cfg_module.translator_params, GET_VALID_TRANSLATORS(), TRANSLATORS.get)
        translator_panel.addModulesParamWidgets(translator_params)
        translator_panel.translator_changed.connect(self.setTranslator)
        translator_panel.paramwidget_edited.connect(self.on_translatorparam_edited)
        translator_panel.translateByTextblockBox.checker_changed.connect(self.on_translatebyblock_checker_changed)
        translator_panel.translateByTextblockBox.checker.setChecked(cfg_module.translate_by_textblock)

        from modules.translators.hooks import chs2cht
        BaseTranslator.register_preprocess_hooks({'keyword_sub': translate_preprocess})
        BaseTranslator.register_postprocess_hooks({'chs2cht': chs2cht, 'keyword_sub': translate_postprocess})

        self.inpaint_panel = inpainter_panel = config_panel.inpaint_config_panel
        inpainter_params = merge_config_module_params(cfg_module.inpainter_params, GET_VALID_INPAINTERS(), INPAINTERS.get)
        inpainter_panel.addModulesParamWidgets(inpainter_params)
        inpainter_panel.paramwidget_edited.connect(self.on_inpainterparam_edited)
        inpainter_panel.inpainter_changed.connect(self.setInpainter)
        inpainter_panel.needInpaintChecker.checker_changed.connect(self.on_inpainter_checker_changed)
        inpainter_panel.needInpaintChecker.checker.setChecked(cfg_module.check_need_inpaint)

        self.textdetect_panel = textdetector_panel = config_panel.detect_config_panel
        textdetector_params = merge_config_module_params(cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)
        textdetector_panel.addModulesParamWidgets(textdetector_params)
        textdetector_panel.paramwidget_edited.connect(self.on_textdetectorparam_edited)
        textdetector_panel.detector_changed.connect(self.setTextDetector)

        self.ocr_panel = ocr_panel = config_panel.ocr_config_panel
        ocr_params = merge_config_module_params(cfg_module.ocr_params, GET_VALID_OCR(), OCR.get)
        ocr_panel.addModulesParamWidgets(ocr_params)
        ocr_panel.paramwidget_edited.connect(self.on_ocrparam_edited)
        ocr_panel.ocr_changed.connect(self.setOCR)
        OCRBase.register_postprocess_hooks(ocr_postprocess)

        config_panel.unload_models.connect(self.unload_all_models)


    def unload_all_models(self):
        unload_modules(self, {'textdetector', 'inpainter', 'ocr', 'translator'})

    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr

    def translatePage(self, run_target: bool, page_key: str):
        if not run_target:
            if self.translate_thread.isRunning():
                LOGGER.warning('Terminating a running translation thread.')
                self.translate_thread.terminate()
            return
        self.translate_thread.translatePage(self.imgtrans_proj.pages, page_key)

    def inpainterBusy(self):
        return self.inpaint_thread.isRunning()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect = None, **kwargs):
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Waiting for inpainting to finish')
            return
        self.inpaint_thread.inpaint(img, mask, img_key, inpaint_rect)

    def terminateRunningThread(self):
        if self.textdetect_thread.isRunning():
            self.textdetect_thread.quit()
        if self.ocr_thread.isRunning():
            self.ocr_thread.quit()
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.quit()
        if self.translate_thread.isRunning():
            self.translate_thread.quit()

    def check_inpaint_th_finished(self):
        if self.inpaint_thread.isRunning():
            return
        self.block_set_inpainter = False
        self.check_inpaint_fin_timer.stop()
        self.inpaint_th_finished.emit()

    def runImgtransPipeline(self, pages_to_process=None):
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        if cfg_module.all_stages_disabled() and self.imgtrans_proj is not None and self.imgtrans_proj.num_pages > 0:
            for ii in range(self.imgtrans_proj.num_pages):
                self.page_trans_finished.emit(ii)
            self.imgtrans_pipeline_finished.emit()
            return
        
        self.progress_msgbox.detect_bar.setVisible(cfg_module.enable_detect)
        self.progress_msgbox.ocr_bar.setVisible(cfg_module.enable_ocr)
        self.progress_msgbox.translate_bar.setVisible(cfg_module.enable_translate)
        self.progress_msgbox.inpaint_bar.setVisible(cfg_module.enable_inpaint)
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runImgtransPipeline(self.imgtrans_proj, pages_to_process)
    
    def stopImgtransPipeline(self):
        """停止图像翻译流程"""
        LOGGER.info('Stopping image translation pipeline...')
        self.imgtrans_thread.requestStop()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.terminateRunningThread()
        self.progress_msgbox.hide_all_bars()
        if mode >= 0 and mode < 3:
            self.progress_msgbox.ocr_bar.show()
        if mode >= 2:
            self.progress_msgbox.inpaint_bar.show()
        if mode != 0 and mode < 3:
            self.progress_msgbox.translate_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)

    def on_finish_blktrans_stage(self, stage: str, progress: int):
        if stage == 'ocr':
            self.progress_msgbox.updateOCRProgress(progress)
        elif stage == 'translate':
            self.progress_msgbox.updateTranslateProgress(progress)
        elif stage == 'inpaint':
            self.progress_msgbox.updateInpaintProgress(progress)
        else:
            raise NotImplementedError(f'Unknown stage: {stage}')
        
    def on_finish_blktrans(self, mode: int, blk_ids: List):
        self.blktrans_pipeline_finished.emit(mode, blk_ids)
        self.progress_msgbox.hide()

    def on_pipeline_finished(self):
        """Authoritative completion: All pages, OCR, inpainting, and translation verified."""
        self.progress_msgbox.hide()
        self.imgtrans_pipeline_finished.emit()

    def on_update_detect_progress(self, progress: int):
        if 'detect' in shared.pbar:
            shared.pbar['detect'].update(1)
        progress_pct = int(progress / max(1, self.imgtrans_thread.num_pages) * 100)
        self.progress_msgbox.updateDetectProgress(progress_pct)

    def on_update_ocr_progress(self, progress: int):
        if 'ocr' in shared.pbar:
            shared.pbar['ocr'].update(1)
        progress_pct = int(progress / max(1, self.imgtrans_thread.num_pages) * 100)
        self.progress_msgbox.updateOCRProgress(progress_pct)

    def on_update_translate_progress(self, progress: int):
        if 'translate' in shared.pbar:
            shared.pbar['translate'].update(1)
        progress_pct = int(progress / max(1, self.imgtrans_thread.num_pages) * 100)
        self.progress_msgbox.updateTranslateProgress(progress_pct)

    def on_update_inpaint_progress(self, progress: int):
        if 'inpaint' in shared.pbar:
            shared.pbar['inpaint'].update(1)
        progress_pct = int(progress / max(1, self.imgtrans_thread.num_pages) * 100)
        self.progress_msgbox.updateInpaintProgress(progress_pct)

    def progress(self):
        progress = {}
        num_pages = max(1, self.imgtrans_thread.num_pages)
        if cfg_module.enable_detect:
            progress['detect'] = self.imgtrans_thread.detect_counter / num_pages
        if cfg_module.enable_ocr:
            progress['ocr'] = self.imgtrans_thread.ocr_counter / num_pages
        if cfg_module.enable_inpaint:
            progress['inpaint'] = self.imgtrans_thread.inpaint_counter / num_pages
        if cfg_module.enable_translate:
            progress['translate'] = self.imgtrans_thread.translate_counter / num_pages
        return progress

    def proj_finished(self):
        if self.imgtrans_thread.detect_finished() \
            and self.imgtrans_thread.ocr_finished() \
                and self.imgtrans_thread.translate_finished() \
                    and self.imgtrans_thread.inpaint_finished():
            return True
        return False

    def finishImgtransPipeline(self):
        if self.proj_finished():
            self.progress_msgbox.hide()
            self.imgtrans_pipeline_finished.emit()
    
    def on_imgtrans_thread_stopped(self):
        """Pipeline cancelled/stopped: close progress dialog safely."""
        self.progress_msgbox.hide()

    def setTranslator(self, translator: str = None):
        if translator is None:
            translator = cfg_module.translator
        if self.translate_thread.isRunning():
            LOGGER.warning('Terminating a running translation thread.')
            self.translate_thread.terminate()
        self.translate_thread.setTranslator(translator)

    def setInpainter(self, inpainter: str = None):
        
        if self.block_set_inpainter:
            return
        
        if inpainter is None:
            inpainter =cfg_module.inpainter
        
        if self.inpaint_thread.isRunning():
            self.block_set_inpainter = True
            create_info_dialog(self.tr('Set Inpainter...'), modal=True, signal_slot_map_list=[{'signal': self.inpaint_th_finished, 'slot': 'done'}])
            self.check_inpaint_fin_timer.start(300)
            return

        self.inpaint_thread.setInpainter(inpainter)

    def setTextDetector(self, textdetector: str = None):
        if textdetector is None:
            textdetector = cfg_module.textdetector
        if self.textdetect_thread.isRunning():
            LOGGER.warning('Terminating a running text detection thread.')
            self.textdetect_thread.terminate()
        self.textdetect_thread.setTextDetector(textdetector)

    def setOCR(self, ocr: str = None):
        if ocr is None:
            ocr = cfg_module.ocr
        if self.ocr_thread.isRunning():
            LOGGER.warning('Terminating a running OCR thread.')
            self.ocr_thread.terminate()
        self.ocr_thread.setOCR(ocr)

    def on_finish_translate_page(self, page_key: str):
        self.finish_translate_page.emit(page_key)
    
    def on_finish_inpaint(self, inpaint_dict: dict):
        if self.run_canvas_inpaint:
            self.canvas_inpaint_finished.emit(inpaint_dict)
            self.run_canvas_inpaint = False

    def canvas_inpaint(self, inpaint_dict):
        self.run_canvas_inpaint = True
        self.inpaint(**inpaint_dict)
    
    def on_translatorparam_edited(self, param_key: str, param_content: dict):
        if self.translator is not None:
            self.updateModuleSetupParam(self.translator, param_key, param_content)
            cfg_module.translator_params[self.translator.name] = self.translator.params

    def on_inpainterparam_edited(self, param_key: str, param_content: dict):
        if self.inpainter is not None:
            self.updateModuleSetupParam(self.inpainter, param_key, param_content)
            cfg_module.inpainter_params[self.inpainter.name] = self.inpainter.params

    def on_textdetectorparam_edited(self, param_key: str, param_content: dict):
        if self.textdetector is not None:
            self.updateModuleSetupParam(self.textdetector, param_key, param_content)
            cfg_module.textdetector_params[self.textdetector.name] = self.textdetector.params

    def on_ocrparam_edited(self, param_key: str, param_content: dict):
        if self.ocr is not None:
            self.updateModuleSetupParam(self.ocr, param_key, param_content)
            cfg_module.ocr_params[self.ocr.name] = self.ocr.params

    def updateModuleSetupParam(self, 
                               module: Union[InpainterBase, BaseTranslator],
                               param_key: str, param_content: dict):
            
        if param_content.get('flush', False):
            param_widget: ParamComboBox = param_content['widget']
            param_widget.blockSignals(True)
            current_item = param_widget.currentText()
            param_widget.clear()
            param_widget.addItems(module.flush(param_key))
            param_widget.setCurrentText(current_item)
            param_widget.blockSignals(False)
        elif param_content.get('select_path', False):
            dialog = QFileDialog()
            f = module.params[param_key].get('path_filter', None)
            p = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
            if osp.exists(p):
                param_widget: ParamComboBox = param_content['widget']
                param_widget.setCurrentText(p)
        else:
            module.updateParam(param_key, param_content['content'])

    def handle_page_changed(self):
        if not self.imgtrans_thread.isRunning():
            if self.inpaint_thread.inpainting:
                self.run_canvas_inpaint = False
                self.inpaint_thread.terminate()

    def on_inpainter_checker_changed(self, is_checked: bool):
        cfg_module.check_need_inpaint = is_checked
        InpainterBase.check_need_inpaint = is_checked

    def on_translatebyblock_checker_changed(self, is_checked: bool):
        cfg_module.translate_by_textblock = is_checked
        BaseTranslator.translate_by_textblock = is_checked