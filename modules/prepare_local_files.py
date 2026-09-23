from typing import Union, List
import os.path as osp
import os

from . import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
from .base import BaseModule, LOGGER
import utils.shared as shared
from utils.download_util import download_and_check_files


def download_and_check_module_files(module_class_list: List[BaseModule] = None):
    if module_class_list is None:
        module_class_list = []
        for registered in [INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS]:
            for module_key in registered.module_dict.keys():
                module_class_list.append(registered.get(module_key))

    for module_class in module_class_list:
        if module_class.download_file_on_load or module_class.download_file_list is None:
            continue
        for download_kwargs in module_class.download_file_list:
            all_successful = download_and_check_files(**download_kwargs)
            if all_successful:
                continue
            LOGGER.error(f'Please save these files manually to sepcified path and restart the application, otherwise {module_class} will be unavailable.')

def prepare_pkuseg():
    # pkuseg is only for Chinese text segmentation and is disabled for Japanese->Vietnamese pipeline.
    return


def prepare_local_files_forall():
    # If running with --frozen or offline, skip slow network scans for all modules
    if shared.args and getattr(shared.args, 'frozen', False):
        return

    # download files required by detect, ocr, inpaint and translators
    download_and_check_module_files()

    prepare_pkuseg()

    if shared.CACHE_UPDATED:
        shared.dump_cache()


