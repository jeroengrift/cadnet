# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import importlib
import importlib.util
import sys
import os


def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module

# def create_dargs(cfg, dataset='train'):
#     return dict(img_root=os.path.join(cfg.DATASETS.DATA_DIR, dataset, cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR),
#                 ref_root=os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
#                 )