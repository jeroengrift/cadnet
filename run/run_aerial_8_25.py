import os
import shutil
import torch
import importlib
import argparse
import numpy as np

import sys
sys.path.insert(1, '/home/griftj/phd/cadnet')

from config.default import cfg
from dataset.custom_dataset import collate_fn, CustomDatasetCad
from dataset import transforms
# from preprocess.split_dataset import SplitDataset
from preprocess.create_patches import CreatePatches
from pipeline.train import Train
from pipeline.predict import PredictRaster, PredictVector

# os.environ["CUDA_HOME"] = "/usr/local/cuda-12.6/lib64"
# os.environ["LD_LIBRARY_PATH"] = "/home/griftj/miniconda3/envs/cad/lib"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# ----------------------------------------------------------------------------------------------
# step config
parser = argparse.ArgumentParser(description='Config file')
parser.add_argument("--config", type=str, default='')
args = parser.parse_args()

config_file = args.config
cfg.merge_from_file(config_file)
cfg.freeze()  

dataset_name = cfg.DATASETS.DATA_DIR.split('/')[-1]
# Define output directory path
parts = [
    f"model-{cfg.MODEL.CLASS_NAME}",
    f"bb-{cfg.MODEL.BACKBONE}",
    # f"brk-{cfg.MODEL.USE_BRK}",
    # f"top-{cfg.MODEL.USE_TOPO}",
    # f"multi-{cfg.MODEL.USE_MULTI}",
    # "ff-False",
    # f"coa-{cfg.MODEL.USE_COA}",
    # f"ds-{dataset_name}",
    # f"s-{cfg.DATASETS.TRAIN_SAMPLE}",
    # f"sr-{cfg.DATASETS.TRAIN_SAMPLE_RATE}",
    # f"bs-{cfg.SOLVER.IMS_PER_BATCH}",
    # f"lr-{cfg.SOLVER.BASE_LR}",
    # f"buf-{cfg.DATASETS.BUFFER}",
    # f"isvis-{cfg.DATASETS.SELECT_VISIBLE}",
    # f"urb-{cfg.DATASETS.INCLUDE_URBAN}",
]

folder_name = "_".join(parts)
output_dir = os.path.join(
    cfg.OUTPUT_DIR,
    cfg.DATASETS.REFERENCE_TYPE,
    folder_name,
)

# ----------------------------------------------------------------------------------------------
# # step split dataset into train, validate and test
# if cfg.DATASETS.SPLIT:
#     split_dataset_obj = SplitDataset(cfg=cfg)
#     split_dataset_obj.split_dataset_file()

# ----------------------------------------------------------------------------------------------
# step create patches of 512x512
if cfg.DATASETS.CREATE_PATCHES:
    create_patches_obj = CreatePatches(cfg=cfg)
    if cfg.DATASETS.DOWNSAMPLE_FACTOR == 1.:
        create_patches_obj.create_patch()
    else:
        create_patches_obj.create_patch_resample()

# ----------------------------------------------------------------------------------------------
# step train the model
if cfg.MODEL.RUN_TYPE == 'train':
    dataset = 'train'
    train_dargs = dict()
    train_dargs['img_root'] = os.path.join(cfg.DATASETS.DATA_DIR, dataset, cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR)
    train_dargs['ref_root'] = os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
    train_dargs['cfg'] = cfg
    train_dargs['run_type'] = cfg.MODEL.RUN_TYPE
    train_dargs['transform'] = transforms.ToTensor()
        
    train_dataset = CustomDatasetCad(**train_dargs)
    
    norm_train_data = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                    collate_fn=collate_fn,
                                    shuffle=True,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                                    drop_last=True,
                                    worker_init_fn = np.random.seed(42))
    
    train_data = torch.utils.data.DataLoader(train_dataset,
                                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                collate_fn=collate_fn,
                                shuffle=True,
                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                drop_last=True,
                                worker_init_fn = np.random.seed(42))

    dataset = 'validate'
    val_dargs = dict()
    val_dargs['img_root'] = os.path.join(cfg.DATASETS.DATA_DIR, dataset, cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR)
    val_dargs['ref_root'] = os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
    val_dargs['cfg'] = cfg
    val_dargs['run_type'] = cfg.MODEL.RUN_TYPE
    val_dargs['transform'] = transforms.ToTensor()

    val_data = CustomDatasetCad(**val_dargs)
    val_data = torch.utils.data.DataLoader(val_data,
                                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                                    drop_last=True)
    
    if output_dir:
        if os.path.isdir(output_dir):
            raise ValueError(f"Output directory {output_dir} already exists.")
        os.makedirs(output_dir, exist_ok=True)


    model_module = importlib.import_module(cfg.MODEL.PACKAGE)
    MyModel = getattr(model_module, cfg.MODEL.CLASS_NAME)
    model = MyModel(cfg=cfg, arch=cfg.MODEL.CLASS_NAME, run_type=cfg.MODEL.RUN_TYPE)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_params}")

    if cfg.MODEL.USE_PRETRAINED:
        checkpoint = torch.load(os.path.join(output_dir, cfg.MODEL.TRAINED_MODEL))
        model.load_state_dict(checkpoint['model'])
    
    train = Train(cfg, output_dir, model, norm_train_data, train_data, val_data)
    train.train_model()

# ----------------------------------------------------------------------------------------------
# step create predictions
if cfg.MODEL.RUN_TYPE == 'predict':  
    img_root = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR)
    ref_root = os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
    
    if cfg.MODEL.TEST_OVERLAP_50:
        img_root = img_root.replace('patches_25', 'patches_25_overlap_50').replace('patches_8', 'patches_8_overlap_50')
        ref_root = os.path.join(ref_root, 'overlap_50')
        
    if cfg.MODEL.WRITE_PATCH_PREDS or cfg.MODEL.WRITE_MOSAIC_RASTER:
        weights_path = os.path.join(output_dir, cfg.MODEL.TRAINED_MODEL)

        test_dargs = dict()
        test_dargs['img_root'] = img_root
        test_dargs['ref_root'] = ref_root
        test_dargs['cfg'] = cfg
        test_dargs['run_type'] = cfg.MODEL.RUN_TYPE
        test_dargs['transform'] = transforms.ToTensor()

        test_data = CustomDatasetCad(**test_dargs)
        test_data = torch.utils.data.DataLoader(test_data,
                                        batch_size=1,
                                        collate_fn=collate_fn,
                                        shuffle=False,
                                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                                        drop_last=True)


        model_module = importlib.import_module(cfg.MODEL.PACKAGE)
        MyModel = getattr(model_module, cfg.MODEL.CLASS_NAME)
        model = MyModel(cfg=cfg, arch=cfg.MODEL.CLASS_NAME, run_type=cfg.MODEL.RUN_TYPE)
        model.eval()
        polygonize = PredictRaster(cfg, output_dir, model, test_data, weights_path)
        polygonize.write()

    if cfg.MODEL.WRITE_RASTER_STATS:
        polygonize = PredictRaster(cfg, output_dir)
        polygonize.evaluate()
        
    if cfg.MODEL.WRITE_EVALUATE_POLIS:
        polygonize = PredictVector(cfg, output_dir)
        polygonize.write_evaluate()
