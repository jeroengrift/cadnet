import os
import time
import logging
import datetime
import torch
import time

from matplotlib import pyplot as plt
from config.default import cfg
from utils.comm import to_single_device
from utils.earlystopper import EarlyStopper
from utils.logger import setup_logger
from utils.metric_logger import MetricLogger
from utils.checkpoint import DetectronCheckpointer
from utils.seed import set_random_seed
from models.solver import make_lr_scheduler, make_optimizer

torch.multiprocessing.set_sharing_strategy('file_system')


class Train():
    def __init__(self, cfg, output_dir, model, norm_train_data, train_data, val_data):
        self.cfg = cfg
        self.model = model
        self.norm_train_data = norm_train_data
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir 
        
    def train_model(self):
        logger = setup_logger('training', self.output_dir, out_file='train.log')
        logger.info("Loaded configuration file")
        output_config_path = os.path.join(self.output_dir, 'config.yml')
        logger.info("Saving config into: {}".format(output_config_path))

        with open(output_config_path, 'w') as f:
            f.write(cfg.dump())

        set_random_seed(2, True)

        logger = logging.getLogger("training")
        device = cfg.MODEL.DEVICE
        self.model.to(device)

        optimizer = make_optimizer(cfg, self.model)
        scheduler = make_lr_scheduler(cfg, optimizer)
        early_stopper = EarlyStopper(cfg)

        arguments = {"epoch": 0}
        max_epoch = cfg.SOLVER.MAX_EPOCH
        arguments["max_epoch"] = max_epoch

        checkpointer = DetectronCheckpointer(cfg,
                                            self.model,
                                            optimizer,
                                            save_dir=self.output_dir,
                                            save_to_disk=True,
                                            logger=logger)
        
        start_training_time = time.time()
        start_epoch = arguments['epoch']
        epoch_size = len(self.train_data)
        global_iteration = epoch_size * start_epoch

        plot_val_losses = []
        plot_train_losses = []

        def create_global_loss_dict():
            return {
            'seg_edge_loss_topo_512' : [],
            'seg_edge_loss_topo_s512' :  [],
            'seg_edge_loss_topo_s256' :  [],
            'seg_edge_loss_topo_s128' :  [],
            'seg_edge_loss_topo_s64' :  [],
            'seg_edge_loss_topo_s32' :  [],
            
            'seg_edge_coa_loss_topo_512' :  [],
            'seg_edge_coa_loss_topo_s512' :  [],
            'seg_edge_coa_loss_topo_s256' :  [],
            'seg_edge_coa_loss_topo_s128' :  [],
            'seg_edge_coa_loss_topo_s64' :  [],
            'seg_edge_coa_loss_topo_s32' :  [],
            
            'seg_edge_loss_brk_512' : [],
            'seg_edge_loss_brk_s512' :  [],
            'seg_edge_loss_brk_s256' :  [],
            'seg_edge_loss_brk_s128' :  [],
            'seg_edge_loss_brk_s64' :  [],
            'seg_edge_loss_brk_s32' :  [],
            
            'seg_edge_coa_loss_brk_512' :  [],
            'seg_edge_coa_loss_brk_s512' :  [],
            'seg_edge_coa_loss_brk_s256' :  [],
            'seg_edge_coa_loss_brk_s128' :  [],
            'seg_edge_coa_loss_brk_s64' :  [],
            'seg_edge_coa_loss_brk_s32' :  [],
            
            'seg_edge_loss_brk_512_vis' :  [],
            'seg_edge_loss_brk_s512_vis' :  [],
            'seg_edge_loss_brk_s256_vis' :  [],
            'seg_edge_loss_brk_s128_vis' :  [],
            'seg_edge_loss_brk_s64_vis' :  [],
            'seg_edge_loss_brk_s32_vis' :  [],
            
            'seg_edge_coa_loss_brk_512_vis' :  [],
            'seg_edge_coa_loss_brk_s512_vis' :  [],
            'seg_edge_coa_loss_brk_s256_vis' :  [],
            'seg_edge_coa_loss_brk_s128_vis' :  [],
            'seg_edge_coa_loss_brk_s64_vis' :  [],
            'seg_edge_coa_loss_brk_s32_vis' :  [],
            
            'seg_edge_loss_brk_512_inv' :  [],
            'seg_edge_loss_brk_s512_inv' :  [],
            'seg_edge_loss_brk_s256_inv' :  [],
            'seg_edge_loss_brk_s128_inv' :  [],
            'seg_edge_loss_brk_s64_inv' :  [],
            'seg_edge_loss_brk_s32_inv' :  [],
            
            'seg_edge_coa_loss_brk_512_inv' :  [],
            'seg_edge_coa_loss_brk_s512_inv' :  [],
            'seg_edge_coa_loss_brk_s256_inv' :  [],
            'seg_edge_coa_loss_brk_s128_inv' :  [],
            'seg_edge_coa_loss_brk_s64_inv' :  [],
            'seg_edge_coa_loss_brk_s32_inv' :  [],
        }
        
        train_extra_losses = create_global_loss_dict()
        val_extra_losses = create_global_loss_dict()

        train_loss_dict_norm = None
        
        loss_weights = {
            'seg_edge_loss_topo_512' : cfg.WEIGHTS.SEG_LOSS_512_TOPO,
            'seg_edge_loss_topo_s512' : cfg.WEIGHTS.SEG_LOSS_S512_TOPO,
            'seg_edge_loss_topo_s256' : cfg.WEIGHTS.SEG_LOSS_S256_TOPO,
            'seg_edge_loss_topo_s128' : cfg.WEIGHTS.SEG_LOSS_S128_TOPO,
            'seg_edge_loss_topo_s64' : cfg.WEIGHTS.SEG_LOSS_S64_TOPO,
            'seg_edge_loss_topo_s32' : cfg.WEIGHTS.SEG_LOSS_S32_TOPO,
            
            'seg_edge_coa_loss_topo_512' : cfg.WEIGHTS.SEG_COA_LOSS_512_TOPO,
            'seg_edge_coa_loss_topo_s512' : cfg.WEIGHTS.SEG_COA_LOSS_S512_TOPO,
            'seg_edge_coa_loss_topo_s256' : cfg.WEIGHTS.SEG_COA_LOSS_S256_TOPO,
            'seg_edge_coa_loss_topo_s128' : cfg.WEIGHTS.SEG_COA_LOSS_S128_TOPO,
            'seg_edge_coa_loss_topo_s64' : cfg.WEIGHTS.SEG_COA_LOSS_S64_TOPO,
            'seg_edge_coa_loss_topo_s32' : cfg.WEIGHTS.SEG_COA_LOSS_S32_TOPO,
            
            'seg_edge_loss_brk_512' : cfg.WEIGHTS.SEG_LOSS_512_CAD,
            'seg_edge_loss_brk_s512' : cfg.WEIGHTS.SEG_LOSS_S512_CAD,
            'seg_edge_loss_brk_s256' : cfg.WEIGHTS.SEG_LOSS_S256_CAD,
            'seg_edge_loss_brk_s128' : cfg.WEIGHTS.SEG_LOSS_S128_CAD,
            'seg_edge_loss_brk_s64' : cfg.WEIGHTS.SEG_LOSS_S64_CAD,
            'seg_edge_loss_brk_s32' : cfg.WEIGHTS.SEG_LOSS_S32_CAD,
            
            'seg_edge_coa_loss_brk_512' : cfg.WEIGHTS.SEG_COA_LOSS_512_CAD,
            'seg_edge_coa_loss_brk_s512' : cfg.WEIGHTS.SEG_COA_LOSS_S512_CAD,
            'seg_edge_coa_loss_brk_s256' : cfg.WEIGHTS.SEG_COA_LOSS_S256_CAD,
            'seg_edge_coa_loss_brk_s128' : cfg.WEIGHTS.SEG_COA_LOSS_S128_CAD,
            'seg_edge_coa_loss_brk_s64' : cfg.WEIGHTS.SEG_COA_LOSS_S64_CAD,
            'seg_edge_coa_loss_brk_s32' : cfg.WEIGHTS.SEG_COA_LOSS_S32_CAD,
            
            'seg_edge_loss_brk_512_vis' : cfg.WEIGHTS.SEG_LOSS_512_CAD_VIS,
            'seg_edge_loss_brk_s512_vis' : cfg.WEIGHTS.SEG_LOSS_S512_CAD_VIS,
            'seg_edge_loss_brk_s256_vis' : cfg.WEIGHTS.SEG_LOSS_S256_CAD_VIS,
            'seg_edge_loss_brk_s128_vis' : cfg.WEIGHTS.SEG_LOSS_S128_CAD_VIS,
            'seg_edge_loss_brk_s64_vis' : cfg.WEIGHTS.SEG_LOSS_S64_CAD_VIS,
            'seg_edge_loss_brk_s32_vis' : cfg.WEIGHTS.SEG_LOSS_S32_CAD_VIS,
            
            'seg_edge_coa_loss_brk_512_vis' : cfg.WEIGHTS.SEG_COA_LOSS_512_CAD_VIS,
            'seg_edge_coa_loss_brk_s512_vis' : cfg.WEIGHTS.SEG_COA_LOSS_S512_CAD_VIS,
            'seg_edge_coa_loss_brk_s256_vis' : cfg.WEIGHTS.SEG_COA_LOSS_S256_CAD_VIS,
            'seg_edge_coa_loss_brk_s128_vis' : cfg.WEIGHTS.SEG_COA_LOSS_S128_CAD_VIS,
            'seg_edge_coa_loss_brk_s64_vis' : cfg.WEIGHTS.SEG_COA_LOSS_S64_CAD_VIS,
            'seg_edge_coa_loss_brk_s32_vis' : cfg.WEIGHTS.SEG_COA_LOSS_S32_CAD_VIS,
            
            'seg_edge_loss_brk_512_inv' : cfg.WEIGHTS.SEG_LOSS_512_CAD_INV,
            'seg_edge_loss_brk_s512_inv' : cfg.WEIGHTS.SEG_LOSS_S512_CAD_INV,
            'seg_edge_loss_brk_s256_inv' : cfg.WEIGHTS.SEG_LOSS_S256_CAD_INV,
            'seg_edge_loss_brk_s128_inv' : cfg.WEIGHTS.SEG_LOSS_S128_CAD_INV,
            'seg_edge_loss_brk_s64_inv' : cfg.WEIGHTS.SEG_LOSS_S64_CAD_INV,
            'seg_edge_loss_brk_s32_inv' : cfg.WEIGHTS.SEG_LOSS_S32_CAD_INV,
            
            'seg_edge_coa_loss_brk_512_inv' : cfg.WEIGHTS.SEG_COA_LOSS_512_CAD_INV,
            'seg_edge_coa_loss_brk_s512_inv' : cfg.WEIGHTS.SEG_COA_LOSS_S512_CAD_INV,
            'seg_edge_coa_loss_brk_s256_inv' : cfg.WEIGHTS.SEG_COA_LOSS_S256_CAD_INV,
            'seg_edge_coa_loss_brk_s128_inv' : cfg.WEIGHTS.SEG_COA_LOSS_S128_CAD_INV,
            'seg_edge_coa_loss_brk_s64_inv' : cfg.WEIGHTS.SEG_COA_LOSS_S64_CAD_INV,
            'seg_edge_coa_loss_brk_s32_inv' : cfg.WEIGHTS.SEG_COA_LOSS_S32_CAD_INV,
        }
                
        logger.info("calculate normalization weights")
        norm_iterations = 0
        max_iterations =  800 // self.cfg.SOLVER.IMS_PER_BATCH
        
        for epoch in range(1, 100):
            if norm_iterations > max_iterations:
                break
            norm_meters = MetricLogger(" ")
            self.model.train()
            for it, (norm_images, norm_annotations, norm_ann_ids, norm_filenames) in enumerate(self.norm_train_data):
                with torch.autocast(device_type=device, dtype=torch.float16):
                    norm_images = norm_images.to(device)
                    norm_annotations = to_single_device(norm_annotations, device)
                    norm_loss_dict, norm_recall, norm_precision, norm_f1 = self.model(norm_images, norm_annotations)  
                    
                    norm_loss_dict = {k: v for k, v in norm_loss_dict.items() if v is not None}  
                    
                    norm_loss = None 
                    for key, value in norm_loss_dict.items():
                        if norm_loss is None:
                            norm_loss = value
                        else:
                            norm_loss += value
                                            
                    with torch.no_grad():
                        norm_meters.update(loss=norm_loss)
                        
                        for key, value in norm_loss_dict.items():
                            norm_meters.update(**{key: value})

                optimizer.zero_grad()
                norm_loss.backward()
                optimizer.step()
                norm_iterations += 1 
                
                if norm_iterations > max_iterations:
                    break
        
        loss_norms = {}
        for key, value in norm_meters.meters.items():
            if key not in ['loss', 'recall', 'precision', 'f1']:
                loss_norms[key] = value.global_avg
        
        logger.info('type: norm ' + norm_meters.delimiter.join([f'{key}: {value}' for key, value in loss_norms.items()]))
        
        for epoch in range(start_epoch + 1, arguments['max_epoch'] + 1):
            train_meters = MetricLogger(" ")
            val_meters = MetricLogger(" ")
            self.model.train()
            arguments['epoch'] = epoch

            for it, (train_images, train_annotations, train_ann_ids, train_filenames) in enumerate(self.train_data):
                with torch.autocast(device_type=device, dtype=torch.float16):
                    train_images = train_images.to(device)
                    train_annotations = to_single_device(train_annotations, device)
                    
                    train_loss_dict, train_recall, train_precision, train_f1 = self.model(train_images, train_annotations)    
                    train_loss_dict = {k: v for k, v in train_loss_dict.items() if v is not None}  

                    train_loss_dict_norm = {k: ((v.item() / loss_norms[k]))  for k, v in train_loss_dict.items()}
                                                    
                    train_loss = None 
                    for key, value in train_loss_dict.items():
                        if train_loss is None:
                            train_loss = ((value / loss_norms[key])  * loss_weights[key])
                        else:
                            train_loss += ((value / loss_norms[key]) * loss_weights[key])
                                                                                                                        
                with torch.no_grad():
                    train_meters.update(loss=train_loss,
                                        recall=train_recall, 
                                        precision=train_precision, 
                                        f1=train_f1)
                    
                    for key, value in train_loss_dict_norm.items():
                        train_meters.update(**{key: value})
                
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

                global_iteration += 1
                
                if it % 20 == 0 or it + 1 == len(self.train_data):
                    logger.info(
                        train_meters.delimiter.join(
                            [
                                "type: train",
                                "epoch: {epoch}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            epoch=epoch,
                            iter=it,
                            meters=str(train_meters),
                            lr=optimizer.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
            
            self.model.eval()

            for it, (val_images, val_annotations, val_ann_ids, val_filenames) in enumerate(self.val_data):
                with torch.no_grad():
                    val_images = val_images.to(device)
                    val_annotations = to_single_device(val_annotations, device)   
                    val_loss_dict, val_recall, val_precision, val_f1 = self.model(val_images, val_annotations)    

                    val_loss_dict = {k: v for k, v in val_loss_dict.items() if v is not None}
                    val_loss_dict_norm = {k: ((v.item() / loss_norms[k]))  for k, v in val_loss_dict.items()}

                    val_loss = None 
                    for key, value in val_loss_dict.items():
                        if val_loss is None:
                            val_loss = ((value / loss_norms[key]) * loss_weights[key])
                        else:
                            val_loss += ((value / loss_norms[key]) * loss_weights[key])
                        
                    val_meters.update(loss=val_loss,
                                    recall=val_recall, 
                                    precision=val_precision, 
                                    f1=val_f1)
                                        
                    for key, value in val_loss_dict_norm.items():
                        val_meters.update(**{key: value})
                
                    if it + 1 == len(self.val_data):
                        logger.info('type: train ' +
                                    f'epoch: {epoch} ' +
                                    train_meters.delimiter.join([f'{key}: {train_meters.meters[key].global_avg}' for key in train_meters.meters.keys()])
                                    )
                        logger.info('type: val ' +
                                    f'epoch: {epoch} ' +
                                    val_meters.delimiter.join([f'{key}: {val_meters.meters[key].global_avg}' for key in val_meters.meters.keys()])
                                    )

            # TODO: Not so nice, make it cleaner
            train_loss_epoch = train_meters.meters['loss'].global_avg
            val_loss_epoch = val_meters.meters['loss'].global_avg
            plot_train_losses.append(train_loss_epoch)
            plot_val_losses.append(val_loss_epoch)
        
            for key, value in train_meters.meters.items():
                if key not in ['loss', 'recall', 'precision', 'f1']:
                    train_extra_losses[key].append(value.global_avg)

            for key, value in val_meters.meters.items():
                if key not in ['loss', 'recall', 'precision', 'f1']:
                    val_extra_losses[key].append(value.global_avg)

            checkpointer.save('model_{:05d}'.format(epoch))

            if early_stopper.early_stop(val_loss_epoch):             
                break
            
            scheduler.step()
            
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation Loss")
        plt.plot(plot_val_losses,label="val")
        plt.plot(plot_train_losses,label="train")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "total_loss_plot.jpg"))

        for key, value in train_extra_losses.items():
            if value:
                plt.figure(figsize=(10,5))
                plt.title(f"Training and Validation Loss {key}")
                plt.plot(value,label='train')
                plt.plot(val_extra_losses[key],label='val')
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f"individual_loss_plot_{key}.jpg"))
            
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / epoch)".format(
                total_time_str, total_training_time / max_epoch
            )
        )
        