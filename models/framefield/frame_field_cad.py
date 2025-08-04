import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.nn.modules.loss import _Loss
from . import frame_field_utils
from models.framefield.torch_lydorn import kornia
from models.basemodel import BaseModel, FocalLoss
from models.framefield.unet import UNetBackbone
from models.basemodel import BaseModel

loss_params = {
"epoch_thresholds": [0, 5, 10],
"bce_coef": 1.0,
"dice_coef": 0.2,

"weight_seg": 10.0,
"weight_crossfield_align": 1.0,
"weight_crossfield_align90": 0.2,
"weight_crossfield_smooth": 0.005,
"weight_seg_edge_crossfield": 0.2,
"weight_seg_int_crossfield": 0.2,
"weight_seg_edge_interior": 0.2,

"norm_seg_edge": np.mean([1.0462126731872559, 1.076125979423523, 1.144355297088623, 1.1596720218658447, 1.0561773777008057]),
"norm_seg_int": np.mean([0.3998872935771942, 0.382240355014801, 0.360551118850708, 0.3391876816749573, 0.3352455794811249]),
"norm_crossfield_align": np.mean([0.45815151929855347, 0.29313939809799194, 0.19811594486236572, 0.12370175123214722, 0.11108013987541199]),
"norm_crossfield_align90": np.mean([0.7468398213386536, 0.5968919992446899, 0.4883197247982025, 0.32593637704849243, 0.27209287881851196]),
"norm_crossfield_smooth": np.mean([0.07896240055561066, 0.06346185505390167, 0.05538690835237503, 0.03921051323413849, 0.036167874932289124]),
"norm_seg_edge_crossfield": np.mean([0.30270183086395264, 0.14040610194206238, 0.11770360171794891, 0.08619124442338943, 0.07911717146635056]),
"norm_seg_int_crossfield": np.mean([0.3441044092178345, 0.15385043621063232, 0.12035398930311203, 0.09804925322532654, 0.08839933574199677]),
"norm_seg_edge_interior": np.mean([0.20109546184539795, 0.21266773343086243, 0.22060319781303406, 0.22036892175674438, 0.22200529277324677])
}


class SegLoss(_Loss):
    def __init__(self, name, bce_coef=0.5, dice_coef=0.5):
        super(SegLoss, self).__init__(name)
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def compute(self, pred_mask, gt_mask):
        
        dice = smp.losses.DiceLoss(mode='binary', from_logits=False)
        dice_loss = dice(pred_mask, gt_mask)
        
        focal = FocalLoss(mode='binary')
        focal_loss = focal(pred_mask, gt_mask)
        
        return dice_loss + focal_loss
        
class CrossfieldAlignLoss(_Loss):
    def __init__(self, name):
        super(CrossfieldAlignLoss, self).__init__(name)

    def compute(self, pred_ff, gt_ff, gt_mask):
        c0 = pred_ff[:, :2]
        c2 = pred_ff[:, 2:]
        z = gt_ff       
        align_loss = frame_field_utils.framefield_align_error(c0, c2, z, complex_dim=1)
        avg_align_loss = torch.mean(align_loss * gt_mask)

        return avg_align_loss

class CrossfieldAlign90Loss(_Loss):
    def __init__(self, name):
        super(CrossfieldAlign90Loss, self).__init__(name)

    def compute(self, pred_ff, gt_ff, gt_mask):
        c0 = pred_ff[:, :2]
        c2 = pred_ff[:, 2:]
        z = gt_ff   
        # gt_edges_minus_vertices = gt_mask - gt_vertices
        # gt_edges_minus_vertices = gt_edges_minus_vertices.clamp(0, 1)
        z_90deg = torch.cat((- z[:, 1:2, ...], z[:, 0:1, ...]), dim=1)    
        align90_loss = frame_field_utils.framefield_align_error(c0, c2, z_90deg, complex_dim=1)
        # avg_align90_loss = torch.mean(align90_loss * gt_edges_minus_vertices)
        avg_align90_loss = torch.mean(align90_loss * gt_mask)
        
        return avg_align90_loss
        
class CrossfieldSmoothLoss(_Loss):
    def __init__(self, name):
        super(CrossfieldSmoothLoss, self).__init__(name)

    def compute(self, pred_ff, gt_mask):
        gt_edges_inv = 1 - gt_mask
        penalty = frame_field_utils.LaplacianPenalty(channels=4).laplacian_filter(pred_ff)
        avg_penalty = torch.mean(penalty * gt_edges_inv[:, None, ...])
        
        return avg_penalty

class SegCrossfieldLoss(_Loss):
    def __init__(self, name):
        super(SegCrossfieldLoss, self).__init__(name)

    def compute(self, pred_ff, pred_mask, device="cuda:0"):
        c0 = pred_ff[:, :2]
        c2 = pred_ff[:, 2:]
        spatial_gradient = kornia.filters.SpatialGradient(mode="scharr", coord="ij", normalized=True, device=device)

        seg_grads = 2 * spatial_gradient(pred_mask)  # (b, c, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
        seg_grad_norm = seg_grads.norm(dim=2)  # (b, c, h, w)
        seg_grads_normed = seg_grads / (seg_grad_norm[:, :, None, ...] + 1e-6)  # (b, c, 2, h, w)
        
        seg_slice_grads_normed = seg_grads_normed[:, 0, ...]
        seg_slice_grad_norm = seg_grad_norm[:, 0, ...]
        align_loss = frame_field_utils.framefield_align_error(c0, c2, seg_slice_grads_normed, complex_dim=1)
        avg_align_loss = torch.mean(align_loss * seg_slice_grad_norm.detach())

        return avg_align_loss
    
class SegEdgeInteriorLoss(_Loss):
    def __init__(self, name):
        super(SegEdgeInteriorLoss, self).__init__(name)

    def compute(self, seg_edge, seg_interior, device="cuda:0"):
        spatial_gradient = kornia.filters.SpatialGradient(mode="scharr", coord="ij", normalized=True, device=device)
        seg_grads = 2 * spatial_gradient(seg_interior)  # (b, c, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
        seg_grad_norm = seg_grads.norm(dim=2)  # (b, c, h, w)
        seg_interior_grad_norm = seg_grad_norm[:, 0, ...]
        
        raw_loss = torch.abs(seg_edge - seg_interior_grad_norm)
        # Apply the loss only on interior boundaries and outside of objects
        outside_mask = (torch.cos(np.pi * seg_interior) + 1) / 2
        boundary_mask = (1 - torch.cos(np.pi * seg_interior_grad_norm)) / 2
        mask = torch.max(outside_mask, boundary_mask).float()
        avg_loss = torch.mean(raw_loss * mask)
        return avg_loss
    
    
def get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        print(out_channels)
        i += 1
    return out_channels

def seg_module(backbone_features, seg_channels=2):
    seg_module = torch.nn.Sequential(
        torch.nn.Conv2d(backbone_features, backbone_features, 3, padding=1),
        torch.nn.BatchNorm2d(backbone_features),
        torch.nn.ELU(),
        torch.nn.Conv2d(backbone_features, seg_channels, 1),
        torch.nn.Sigmoid(),)
    return seg_module

def crossfield_module(backbone_features, seg_channels=2, crossfield_channels=4):
    crossfield_module = torch.nn.Sequential(
        torch.nn.Conv2d(backbone_features + seg_channels, backbone_features, 3, padding=1),
        torch.nn.BatchNorm2d(backbone_features),
        torch.nn.ELU(),
        torch.nn.Conv2d(backbone_features, crossfield_channels, 1),
        torch.nn.Tanh(),)
    return crossfield_module


class FrameFieldModel(BaseModel):
    def __init__(self, cfg, arch, encoder_name, run_type="train", **kwargs):
        super().__init__(cfg, run_type)      
        self.cfg = cfg
        self.backbone = UNetBackbone(3, 16)
        
        # self.backbone = _SimpleSegmentationModel(UNetBackbone(3, 16), classifier=torch.nn.Identity())
        # self.backbone = _SimpleSegmentationModel(UNetResNetBackbone(101, pretrained=True), classifier=torch.nn.Identity())
        # backbone_out_features = get_out_channels(self.backbone)
             
    def build_model(self, image, ff= True, multi=True):
        outputs = {}
        backbone = self.backbone(image)
        
        def build_resolution_ff(resolution, seg_module, crossfield_module):  
            backbone_features = backbone[f"out_{resolution}"]
            seg = seg_module(backbone_features)
            seg_to_cat = seg.clone().detach()
            backbone_features = torch.cat([backbone_features, seg_to_cat], dim=1) # Add seg to image features
            crossfield = 2 * crossfield_module(backbone_features) # Outputs c_0, c_2 values in [-2, 2]  
            
            return seg, crossfield
        
        def build_resolution(resolution, seg_module):  
            backbone_features = backbone[f"out_{resolution}"]
            seg = seg_module(backbone_features)

            return seg
        
        if not multi and ff:
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[0]}"], outputs[f"crossfield_{self.cfg.DATASETS.RESOLUTIONS[0]}"] = build_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[0], self.seg_module, self.crossfield_module) 
        
        elif multi and ff:
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[0]}"], outputs[f"crossfield_{self.cfg.DATASETS.RESOLUTIONS[0]}"] = build_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[0], self.seg_module_1, self.crossfield_module_1) 
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[1]}"], outputs[f"crossfield_{self.cfg.DATASETS.RESOLUTIONS[1]}"] = build_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[1], self.seg_module_2, self.crossfield_module_2)
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[2]}"], outputs[f"crossfield_{self.cfg.DATASETS.RESOLUTIONS[2]}"] = build_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[2], self.seg_module_3, self.crossfield_module_3)
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[3]}"], outputs[f"crossfield_{self.cfg.DATASETS.RESOLUTIONS[3]}"] = build_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[3], self.seg_module_4, self.crossfield_module_4)
        
        elif not multi and not ff:
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[0]}"] = build_resolution(self.cfg.DATASETS.RESOLUTIONS[0], self.seg_module) 

        elif multi and not ff:
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[0]}"] = build_resolution(self.cfg.DATASETS.RESOLUTIONS[0], self.seg_module_1) 
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[1]}"] = build_resolution(self.cfg.DATASETS.RESOLUTIONS[1], self.seg_module_2)
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[2]}"] = build_resolution(self.cfg.DATASETS.RESOLUTIONS[2], self.seg_module_3)
            outputs[f"seg_{self.cfg.DATASETS.RESOLUTIONS[3]}"] = build_resolution(self.cfg.DATASETS.RESOLUTIONS[3], self.seg_module_4)

        return outputs

    def forward(self, images, annotations):
        if self.run_type == 'train':
            return self.forward_train(images, annotations)
        elif self.run_type == 'predict':
            return self.forward_predict(images, annotations)

    def forward_train(self, images, annotations):
        pass
        
    def forward_predict(self, images, annotations):
        pass
    
    def loss_resolution_ff(self, resolution, pred, annotations):
        gt_edge = annotations[f"mask_{resolution}"][:,0,:,:].unsqueeze(1)
        gt_ff = annotations[f"frame_field_{resolution}"].unsqueeze(1) 
        gt_ff = torch.cat([torch.cos(gt_ff), torch.sin(gt_ff)], dim=1)
        pred_edge = pred[f"seg_{resolution}"][:,0,:,:].unsqueeze(1)
        pred_ff = pred[f"crossfield_{resolution}"]
        
        # segmentation loss = dice + focal
        ledge = SegLoss("seg", loss_params['bce_coef'], loss_params['dice_coef']).compute(pred_edge, gt_edge)
        # compares the predicted frame field to the ground truth frame field
        lalign = CrossfieldAlignLoss("crossfield_align").compute(pred_ff, gt_ff, gt_edge)        
        # compares the predicted frame field to the ground truth frame field rotated by 90 degrees
        lalign90 = CrossfieldAlign90Loss("crossfield_align90").compute(pred_ff, gt_ff, gt_edge)               
        # compares the smoothened predicted frame field to the inverse ground truth frame field
        lsmooth = CrossfieldSmoothLoss("crossfield_smooth").compute(pred_ff, gt_edge) 
        # aligns the spatial gradient of the predicted edge map with the frame field c0c2
        ledge_align = SegCrossfieldLoss("seg_edge_crossfield").compute(pred_ff, pred_edge)
        
        ledge = loss_params['weight_seg'] * (ledge/loss_params['norm_seg_edge'])
        lalign = loss_params['weight_crossfield_align'] * (lalign/loss_params['norm_crossfield_align'])
        lalign90 = loss_params['weight_crossfield_align90'] * (lalign90/loss_params['norm_crossfield_align90'])
        lsmooth = loss_params['weight_crossfield_smooth'] * (lsmooth/loss_params['norm_crossfield_smooth'])
        ledge_align = loss_params['weight_seg_edge_crossfield'] * (ledge_align/loss_params['norm_seg_edge_crossfield'])
                
        total_loss = ledge + lalign + lalign90 + lsmooth + ledge_align
        recall, precision, f1 = self.calculate_metrics(gt_edge, pred_edge)
        
        return total_loss, recall, precision, f1

    def loss_resolution(self, resolution, pred, annotations):
        gt_edge = annotations[f"mask_{resolution}"][:,0,:,:].unsqueeze(1)
        pred_edge = pred[f"seg_{resolution}"][:,0,:,:].unsqueeze(1)
        
        # segmentation loss = dice + focal
        total_loss = SegLoss("seg", loss_params['bce_coef'], loss_params['dice_coef']).compute(pred_edge, gt_edge)
        total_loss = loss_params['weight_seg'] * (total_loss/loss_params['norm_seg_edge'])
        recall, precision, f1 = self.calculate_metrics(gt_edge, pred_edge)
        
        return total_loss, recall, precision, f1

    
class FrameFieldModelMulti(FrameFieldModel):
    def __init__(self, cfg, arch, encoder_name, run_type="train", **kwargs):
            super().__init__(cfg, arch, encoder_name, run_type, **kwargs)     

            # TODO create function to get out_channels
            self.seg_module_1 = seg_module(int(cfg.DATASETS.RESOLUTIONS[0] / 32))
            self.seg_module_2 = seg_module(int(cfg.DATASETS.RESOLUTIONS[1] / 8))
            self.seg_module_3 = seg_module(int(cfg.DATASETS.RESOLUTIONS[2] / 2))
            self.seg_module_4 = seg_module(int(cfg.DATASETS.RESOLUTIONS[3] / 0.5))
            
            self.crossfield_module_1 = crossfield_module(int(cfg.DATASETS.RESOLUTIONS[0] / 32))
            self.crossfield_module_2 = crossfield_module(int(cfg.DATASETS.RESOLUTIONS[1] / 8))
            self.crossfield_module_3 = crossfield_module(int(cfg.DATASETS.RESOLUTIONS[2] / 2))
            self.crossfield_module_4 = crossfield_module(int(cfg.DATASETS.RESOLUTIONS[3] / 0.5)) 
            
    def forward_train(self, images, annotations):
        pred = self.build_model(images, ff=True, multi=True)
            
        loss_1, recall_1, precision_1, f1_1 = self.loss_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[0], pred, annotations)
        loss_2, recall_2, precision_2, f1_2 = self.loss_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[1], pred, annotations)
        loss_3, recall_3, precision_3, f1_3 = self.loss_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[2], pred, annotations)
        loss_4, recall_4, precision_4, f1_4 = self.loss_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[3], pred, annotations)
        
        overall_loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4
        overall_recall = (recall_1 + recall_2 + recall_3 + recall_4) / 4
        overall_precision = (precision_1 + precision_2 + precision_3 + precision_4) / 4
        overall_f1 = (f1_1 + f1_2 + f1_3 + f1_4) / 4
        
        return overall_loss, overall_recall, overall_precision, overall_f1 
    
    def forward_predict(self, images, annotations):
        preds = self.build_model(images, ff=True, multi=True)
        return annotations, images, preds
    
    
class FrameFieldModelSingle(FrameFieldModel):
    def __init__(self, cfg, arch, encoder_name, run_type="train", **kwargs):
            super().__init__(cfg, arch, encoder_name, run_type, **kwargs)
            
            self.seg_module = seg_module(int(cfg.DATASETS.RESOLUTIONS[0] / 32))
            self.crossfield_module = crossfield_module(int(cfg.DATASETS.RESOLUTIONS[0] / 32))      

    def forward_train(self, images, annotations):
        pred = self.build_model(images, ff=True, multi=False)        
        overall_loss, overall_recall, overall_precision, overall_f1 = self.loss_resolution_ff(self.cfg.DATASETS.RESOLUTIONS[0], pred, annotations)

        return overall_loss, overall_recall, overall_precision, overall_f1 

    def forward_predict(self, images, annotations):
        preds = self.build_model(images, ff=True, multi=False)
        return annotations, images, preds


class SegmentationSingle(FrameFieldModel):
    def __init__(self, cfg, arch, encoder_name, run_type="train", **kwargs):
            super().__init__(cfg, arch, encoder_name, run_type, **kwargs)    
            self.seg_module = seg_module(int(cfg.DATASETS.RESOLUTIONS[0] / 32))  


    def forward_train(self, images, annotations):
        pred = self.build_model(images, ff=False, multi=False)
        overall_loss, overall_recall, overall_precision, overall_f1 = self.loss_resolution(self.cfg.DATASETS.RESOLUTIONS[0], pred, annotations)

        return overall_loss, overall_recall, overall_precision, overall_f1 


    def forward_predict(self, images, annotations):
        preds = self.build_model(images, ff=False, multi=False)
        return annotations, images, preds


class SegmentationMulti(FrameFieldModel):
    def __init__(self, cfg, arch, encoder_name, run_type="train", **kwargs):
            super().__init__(cfg, arch, encoder_name, run_type, **kwargs)   

            self.seg_module_1 = seg_module(int(cfg.DATASETS.RESOLUTIONS[0] / 32))
            self.seg_module_2 = seg_module(int(cfg.DATASETS.RESOLUTIONS[1] / 8))
            self.seg_module_3 = seg_module(int(cfg.DATASETS.RESOLUTIONS[2] / 2))
            self.seg_module_4 = seg_module(int(cfg.DATASETS.RESOLUTIONS[3] / 0.5))   

    def forward_train(self, images, annotations):
        pred = self.build_model(images, ff=False, multi=True)
            
        loss_1, recall_1, precision_1, f1_1 = self.loss_resolution(self.cfg.DATASETS.RESOLUTIONS[0], pred, annotations)
        loss_2, recall_2, precision_2, f1_2 = self.loss_resolution(self.cfg.DATASETS.RESOLUTIONS[1], pred, annotations)
        loss_3, recall_3, precision_3, f1_3 = self.loss_resolution(self.cfg.DATASETS.RESOLUTIONS[2], pred, annotations)
        loss_4, recall_4, precision_4, f1_4 = self.loss_resolution(self.cfg.DATASETS.RESOLUTIONS[3], pred, annotations)
        
        overall_loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4
        overall_recall = (recall_1 + recall_2 + recall_3 + recall_4) / 4
        overall_precision = (precision_1 + precision_2 + precision_3 + precision_4) / 4
        overall_f1 = (f1_1 + f1_2 + f1_3 + f1_4) / 4
        
        return overall_loss, overall_recall, overall_precision, overall_f1 
            
    def forward_predict(self, images, annotations):
        preds = self.build_model(images, ff=False, multi=True)
        return annotations, images, preds