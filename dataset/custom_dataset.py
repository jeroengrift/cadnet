import os
import numpy as np
import geopandas as gp
import torch.utils.data
import rasterio as rio
import pandas as pd

from skimage import io
from shapely.geometry import LineString, box
from rasterio.features import rasterize
from torch.utils.data.dataloader import default_collate

# pd.options.mode.chained_assignment = None
dtype_train = np.float32


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, run_type, cfg, img_root, ref_root, transform=None):      
        self.cfg = cfg
        self.run_type = run_type
        self.img_root = img_root
        self.ref_root = ref_root
                        
        if self.run_type not in self.cfg.MODEL.RUN_TYPES: 
            raise ValueError("Invalid run type. Expected one of: %s" % self.cfg.MODEL.RUN_TYPES)
        
        images = sorted(os.listdir(img_root))
        images_df = pd.DataFrame({'image_id':images})
        print(f'all images: {len(images_df)}')  
        
        # select the correct reference files
        patches = "ps_25" if self.cfg.DATASETS.PATCHES_DIR == "patches_25" else "ps_8"
        sub_tag = ""
        if "sub" in self.cfg.DATASETS.DATA_DIR:
            sub_tag = "sub2" if "sub2" in self.cfg.DATASETS.DATA_DIR else "sub"
            sub_prefix = f"{sub_tag}_"
        else: sub_prefix = ""
        
        base = "reference_cad_aerial_8_"
        fname_brk = f"{base}{sub_prefix}{patches}_brk.gpkg"
        fname_topo = f"{base}{sub_prefix}{patches}_topo.gpkg"
        layer_brk = fname_brk.split(".")[0]
        layer_topo = fname_topo.split(".")[0]
        
        self.brk_reference = gp.read_file(os.path.join(self.ref_root, fname_brk), layer=layer_brk, engine="pyogrio")
        self.topo_reference = gp.read_file(os.path.join(self.ref_root, fname_topo), layer=layer_topo, engine="pyogrio")
        
        print(f'all brk_reference: {len(self.brk_reference)}')      
        print(f'all topo_reference: {len(self.topo_reference)}')
      
        if self.cfg.DATASETS.SELECT_VISIBLE:
            self.brk_reference = self.brk_reference[self.brk_reference['visible']]
            print(f'visible brk_reference: {len(self.brk_reference)}')

        if not self.cfg.DATASETS.INCLUDE_URBAN:
            images_urban = gp.read_file(os.path.join(self.ref_root, 'urban_25.gpkg'), layer='urban_25', engine="pyogrio")
            images_urban = images_urban[images_urban['urban'] == False]
            images_urban = images_urban.reindex(columns=['image_id']).reset_index(drop=True)
            images_df = images_df.merge(images_urban, how='inner', on='image_id')
            print(f'all images after sub1 urban filtering: {len(images_df)}')
                            
        if cfg.DATASETS.TRAIN_SAMPLE and cfg.MODEL.RUN_TYPE == 'train':
            images_df = images_df.sample(frac=cfg.DATASETS.TRAIN_SAMPLE_RATE, random_state=42)
            print(f'images after train sample filtering: {len(images_df)}')

        if cfg.DATASETS.TEST_SAMPLE and cfg.MODEL.RUN_TYPE == 'predict':
            images_df = images_df.sample(frac=cfg.DATASETS.TEST_SAMPLE_RATE, random_state=42)
            print(f'images after test sample filtering: {len(images_df)}')
                        
        if cfg.MODEL.RUN_TYPE == 'train':
            images_df = images_df[images_df['image_id'].isin(self.brk_reference['image_id']) & images_df['image_id'].isin(self.topo_reference['image_id'])]
            print(f'images after reference present filtering: {len(images_df)}')
            
        self.images = images_df['image_id'].tolist()  

        # select only reference that are in the images
        self.topo_reference = self.topo_reference[self.topo_reference['image_id'].isin(self.images)]
        print(f'total topo reference after image filtering: {len(self.topo_reference)}')
        self.brk_reference = self.brk_reference[self.brk_reference['image_id'].isin(self.images)]
        print(f'total brk reference after image filtering: {len(self.brk_reference)}')
        
        print(f'total images: {len(images_df)}')
        print(f'total brk reference: {len(self.brk_reference)}')
        print(f'total topo reference: {len(self.topo_reference)}')
        
        print('---' * 5)
        self.transform = transform
                
    def __getitem__(self, idx_):
       return idx_


class CustomDatasetCad(CustomDataSet):
    def __init__(self, run_type, cfg, img_root, ref_root, transform=None):
        super().__init__(run_type, cfg, img_root, ref_root, transform) 
        
    def pixel_from_28992(self, x,y, img_bounds, output_size):
        # https://gis.stackexchange.com/questions/395293/convert-from-latitude-longitude-to-geotiff-screen-pixel-x-y-coordinates-in-pyth
        px_pc = min(((x - img_bounds.bounds.left) / (img_bounds.bounds.right - img_bounds.bounds.left) * img_bounds.width) / (self.cfg.DATASETS.IMG_SIZE / output_size), output_size - .001)
        py_pc = min(((img_bounds.bounds.top - y) / (img_bounds.bounds.top - img_bounds.bounds.bottom) * img_bounds.height) / (self.cfg.DATASETS.IMG_SIZE / output_size), output_size - .001)
        
        return [px_pc, py_pc]      

    def create_mask(self, ann_ids, resolution, clip_mask, transform):
        ann_ids_copy = ann_ids.copy()
        ann_ids_copy = ann_ids_copy['geometry'].buffer(self.cfg.DATASETS.BUFFER)
        ann_ids_copy = ann_ids_copy.clip(mask=clip_mask) # https://geopandas.org/en/stable/docs/reference/api/geopandas.clip.html    
        geom_buffer = [shapes for shapes in ann_ids_copy.geometry]  
        seg_mask_buffer = rasterize(shapes=geom_buffer, out_shape=(resolution, resolution), transform=transform, all_touched=True)                         
        
        return seg_mask_buffer.astype(dtype_train)

    def create_visibility_mask(self, ann_ids, resolution, clip_mask, transform):
        ann_ids_copy = ann_ids.copy()       
         
        geom_vis = ann_ids_copy[ann_ids_copy['visible'] == True]
        geom_vis = geom_vis['geometry'].buffer(self.cfg.DATASETS.BUFFER)
        geom_vis = geom_vis.clip(mask=clip_mask)     
        geom_vis = [shapes for shapes in geom_vis.geometry]  

        geom_invis = ann_ids_copy[ann_ids_copy['visible'] == False]
        geom_invis = geom_invis['geometry'].buffer(self.cfg.DATASETS.BUFFER)
        geom_invis = geom_invis.clip(mask=clip_mask)
        geom_invis = [shapes for shapes in geom_invis.geometry]
    
        if len(geom_vis) == 0: seg_mask_vis = np.zeros([resolution, resolution])
        else: seg_mask_vis = rasterize(shapes=geom_vis, out_shape=(resolution, resolution), transform=transform, all_touched=True)
        
        if len(geom_invis) == 0: seg_mask_invis = np.zeros([resolution, resolution])
        else: seg_mask_invis = rasterize(shapes=geom_invis, out_shape=(resolution, resolution), transform=transform, all_touched=True)
        
        stacked_vis_invis = np.stack([seg_mask_vis.astype(dtype_train), seg_mask_invis.astype(dtype_train)], axis=0)

        return stacked_vis_invis.astype(dtype_train)
        
    def create_connectivity_cube(self, seg_mask, resolution):
        # create array of zeros, with a larger size than the image size
        img_pad_d1 = np.zeros([resolution + 4, resolution + 4])
        
        # fill the array with the seg mask (with padding of 2 on all sides)
        img_pad_d1[2:-2, 2:-2] = seg_mask
        # create connectivity cube with 8 dimensions
        conn_cube_d1 = np.zeros([8, resolution, resolution])

        # loop over the image both image dimensions
        for i in range(resolution):
            for j in range(resolution):
                # if pixel is 0, than output will alos be 0, you can continue
                if seg_mask[i, j] == 0:
                    continue
                
                # check for every pixel the 8 surrounding pixels and fill the connectivity cube
                conn_cube_d1[0, i, j] = img_pad_d1[i, j + 2]
                conn_cube_d1[1, i, j] = img_pad_d1[i, j - 2]
                conn_cube_d1[2, i, j] = img_pad_d1[i + 2, j]
                conn_cube_d1[3, i, j] = img_pad_d1[i + 2, j + 2]
                conn_cube_d1[4, i, j] = img_pad_d1[i + 2, j - 2]
                conn_cube_d1[5, i, j] = img_pad_d1[i - 2, j]
                conn_cube_d1[6, i, j] = img_pad_d1[i - 2, j + 2]
                conn_cube_d1[7, i, j] = img_pad_d1[i - 2, j - 2]
                         
        # create connectivity cube d3, the same procedure as for the d1 cube
        img_pad_d3 = np.zeros([resolution + 8, resolution + 8])
        img_pad_d3[4:-4, 4:-4] = seg_mask
        conn_cube_d3 = np.zeros([8, resolution, resolution])

        for i in range(resolution):
            for j in range(resolution):
                if seg_mask[i, j] == 0:
                    continue
                
                conn_cube_d3[0, i, j] = img_pad_d3[i, j + 4]
                conn_cube_d3[1, i, j] = img_pad_d3[i, j - 4]
                conn_cube_d3[2, i, j] = img_pad_d3[i + 4, j]
                conn_cube_d3[3, i, j] = img_pad_d3[i + 4, j + 4]
                conn_cube_d3[4, i, j] = img_pad_d3[i + 4, j - 4]
                conn_cube_d3[5, i, j] = img_pad_d3[i - 4, j]
                conn_cube_d3[6, i, j] = img_pad_d3[i - 4, j + 4]
                
        return conn_cube_d1.astype(dtype_train), conn_cube_d3.astype(dtype_train)

    def create_reference_vector(self, ann_ids, img_bounds, resolution):
        reference_vector = []
        
        for index, row in ann_ids.iterrows():
            coords = [[round(c, 3) for c in cp] for cp in row.geometry.coords[:]]
            p1 = self.pixel_from_28992(coords[0][0], coords[0][1], img_bounds, resolution)
            p2 = self.pixel_from_28992(coords[1][0], coords[1][1], img_bounds, resolution)
            coords = (p1, p2)
            reference_vector.append((LineString(coords), row['visible'], img_bounds.bounds.left, img_bounds.bounds.top, img_bounds.bounds.right, img_bounds.bounds.bottom))
            
        return reference_vector
    
    def __getitem__(self, idx_):
        resolution = 512
        img_id = self.images[idx_]
        filename = img_id.replace('.tif', '')
        image = io.imread(os.path.join(self.img_root, img_id)).astype(dtype_train)[:, :, :3] / 255.0
        img_bounds = rio.open(os.path.join(self.img_root, img_id), crs='epsg:28992')
    
        scales = {1: "512", 2: "256", 4: "128", 8: "64", 16: "32"}
        ann = {}
        for factor, suffix in scales.items():
            size = resolution // factor

            ann[f'gt_bin_brk_{suffix}_visibility'] = np.zeros((2, size, size)).astype(dtype_train)
            ann[f'gt_bin_brk_{suffix}'] = np.zeros((size, size)).astype(dtype_train)
            ann[f'gt_bin_topo_{suffix}'] = np.zeros((size, size)).astype(dtype_train)
            
            ann[f'gt_cc_d1_brk_{suffix}_vis_'] = np.zeros((8, size, size)).astype(dtype_train)
            ann[f'gt_cc_d3_brk_{suffix}_vis_'] = np.zeros((8, size, size)).astype(dtype_train)
            ann[f'gt_cc_d1_brk_{suffix}_inv_'] = np.zeros((8, size, size)).astype(dtype_train)
            ann[f'gt_cc_d3_brk_{suffix}_inv_'] = np.zeros((8, size, size)).astype(dtype_train)
            
            ann[f'gt_cc_d1_brk_{suffix}'] = np.zeros((8, size, size)).astype(dtype_train)
            ann[f'gt_cc_d3_brk_{suffix}'] = np.zeros((8, size, size)).astype(dtype_train)
            
            ann[f'gt_cc_d1_topo_{suffix}'] = np.zeros((8, size, size)).astype(dtype_train)
            ann[f'gt_cc_d3_topo_{suffix}'] = np.zeros((8, size, size)).astype(dtype_train)
        
        brk_reference_vector_all = []
        topo_reference_vector_all = []
        
        brk_ann_ids = self.brk_reference[self.brk_reference['image_id'] == img_id]
        
        if len(brk_ann_ids) > 0:  
            if self.cfg.MODEL.RUN_TYPE == 'predict':
                brk_reference_vector_all = self.create_reference_vector(brk_ann_ids, img_bounds, resolution)
            
            for res in self.cfg.DATASETS.RESOLUTIONS:  
                clip_mask = box(*img_bounds.bounds) # https://gis.stackexchange.com/questions/352445/make-shapefile-from-raster-bounds-in-python
                transform = rio.transform.from_bounds(*img_bounds.bounds, width=res, height=res)
                
                brk_mask = self.create_mask(brk_ann_ids, res, clip_mask, transform)
                ann[f'gt_bin_brk_{res}'] = brk_mask
                
                brk_connect = self.create_connectivity_cube(brk_mask, res)
                ann[f'gt_cc_d1_brk_{res}'] = brk_connect[0]
                ann[f'gt_cc_d3_brk_{res}'] = brk_connect[1]   
                
                brk_vis_mask = self.create_visibility_mask(brk_ann_ids, res, clip_mask, transform)
                ann[f'gt_bin_brk_{res}_visibility'] = brk_vis_mask

                brk_connect_vis = self.create_connectivity_cube(brk_vis_mask[0,:,:], res)
                ann[f'gt_cc_d1_brk_{res}_vis_'] = brk_connect_vis[0]
                ann[f'gt_cc_d3_brk_{res}_vis_'] = brk_connect_vis[1]   
                
                brk_connect_inv = self.create_connectivity_cube(brk_vis_mask[1,:,:], res)
                ann[f'gt_cc_d1_brk_{res}_inv_'] = brk_connect_inv[0]
                ann[f'gt_cc_d3_brk_{res}_inv_'] = brk_connect_inv[1]
                    
        topo_ann_ids = self.topo_reference[self.topo_reference['image_id'] == img_id]
        
        if len(topo_ann_ids) > 0:  
            if self.cfg.MODEL.RUN_TYPE == 'predict':
                topo_reference_vector_all = self.create_reference_vector(topo_ann_ids, img_bounds, resolution)
                
            for res in self.cfg.DATASETS.RESOLUTIONS:  
                clip_mask = box(*img_bounds.bounds) # https://gis.stackexchange.com/questions/352445/make-shapefile-from-raster-bounds-in-python
                transform = rio.transform.from_bounds(*img_bounds.bounds, width=res, height=res)
                topo_mask = self.create_mask(topo_ann_ids, res, clip_mask, transform)
                ann[f'gt_bin_topo_{res}'] = topo_mask
                topo_connect = self.create_connectivity_cube(topo_mask, res)
                ann[f'gt_cc_d1_topo_{res}'] = topo_connect[0]
                ann[f'gt_cc_d3_topo_{res}'] = topo_connect[1]   
                
        if self.transform is not None:
            return self.transform(image, ann, [brk_reference_vector_all, topo_reference_vector_all], filename)
        
        return image, ann, [brk_reference_vector_all, topo_reference_vector_all], filename

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]), 
            [b[2] for b in batch],
            [b[3] for b in batch]
        )

    