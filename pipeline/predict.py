# code partly derived from: https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning

import numpy as np
import skan
import skimage
import skimage.morphology
import shapely.geometry
import torch
import shutil
import cv2
import os 
import geopandas as gp
import pandas as pd
import csv
import math
import networkx as nx
import rasterio
import geopandas as gp
import time

from sklearn.metrics import recall_score, precision_score, f1_score
from rasterio.crs import CRS
from shapely.geometry import Point, LineString, MultiPoint, box
from shapely import line_merge
from shapely.ops import split, substring
from torchview import draw_graph
from utils.visualizer import plot_features
from utils.comm import to_single_device
from typing import List
from models.framefield.torch_lydorn.torchvision.transforms import Skeleton
from shapely.geometry import LineString

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))

pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

# code: https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely/62994304#62994304
def create_segments(line, max_segment_length):
    mp = shapely.geometry.MultiPoint()
    for i in np.arange(0, line.length, max_segment_length):
        s = substring(line, i, i+max_segment_length)
        mp = mp.union(s.boundary)
    return mp    

def segment_geometries(geo_df, max_seg_length=10):    
    segmented_geoms = []
    for idx, row in geo_df.iterrows():
        line = row.geometry
        seg_points = create_segments(line, max_seg_length)
        seg_points = [p for p in seg_points.geoms]
        seg_points = sorted(seg_points, key=lambda p: line.project(p))
        for start, end in zip(seg_points, seg_points[1:]):
            segmented_geoms.append(shapely.geometry.LineString([start, end]))
    return segmented_geoms 

def extract_endpoints(geom):
    if geom.geom_type == 'LineString' and len(geom.coords) >= 2:
        return shapely.geometry.LineString([geom.coords[0], geom.coords[-1]])
    else: 
        return geom

def calculate_azimuth(geo_df):
    azimuth_list = []
    geo_df = geo_df[geo_df.geometry.type == 'LineString']

    for i, row in enumerate(geo_df.itertuples()):
        if isinstance(row.geometry, LineString):
            point_list = list(row.geometry.coords)
            x0, x1 = point_list[0][0], point_list[1][0]
            y0, y1 = point_list[0][1], point_list[1][1]
            azimuth = math.atan2((x1 - x0), (y1 - y0))

            if azimuth > math.pi:
                azimuth -= math.pi
            elif 0 > azimuth > -math.pi:
                azimuth += math.pi
            elif -math.pi > azimuth > -(2 * math.pi):
                azimuth += (2 * math.pi)
            azimuth_list.append(azimuth)

    geo_df['azimuth'] = azimuth_list

    return geo_df

def pair_points(list):
    for i in range(0, len(list) - 1):
        yield list[i-1], list[i]

def chop_lines(gdf): 
    gdf_chopped = {'geometry': []}

    for i in range(len(gdf)):
        geom_column = gdf.geometry[i]
        if isinstance(geom_column, LineString):
            linestring = gdf.geometry[i]
            coordinates = linestring.coords    
        elif isinstance(geom_column, gp.geoseries.GeoSeries):
            linestring = gdf.geometry[i].iloc[0]
            coordinates = list(gdf.geometry[i].iloc[0].coords)
                            
        points = []
        
        for seg_start, seg_end in pair_points(coordinates):
            line_start = Point(seg_start)
            line_end = Point(seg_end) 
            points.append(line_start)
            points.append(line_end)

        chop_points = MultiPoint(points)       
        geom_collection = list(split(linestring, chop_points).geoms)

        for j in range(len(geom_collection)):    
            result = LineString(geom_collection[j])
            gdf_chopped['geometry'].append(result)
        
    return gp.GeoDataFrame.from_dict(gdf_chopped)

def concatenate_lines(gdf, threshold):   
    polylines_azimuths = calculate_azimuth(gdf)
    polylines_azimuths["id"] = polylines_azimuths.index + 1

    polylines_azimuths_nn = gp.sjoin(polylines_azimuths, polylines_azimuths,
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right")
    
    polylines_azimuths_nn = polylines_azimuths_nn[polylines_azimuths_nn.index != polylines_azimuths_nn.index_right]
    polylines_azimuths_nn = polylines_azimuths_nn.drop(columns=['index_right'])

    polylines_azimuths_nn = polylines_azimuths_nn[(polylines_azimuths_nn.azimuth_left <= (polylines_azimuths_nn.azimuth_right + threshold)) 
            & (polylines_azimuths_nn.azimuth_left >= (polylines_azimuths_nn.azimuth_right - threshold))].reset_index().drop(columns=['index'])

    graph_list = [sorted((row.id_left, row.id_right)) for _, row in polylines_azimuths_nn.iterrows()]

    if len(graph_list) > 0:
        connected_components = list(map(sorted, nx.connected_components(nx.Graph(graph_list))))
        connected_components = list([i, def_instance] for i, def_instances in enumerate(connected_components)
                for def_instance in def_instances)
        
        connected_components = pd.DataFrame(connected_components).rename(columns={0: 'graph_id', 1: 'seg_id'})
        connected_components['graph'] = True
        connected_components = gp.GeoDataFrame(polylines_azimuths.merge(connected_components, left_on='id', right_on='seg_id', how='left'), geometry='geometry')
        connected_components.fillna({'graph': False}, inplace=True)
        
        out_graph = connected_components[~connected_components['graph']].reset_index().drop(columns=['index'])
        in_graph = connected_components[connected_components['graph']].reset_index().drop(columns=['index'])

        agg_segments = in_graph.dissolve(by='graph_id').drop(columns=['seg_id', 'azimuth', 'id', 'graph'], axis=1).reset_index()
        agg_segments['geometry'] = agg_segments['geometry'].apply(line_merge)

        agg_segments_list = pd.DataFrame(in_graph.groupby('graph_id')['seg_id'].apply(list))
        agg_segments = agg_segments.merge(agg_segments_list, on='graph_id', how='left')

        merged_segments = pd.concat([agg_segments, out_graph], sort=True) \
            .drop(columns=['id', 'azimuth'], axis=1).reset_index() \
            .drop(columns=['index'])
        
        merged_segments = merged_segments.reset_index().drop(columns=['index'])
        
        return merged_segments

    else:
        return gdf


class BasePolygonizer:
    def __init__(self, cfg, file_output_dir):
        self.cfg = cfg
        self.file_output_dir = file_output_dir

    def skeleton_to_polylines(self, skeleton: Skeleton) -> List[np.ndarray]:
        polylines = []
        for path_i in range(skeleton.paths.indptr.shape[0] - 1):
            start, stop = skeleton.paths.indptr[path_i:path_i + 2]
            path_indices = skeleton.paths.indices[start:stop]
            path_coordinates = skeleton.coordinates[path_indices]
            polylines.append(path_coordinates)
        return polylines

    def compute_skeletons(self, seg_batch, cfg) -> List[Skeleton]:
        assert len(seg_batch.shape) == 4 and seg_batch.shape[1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

        corrected_edge_mask_batch = cfg.DATASETS.BINARY_TRESHOLD < seg_batch[:, 0, :, :]
        np_corrected_edge_mask_batch = corrected_edge_mask_batch.cpu().numpy()
        np_corrected_edge_mask_batch = np_corrected_edge_mask_batch.squeeze() 
        
        np_edge_mask_padded = np.pad(np_corrected_edge_mask_batch, pad_width=2, mode="edge")
        np_edge_mask_padded = skimage.morphology.binary_closing(np_edge_mask_padded)
        skeleton_image = skimage.morphology.skeletonize(np_edge_mask_padded)
        skeleton_image = skeleton_image[2:-2, 2:-2]

        skeleton = Skeleton()
        if 0 < skeleton_image.sum():
            try:
                skeleton = skan.Skeleton(skeleton_image, keep_images=False)            
                skeleton.coordinates = skeleton.coordinates[:skeleton.paths.indices.max() + 1]
                if skeleton.coordinates.shape[0] != skeleton.degrees.shape[0]:
                    raise ValueError(f"skeleton.coordinates.shape[0] = {skeleton.coordinates.shape[0]} while skeleton.degrees.shape[0] = {skeleton.degrees.shape[0]}. They should be of same size.")
            except ValueError as e:
                print(e)
                
        return skeleton


class PolygonizerSimple(BasePolygonizer):
    def __init__(self, cfg, file_output_dir):
        super().__init__(cfg, file_output_dir)

    def __call__(self, seg_batch):
        assert len(seg_batch.shape) == 4 and seg_batch.shape[
            1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

        seg_batch = seg_batch.to(self.cfg.MODEL.DEVICE)
        skeletons_batch = self.compute_skeletons(seg_batch, self.cfg)
        polylines_batch = [self.skeleton_to_polylines(skeleton) for skeleton in [skeletons_batch]]
        
        return polylines_batch
    

class PredictRaster():
    def __init__(self, cfg, output_dir, model=None, test_data=None, weights_path=None):
        self.cfg = cfg
        self.model = model
        self.test_data = test_data
        self.weights_path = weights_path
        self.output_dir = output_dir    
        self.use_vis = ''
        self.vis_types = {'': 0}
        self.pred_dir = os.path.join(self.output_dir, f'{self.cfg.DATASETS.PRED_DIR}_bt-{self.cfg.DATASETS.BINARY_TRESHOLD}')
        self.output_mosaic_dir = os.path.join(self.output_dir, self.cfg.DATASETS.MOSAIC_DIR + '_overlap_' + str(self.cfg.MODEL.TEST_OVERLAP_50))
        
        self.subtype = '_sub2' if '_sub2' in cfg.DATASETS.DATA_DIR else '_sub'
        self.tile_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.IMAGES_DIR)
        # self.tile_dir = os.path.join(cfg.DATASETS.DATA_DIR.replace(self.subtype, ''), cfg.DATASETS.IMAGES_DIR)
            
    def write(self):  
        device = self.cfg.MODEL.DEVICE
        checkpoint = torch.load(self.weights_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(device)
        
        if os.path.isdir(self.output_mosaic_dir): 
            pass
        else: 
            os.makedirs(self.output_mosaic_dir, exist_ok=True) 
        
        if self.cfg.MODEL.WRITE_PATCH_PREDS:
            if os.path.isdir(self.pred_dir):
                shutil.rmtree(self.pred_dir)
            os.makedirs(self.pred_dir, exist_ok=True)
        
        for tile in self.cfg.MODEL.MOSAIC_TILE:
            print(f'Processing tile {tile}')
            if self.cfg.MODEL.WRITE_MOSAIC_RASTER:
                mosaics = {}    
                tile_raster_open = rasterio.open(os.path.join(self.tile_dir, f'{tile}.tif'))
                tile_extents = tile_raster_open.bounds
                tile_width = 40000
                tile_height = 40000
                tile_transform = rasterio.transform.from_bounds(*tile_extents, width=tile_width,
                                                        height=tile_height)
                def create_layer_dict():
                    base = {}
                    
                    fp_array = np.empty((tile_width, tile_height), dtype=np.float32)
                    fp_array.fill(np.nan)
                    base[''] = fp_array

                    count_array = np.empty((tile_width, tile_height), dtype=np.uint8)
                    count_array.fill(0)
                    base['count_array'] = count_array
                    
                    ui_array = np.empty((tile_width, tile_height), dtype=np.uint8)
                    ui_array.fill(0)
                    base['_gt_'] = ui_array
                                    
                    return base
                
                if self.cfg.MODEL.USE_BRK: 
                    mosaics['brk'] = create_layer_dict()
                    
            if self.cfg.MODEL.WRITE_PATCH_PREDS:
                if os.path.isdir(self.pred_dir):
                    shutil.rmtree(self.pred_dir)
                os.makedirs(self.pred_dir, exist_ok=True)
                
            def iterate_data(split_gt=False, gt=False):
                for it, (test_images, test_annotations, test_ann_ids, test_filenames) in enumerate(self.test_data):
                    filename = test_filenames[0]  
                                    
                    if self.cfg.MODEL.WRITE_MOSAIC_RASTER:
                        if not tile.replace('tile-', '') in filename:
                            continue
                        
                    print(f"[{it:03d}] Processing file: {filename}")

                    test_images = test_images.to(device)
                    test_annotations = to_single_device(test_annotations, device)

                    with torch.no_grad():
                        preds = self.model(test_images, test_annotations)            

                    if it == 0:
                        model_graph = draw_graph(self.model, input_data=(test_images, test_annotations)).visual_graph
                        graph_svg = model_graph.pipe(format='png')
                        save_path = os.path.join(self.output_dir, 'model_graph.png')
                        with open(save_path, 'wb') as f:
                            f.write(graph_svg)
                    
                    pred_output_dir = os.path.join(self.pred_dir, filename)  
                    if self.cfg.MODEL.WRITE_PATCH_PREDS and it % self.cfg.MODEL.WRITE_PATCH_PREDS_SAMPLE_RATE == 0:
                        if pred_output_dir:
                            if os.path.isdir(pred_output_dir):
                                shutil.rmtree(pred_output_dir)
                            os.makedirs(pred_output_dir, exist_ok=True)
                    
                    def process_raster(type_, it, separate=True, concatenate=True):                        
                        assert type_ in ['brk', 'topo']
                        
                        print(f'Processing {type_} {filename}')
                        gt_type = type_
                            
                        if self.cfg.MODEL.USE_MULTI:
                            resolutions = self.cfg.DATASETS.RESOLUTIONS
                            resolutions = resolutions[::-1]
                            resolutions = [f's{res}' for res in resolutions]
                            resolutions.append(512)
                        else:
                            resolutions = [512]
                        
                        self.use_vis = ''
                        self.vis_types = {'': 0}
                                                    
                        for resolution in resolutions:
                            if self.cfg.MODEL.WRITE_PATCH_PREDS: pass
                            elif resolution == 512: pass
                            else: continue
                            
                            if self.cfg.MODEL.WRITE_PATCH_PREDS and it % self.cfg.MODEL.WRITE_PATCH_PREDS_SAMPLE_RATE == 0:
                                cv2.imwrite(os.path.join(pred_output_dir, f'{filename}.jpeg'), test_images[0].permute(1, 2, 0).cpu().detach().numpy() * 255)
                            for vis_type, vis_channel in self.vis_types.items():
                                pred_bin_edge = preds[f"pred_bin_{type_}_{resolution}{vis_type}"].squeeze().unsqueeze(0).unsqueeze(0)
                                
                                if resolution == 512:
                                    gt_bin_edge = test_annotations[f"gt_bin_{gt_type}_{resolution}{self.use_vis}"]
                                    if gt_bin_edge.dim() == 4:
                                        gt_bin_edge = gt_bin_edge.squeeze().unsqueeze(0)[:,vis_channel,:,:].unsqueeze(0)
                                    elif gt_bin_edge.dim() == 3:
                                        gt_bin_edge = gt_bin_edge.unsqueeze(0)[:,vis_channel,:,:].unsqueeze(0)
                                else:
                                    gt_bin_edge = test_annotations[f"gt_bin_{gt_type}_{resolution[1:]}{self.use_vis}"]
                                    if gt_bin_edge.dim() == 4:
                                        gt_bin_edge = gt_bin_edge.squeeze().unsqueeze(0)[:,vis_channel,:,:].unsqueeze(0)
                                    elif gt_bin_edge.dim() == 3:
                                        gt_bin_edge = gt_bin_edge.unsqueeze(0)[:,vis_channel,:,:].unsqueeze(0)
                                                                
                                pred_grad_edge = pred_bin_edge.clone()  
                                pred_bin_edge[pred_bin_edge >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                                pred_bin_edge[pred_bin_edge < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0  

                                if self.cfg.MODEL.WRITE_PATCH_PREDS and it % self.cfg.MODEL.WRITE_PATCH_PREDS_SAMPLE_RATE == 0:
                                    plot_features(gt_bin_edge, os.path.join(pred_output_dir, f'gt_bin_edge_{type_}_{resolution}{vis_type}.jpeg'))                   
                                    plot_features(pred_grad_edge, os.path.join(pred_output_dir, f'pred_gradient_edge_{type_}_{resolution}{vis_type}.jpeg'))
                                    plot_features(pred_bin_edge, os.path.join(pred_output_dir, f'pred_bin_edge_{type_}_{resolution}{vis_type}.jpeg'))   
                                    
                                if self.cfg.MODEL.USE_COA:
                                    if resolution == 512: 
                                        gt_connect_d1 = test_annotations[f'gt_cc_d1_{gt_type}_{resolution}{vis_type}'].float()
                                        gt_connect_d3 = test_annotations[f'gt_cc_d3_{gt_type}_{resolution}{vis_type}'].float()
                                    else:
                                        gt_connect_d1 = test_annotations[f'gt_cc_d1_{gt_type}_{resolution[1:]}{vis_type}'].float()
                                        gt_connect_d3 = test_annotations[f'gt_cc_d3_{gt_type}_{resolution[1:]}{vis_type}'].float()
                                                                                                        
                                    pred_connect_d1 = preds[f'pred_cc_d1_{type_}_{resolution}{vis_type}']
                                    pred_connect_d3 = preds[f'pred_cc_d3_{type_}_{resolution}{vis_type}']
                                    
                                    pred_connect_d1_grad = pred_connect_d1.clone()
                                    pred_connect_d1[pred_connect_d1 >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                                    pred_connect_d1[pred_connect_d1 < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0
                                    
                                    pred_connect_d3_grad = pred_connect_d3.clone()
                                    pred_connect_d3[pred_connect_d3 >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                                    pred_connect_d3[pred_connect_d3 < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0
                                    
                                    pred_bin_edge = torch.cat((pred_bin_edge, pred_connect_d1, pred_connect_d3), dim=1)
                                    pred_bin_edge = torch.amax(pred_bin_edge, axis=1).unsqueeze(1)
                                        
                                    pred_grad_edge = torch.cat((pred_grad_edge, pred_connect_d1_grad, pred_connect_d3_grad), dim=1)
                                    pred_grad_edge = torch.amax(pred_grad_edge, axis=1).unsqueeze(1)
                                    
                                    if self.cfg.MODEL.WRITE_PATCH_PREDS and it % self.cfg.MODEL.WRITE_PATCH_PREDS_SAMPLE_RATE == 0:
                                        # write connectivity cube masks
                                        plot_features(gt_connect_d1, os.path.join(pred_output_dir, f'gt_cc_d1_{gt_type}_{resolution}{vis_type}.jpeg'))
                                        plot_features(gt_connect_d3, os.path.join(pred_output_dir, f'gt_cc_d3_{gt_type}_{resolution}{vis_type}.jpeg'))
                                        
                                        # write connectivity cube predictions
                                        plot_features(pred_connect_d1, os.path.join(pred_output_dir, f'pred_cc_d1_{type_}_{resolution}{vis_type}.jpeg'))
                                        plot_features(pred_connect_d3, os.path.join(pred_output_dir, f'pred_cc_d3_{type_}_{resolution}{vis_type}.jpeg'))
                                        
                                        # write final predictions
                                        plot_features(pred_grad_edge, os.path.join(pred_output_dir, f'pred_gradient_seg_coa_{type_}_{resolution}{vis_type}.jpeg'))
                                        plot_features(pred_bin_edge, os.path.join(pred_output_dir, f'pred_bin_seg_coa_{resolution}_{resolution}{vis_type}.jpeg'))
                                
                                if self.cfg.MODEL.WRITE_MOSAIC_RASTER and resolution == 512: 
                                    parts = filename.split("_")
                                    c_off = int(parts[1])
                                    r_off = int(parts[2])
                                    
                                    def update_mosaic(target, patch, img_size, index=None):
                                        region = target[r_off: r_off + img_size, c_off: c_off + img_size]
                                        if region.shape[:2] != patch.shape[:2]:
                                            patch = cv2.resize(patch, dsize=(region.shape[1], region.shape[0]), interpolation=cv2.INTER_CUBIC)
                                        combined = np.nansum(np.dstack((patch, region)), axis=2)
                                        target[r_off: r_off + img_size, c_off: c_off + img_size] = combined
                                        
                                        if index is not None:
                                            region_index = index[r_off: r_off + img_size, c_off: c_off + img_size]
                                            patch_index = np.empty((512, 512), dtype=np.uint8)
                                            patch_index.fill(1) 
                                            if region_index.shape[:2] != patch_index.shape[:2]:
                                                patch_index = cv2.resize(patch_index, dsize=(region_index.shape[1], region_index.shape[0]), interpolation=cv2.INTER_CUBIC)
                                                                                         
                                            combined_index = np.sum(np.dstack((patch_index, region_index)), axis=2)
                                            index[r_off: r_off + img_size, c_off: c_off + img_size] = combined_index

                                    d_arr = pred_grad_edge.squeeze().cpu().numpy().astype(np.float32)
                                    d_arr_gt = gt_bin_edge.squeeze().cpu().numpy().astype(np.uint8)
                                    
                                    if separate:
                                        if not gt or not split_gt:            
                                            update_mosaic(mosaics[type_][vis_type], d_arr, self.cfg.DATASETS.IMG_SIZE, mosaics[type_][f'{vis_type}count_array'])
                                        if gt or not split_gt:      
                                            update_mosaic(mosaics[type_][f'_gt_{vis_type}'], d_arr_gt, self.cfg.DATASETS.IMG_SIZE)
                            
                    if self.cfg.MODEL.USE_BRK: 
                        process_raster('brk', it, separate=True, concatenate=True)
            
            if self.cfg.MODEL.WRITE_MOSAIC_RASTER or self.cfg.MODEL.WRITE_PATCH_PREDS:
                iterate_data(split_gt=False ,gt=False)
        
            if self.cfg.MODEL.WRITE_MOSAIC_RASTER:
                    types = ['brk']
                    for type_ in types:
                        if not getattr(self.cfg.MODEL, f'USE_{type_.upper()}'): 
                            continue  
                        
                        mosaics[type_][''] = mosaics[type_][''] / mosaics[type_]['count_array']
                            
                        for vis_type in mosaics[type_]:
                            if '_gt_' not in vis_type and 'count_array' not in vis_type:    
                                                        
                                with rasterio.open(os.path.join(self.output_mosaic_dir, tile + '_' + type_ + vis_type + '.tif'),
                                        mode='w',
                                        driver='GTiff',
                                        height=mosaics[type_][vis_type].shape[0],
                                        width=mosaics[type_][vis_type].shape[1],
                                        transform=tile_transform,
                                        count=1,
                                        dtype='float32',
                                        crs=CRS.from_epsg(32648)
                                        ) as dest:
                                    dest.write(mosaics[type_][vis_type], 1)
                                    dest.close()    
                                    
                                with rasterio.open(os.path.join(self.output_mosaic_dir, tile + '_' + type_ + vis_type + '_gt_' + '.tif'),
                                                mode='w',
                                                driver='GTiff',
                                                height=mosaics[type_][f'_gt_{vis_type}'].shape[0],
                                                width=mosaics[type_][f'_gt_{vis_type}'].shape[1],
                                                transform=tile_transform,
                                                count=1,
                                                dtype='uint8',
                                                crs=CRS.from_epsg(32648)
                                                ) as dest:
                                    dest.write(mosaics[type_][f'_gt_{vis_type}'], 1)
                                    dest.close()    


    def evaluate(self):
        for tile in self.cfg.MODEL.MOSAIC_TILE:
            raster_stats_file = os.path.join(self.output_mosaic_dir, f'raster_stats_{tile}_{self.cfg.DATASETS.BINARY_TRESHOLD}.csv')
            with open(raster_stats_file, 'w') as raster_csv_file:                
                raster_writer = csv.writer(raster_csv_file)
                raster_writer.writerow(['filename', 'type', 'precision', 'recall', 'f1', 'vis_type'])
                
                for type_ in ['brk']:
                    if not getattr(self.cfg.MODEL, f"USE_{type_.upper()}"): continue

                    for vis_type in ['', '_vis_', '_inv_']:
                        pred_path = os.path.join(self.output_mosaic_dir, f"{tile}_{type_}{vis_type}.tif")
                        gt_path = os.path.join(self.output_mosaic_dir, f"{tile}_{type_}{vis_type}_gt_.tif")
                        
                        try:
                            pred = rasterio.open(pred_path).read(1)
                            gt = rasterio.open(gt_path).read(1)
                        except:
                            continue   
                        
                        pred = np.where(np.isnan(pred), 0, pred)
                        pred[pred >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                        pred[pred < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0
                        pred = pred.astype(np.uint8).flatten()

                        gt = np.where(np.isnan(gt), 0, gt)
                        gt[gt >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                        gt[gt < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0
                        gt = gt.astype(np.uint8).flatten()
                        
                        print(f'Calculating stats for {type_} {vis_type}')
                        print(f'Calculating recall for {type_} {vis_type}')
                        recall = recall_score(gt, pred)
                        print(f'Calculating precision for {type_} {vis_type}')
                        precision = precision_score(gt, pred)
                        print(f'Calculating f1 for {type_} {vis_type}')
                        f1 = f1_score(gt, pred)
                                                                            
                        raster_writer.writerow([tile, type_, precision, recall, f1, vis_type])

             
class PredictVector():
    def __init__(self, cfg, output_dir):  
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_mosaic_dir = os.path.join(self.output_dir, self.cfg.DATASETS.MOSAIC_DIR + '_overlap_' + str(self.cfg.MODEL.TEST_OVERLAP_50))
        self.output_dir_poly = os.path.join(self.output_mosaic_dir, self.cfg.MODEL.TYPE_POLY_PRED)
        
        self.subtype = '_sub2' if '_sub2' in cfg.DATASETS.DATA_DIR else '_sub'
        self.tile_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.IMAGES_DIR)
        # self.tile_dir = os.path.join(cfg.DATASETS.DATA_DIR.replace(self.subtype, ''), cfg.DATASETS.IMAGES_DIR)

        if self.output_dir_poly:
            if os.path.isdir(self.output_dir_poly): pass
            else: os.makedirs(self.output_dir_poly, exist_ok=True)
            
        if self.cfg.MODEL.TEST_OVERLAP_50:
            self.images_urban = gp.read_file(os.path.abspath(os.path.join(parent_dir, 'data', f'cad_aerial_8{self.subtype}', 'reference', 'overlap_50', 'urban_25.gpkg')))
        else: 
            self.images_urban = gp.read_file(os.path.abspath(os.path.join(parent_dir, 'data', f'cad_aerial_8{self.subtype}', 'reference', 'urban_25.gpkg')))

        self.images_urban = self.images_urban[self.images_urban['urban'] == False] 
        self.images_urban = self.images_urban.dissolve()
                
    def write_evaluate(self): 
        for tile in self.cfg.MODEL.MOSAIC_TILE:
            start_time = time.time()
            
            vector_stats_file = os.path.join(self.output_dir_poly, f'polis_{tile}.csv')
            with open(vector_stats_file, 'w') as vector_csv_file:
                vector_writer = csv.writer(vector_csv_file)
                vector_writer.writerow(['type', 'vis_type', 'vector_precision', 'vector_recall', 'vector_f1', 'norm_gt_discrepancy', 'norm_pred_discrepancy'])   

                resolution = 512
                tile_raster_open = rasterio.open(os.path.join(self.tile_dir, f'{tile}.tif'), crs='epsg:28992')
                clip_mask = box(*tile_raster_open.bounds)
                    
                for type_ in ['brk']:
                    if not getattr(self.cfg.MODEL, f"USE_{type_.upper()}"): continue

                    reference_file = os.path.abspath(os.path.join(parent_dir, 'data', 'cad_reference', f'{type_}_reference.gpkg'))
                    gt_polylines = gp.read_file(reference_file, layer=f'{type_}_reference', engine="pyogrio", bbox=clip_mask)
                    gt_polylines = gt_polylines[gt_polylines['filename_10k'] == tile]
                                    
                    for vis_type in ['', '_vis_', '_inv_']:
                        pred_path = os.path.join(self.output_mosaic_dir, f"{tile}_{type_}{vis_type}.tif")                    
                        
                        try: mosaic = rasterio.open(pred_path)
                        except: continue
                                            
                        if vis_type == '':
                            gt_polylines_visibility = gt_polylines
                        elif vis_type == '_vis_':
                            gt_polylines_visibility = gt_polylines[gt_polylines['visible'] == 'True']
                        elif vis_type == '_inv_':
                            gt_polylines_visibility = gt_polylines[gt_polylines['visible'] == 'False']  
                                                
                        left = mosaic.bounds[0]
                        bottom = mosaic.bounds[1]
                        mosaic = mosaic.read(1)
                        
                        mosaic = np.expand_dims(mosaic, axis=0)
                        mosaic = torch.from_numpy(mosaic).float().to(self.cfg.MODEL.DEVICE).unsqueeze(0)

                        polygonizer = PolygonizerSimple(self.cfg, file_output_dir=self.output_dir_poly)
                        polylines_batch = polygonizer(mosaic)
                            
                        out_polylines = []
                        for polylines in polylines_batch:
                            for polyline in polylines:
                                line_string = shapely.geometry.LineString(polyline[:, ::-1])
                                line_string = line_string.simplify(self.cfg.POLY.TOLERANCE, preserve_topology=False)
                                out_polylines.append(line_string) 

                        pixel_size = 128 / resolution

                        pred_polylines = gp.GeoDataFrame({'geometry':out_polylines}, geometry='geometry')  
                                    
                        swapped_pred_polylines = []
                        for line in pred_polylines.geometry:
                            try:
                                swapped_coords = [(point[0], 40000 - point[1]) for point in list(line.coords)] 
                                scaled_coords = [shapely.geometry.Point((point[0] * pixel_size) + left , (point[1] * pixel_size) + bottom) for point in swapped_coords] 
                                swapped_line = shapely.geometry.LineString(scaled_coords)
                                swapped_pred_polylines.append(swapped_line) 
                            except Exception as e:
                                print(e)
                                continue 
                            
                        rd_pred_polylines = gp.GeoDataFrame({'geometry':swapped_pred_polylines}, geometry='geometry', crs='EPSG:28992')
                        rd_pred_polylines['id'] = rd_pred_polylines.index + 1
                        rd_pred_polylines.to_file(os.path.join(self.output_dir_poly, f'pred_polylines_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")                          
                        
                        if self.cfg.MODEL.TYPE_POLY_PRED == 'snip' or self.cfg.MODEL.TYPE_POLY_PRED == 'concat':
                            choppped_pred_polylines = chop_lines(rd_pred_polylines)
                            choppped_pred_polylines = choppped_pred_polylines.explode(index_parts=False)
                            
                            concat_pred_polylines = concatenate_lines(choppped_pred_polylines, self.cfg.MODEL.CONCAT_ANGLE) 
                            concat_pred_polylines = concat_pred_polylines.explode(index_parts=False)
                            concat_pred_polylines = concat_pred_polylines.geometry    
                            concat_pred_polylines = concat_pred_polylines.apply(lambda g: extract_endpoints(g))
                            concat_pred_polylines = gp.GeoDataFrame({'geometry':concat_pred_polylines}, geometry='geometry', crs='EPSG:28992').reset_index(drop=True)
                            concat_pred_polylines['id'] = concat_pred_polylines.index + 1
                            concat_pred_polylines.to_file(os.path.join(self.output_dir_poly, f'pred_concat_polylines_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")
                            
                        if not self.cfg.DATASETS.INCLUDE_URBAN: 
                            gt_polylines_visibility = gt_polylines_visibility.clip(self.images_urban)

                        gt_polylines_visibility = gt_polylines_visibility.explode(index_parts=False)
                        gt_polylines_visibility['id'] = gt_polylines_visibility.index + 1
                        gt_polylines_visibility.to_file(os.path.join(self.output_dir_poly, f'gt_polylines_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")

                        if self.cfg.MODEL.TYPE_POLY_PRED == 'snip' or self.cfg.MODEL.TYPE_POLY_PRED == 'concat':
                            concat_gt_polylines = concatenate_lines(gt_polylines_visibility, self.cfg.MODEL.CONCAT_ANGLE)
                            concat_gt_polylines = concat_gt_polylines.explode(index_parts=False)
                            concat_gt_polylines = concat_gt_polylines.geometry
                            concat_gt_polylines = concat_gt_polylines.apply(lambda g: extract_endpoints(g))
                            concat_gt_polylines = gp.GeoDataFrame({'geometry': concat_gt_polylines}, geometry='geometry', crs='EPSG:28992').reset_index(drop=True)
                            concat_gt_polylines['id'] = concat_gt_polylines.index + 1
                            concat_gt_polylines.to_file(os.path.join(self.output_dir_poly, f'gt_concat_polylines_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")

                        if self.cfg.MODEL.TYPE_POLY_PRED == 'snip':
                            gt_segmented_geoms = segment_geometries(concat_gt_polylines, max_seg_length=self.cfg.MODEL.SNIP_PREDS)
                            gt_segmented_geoms = gp.GeoDataFrame({'geometry': gt_segmented_geoms})
                            gt_segmented_geoms["id"] = gt_segmented_geoms.index + 1
                            gt_segmented_geoms.to_file(os.path.join(self.output_dir_poly, f'gt_segmented_geoms_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")
                            pred_segmented_geoms = segment_geometries(concat_pred_polylines, max_seg_length=self.cfg.MODEL.SNIP_PREDS)
                            pred_segmented_geoms = gp.GeoDataFrame({'geometry': pred_segmented_geoms})
                            pred_segmented_geoms["id"] = pred_segmented_geoms.index + 1
                            pred_segmented_geoms.to_file(os.path.join(self.output_dir_poly, f'pred_segmented_geoms_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")
                        
                        # calculate vector buffer metrics    
                        def calculate_vector_metrics(rd_gt_polylines, rd_pred_polylines, segment_length=1, max_distance=500):
                            # Create buffers and dissolve to get GT and prediction areas.
                            rd_gt_buffer = rd_gt_polylines.copy()
                            rd_pred_buffer = rd_pred_polylines.copy()

                            rd_gt_buffer['geometry'] = rd_gt_buffer.buffer(self.cfg.DATASETS.BUFFER)
                            rd_gt_buffer = rd_gt_buffer.dissolve()
                            rd_gt_buffer['gt_area'] = rd_gt_buffer.geometry.area

                            rd_pred_buffer['geometry'] = rd_pred_buffer.buffer(self.cfg.DATASETS.BUFFER)
                            rd_pred_buffer = rd_pred_buffer.dissolve()
                            rd_pred_buffer['pred_area'] = rd_pred_buffer.geometry.area

                            # Clip and dissolve to compute overlapping (clipped) areas.
                            clipped_gt = rd_gt_buffer.clip(rd_pred_buffer)
                            clipped_gt = clipped_gt.dissolve()
                            clipped_gt['clip_gt_area'] = clipped_gt.geometry.area

                            clipped_preds = rd_pred_buffer.clip(rd_gt_buffer)
                            clipped_preds = clipped_preds.dissolve()
                            clipped_preds['clip_pred_area'] = clipped_preds.geometry.area

                            vector_recall = clipped_gt['clip_gt_area'].sum() / rd_gt_buffer['gt_area'].sum()
                            vector_precision = clipped_preds['clip_pred_area'].sum() / rd_pred_buffer['pred_area'].sum()
                            vector_f1 = 2 * (vector_precision * vector_recall) / (vector_precision + vector_recall)

                            for idx, geom in rd_gt_polylines['geometry'].items():
                                coords = list(geom.coords) if hasattr(geom, 'coords') else []
                                if len(coords) < 2:
                                    print(f"GT geometry at index {idx} has less than 2 coords: {coords}")
                            
                            for idx, geom in rd_pred_polylines['geometry'].items():
                                coords = list(geom.coords) if hasattr(geom, 'coords') else []
                                if len(coords) < 2:
                                    print(f"Pred geometry at index {idx} has less than 2 coords: {coords}")
                                    
                            rd_gt_polylines['geometry'] = rd_gt_polylines['geometry'].apply(
                                lambda geom: geom.segmentize(segment_length) if geom.geom_type == 'LineString' and len(set(geom.coords)) > 1 else geom
                            )
                            rd_pred_polylines['geometry'] = rd_pred_polylines['geometry'].apply(
                                lambda geom: geom.segmentize(segment_length) if geom.geom_type == 'LineString' and len(set(geom.coords)) > 1 else geom
                            )

                            gt_area = rd_gt_buffer['gt_area'].sum()
                            pred_area = rd_pred_buffer['pred_area'].sum()

                            gt_coords = rd_gt_polylines.get_coordinates()
                            gt_points = gp.GeoDataFrame(
                                {'geometry': gp.points_from_xy(gt_coords['x'], gt_coords['y'])},
                                crs='EPSG:28992'
                            )
                            gt_points["id"] = np.arange(1, len(gt_points) + 1)

                            pred_coords = rd_pred_polylines.get_coordinates()
                            pred_points = gp.GeoDataFrame(
                                {'geometry': gp.points_from_xy(pred_coords['x'], pred_coords['y'])},
                                crs='EPSG:28992'
                            )
                            pred_points["id"] = np.arange(1, len(pred_points) + 1)

                            gt_nn = gt_points.sjoin_nearest(
                                pred_points[['geometry', 'id']],
                                how='left',
                                distance_col="distances",
                                max_distance=max_distance
                            )
                            gt_nn = gt_nn.merge(
                                pred_points[['id', 'geometry']],
                                left_on='id_right',
                                right_on='id',
                                suffixes=('_gt', '_pred')
                            )
                            
                            gt_lines = [
                                LineString([pt_gt, pt_pred])
                                for pt_gt, pt_pred in zip(gt_nn['geometry_gt'], gt_nn['geometry_pred'])
                            ]
                            
                            gt_nn = gt_nn.assign(geometry=gt_lines).set_geometry('geometry').drop(columns=['geometry_gt', 'geometry_pred'], errors='ignore')
                            gt_nn['length'] = gt_nn.geometry.length
                            gt_nn.to_file(os.path.join(self.output_dir_poly, f'gt_nn_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG")                             
                            
                            norm_gt_discrepancy = gt_nn.geometry.buffer(self.cfg.DATASETS.BUFFER).unary_union.area - min(gt_area, pred_area)

                            pred_nn = pred_points.sjoin_nearest(
                                gt_points[['geometry', 'id']],
                                how='left',
                                distance_col="distances",
                                max_distance=max_distance
                            )
                            pred_nn = pred_nn.merge(
                                gt_points[['id', 'geometry']],
                                left_on='id_right',
                                right_on='id',
                                suffixes=('_pred', '_gt')
                            )
                                    
                            pred_lines = [
                                LineString([pt_pred, pt_gt])
                                for pt_pred, pt_gt in zip(pred_nn['geometry_pred'], pred_nn['geometry_gt'])
                            ]
                            
                            pred_nn = pred_nn.assign(geometry=pred_lines).set_geometry('geometry').drop(columns=['geometry_gt', 'geometry_pred'], errors='ignore')
                            pred_nn['length'] = pred_nn.geometry.length
                            pred_nn.to_file(os.path.join(self.output_dir_poly, f'pred_nn_{tile}_{type_}_{vis_type}.gpkg'), layer='boundaries', driver="GPKG") 

                            norm_pred_discrepancy = pred_nn.geometry.buffer(self.cfg.DATASETS.BUFFER).unary_union.area - min(gt_area, pred_area)
                            
                            return (
                                vector_precision,
                                vector_recall,
                                vector_f1,
                                norm_gt_discrepancy,
                                norm_pred_discrepancy
                            )
                            
                        if self.cfg.MODEL.TYPE_POLY_PRED == 'snip':
                            vector_precision, vector_recall, vector_f1, norm_gt_discrepancy, norm_pred_discrepancy = calculate_vector_metrics(gt_segmented_geoms, pred_segmented_geoms)
                        elif self.cfg.MODEL.TYPE_POLY_PRED == 'concat':                        
                            vector_precision, vector_recall, vector_f1, norm_gt_discrepancy, norm_pred_discrepancy = calculate_vector_metrics(concat_gt_polylines, concat_pred_polylines)
                        else:
                            vector_precision, vector_recall, vector_f1, norm_gt_discrepancy, norm_pred_discrepancy = calculate_vector_metrics(gt_polylines_visibility, rd_pred_polylines)

                        
                        vector_writer.writerow([type_, vis_type, vector_precision, vector_recall, vector_f1, norm_gt_discrepancy, norm_pred_discrepancy])

            vector_csv_file.close()
            end_time = time.time()
            print(f"Processing time for tile {tile}: {end_time - start_time:.2f} seconds")     
        

