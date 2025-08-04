import os
import geopandas as gp  
import pandas as pd
import rasterio as rio  
import numpy as np
from shapely.geometry import box

# ----------------------------------------------------------------------------------------------
# Configuration parameters
input_dir = 'cad_aerial_8_sub_test'
pixel_size_patch = 25
in_dir = f'data/{input_dir}'
urban_file = 'data/cad_reference/brt_bebouwde_kom.gpkg'
# ----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    patches_dir = f'patches_{pixel_size_patch}'
    out_dir = os.path.join(in_dir, 'reference')
    
    train_img_root = os.path.join(in_dir, 'train', patches_dir, 'images')
    test_img_root = os.path.join(in_dir, 'test', patches_dir, 'images')
    val_img_root = os.path.join(in_dir, 'validate', patches_dir, 'images')
    img_root = {'train': train_img_root, 'test' : test_img_root, 'validate' : val_img_root}
    
    urban = gp.read_file(urban_file, layer='brk_bebouwde_kom', engine="pyogrio")
    
    gdf_patch = gp.GeoDataFrame()
    
    for split, split_root in img_root.items():
        images = sorted(os.listdir(split_root))
        print(split)
        print(len(images))
        from tqdm import tqdm
        for image in tqdm(images):
            img_bounds = rio.open(os.path.join(split_root, image), crs='epsg:28992')
            clip_mask = box(*img_bounds.bounds)
            image_geom = gp.GeoDataFrame(geometry=[clip_mask], crs=28992)
            image_geom = image_geom.sjoin(urban, how='left', predicate='intersects')
            image_geom['urban'] = np.where(image_geom.BBK_ID.isnull(), False, True)  
            image_geom['image_id'] = image 
            image_geom['split'] = split 
            image_geom = image_geom[['image_id', 'urban', 'split', 'geometry']]
            gdf_patch = pd.concat([gdf_patch, image_geom])
                                            
    gdf_patch.to_file(os.path.join(out_dir, f'urban_{pixel_size_patch}.gpkg'), driver='GPKG')
    
    
    