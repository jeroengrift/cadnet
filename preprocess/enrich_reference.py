import os
import geopandas as gp  
import rasterio as rio  
import numpy as np
from shapely.geometry import box, LineString

import sys
sys.path.insert(1, '/home/griftj/phd/')

# ----------------------------------------------------------------------------------------------
# Configuration parameters
type_reference = 'brk'
pixel_size_patch = 25
input_dir = 'cad_aerial_8_sub_test'
# ----------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    patches_dir = f'patches_{pixel_size_patch}'
    in_dir = f'data/{input_dir}'
    urban_file = 'data/cad_reference/brt_bebouwde_kom.gpkg'
    out_dir = os.path.join(in_dir, 'reference')
    
    if type_reference == 'topo':
        reference_file = 'data/cad_reference/topo_reference.gpkg'
    elif type_reference == 'brk':
        reference_file = 'data/cad_reference/brk_reference.gpkg'

    train_img_root = os.path.join(in_dir, 'train', patches_dir, 'images')
    test_img_root = os.path.join(in_dir, 'test', patches_dir, 'images')
    val_img_root = os.path.join(in_dir, 'validate', patches_dir, 'images')
    img_root = {'train': train_img_root, 'test' : test_img_root, 'validate' : val_img_root}
    
    urban = gp.read_file(urban_file, layer='brk_bebouwde_kom', engine="pyogrio")

    reference_dict = {
            'filename_10k': [],
            'image_id': [],
            'visible': [],
            'urban': [],
            'split': [],
            'geometry': []
        }
    
    for split, split_root in img_root.items():
        images = sorted(os.listdir(split_root))
        print(split)
        print(len(images))
        from tqdm import tqdm
        for image in tqdm(images):
            img_bounds = rio.open(os.path.join(split_root, image), crs='epsg:28992')
            clip_mask = box(*img_bounds.bounds)
        
            try:
                if type_reference == 'topo':
                    reference = gp.read_file(reference_file, layer='topo_reference', engine="pyogrio", bbox=clip_mask)
                    reference = reference[reference['topo_source'] == 'BGT']
                    reference = reference.drop_duplicates(subset=['filename_10k', 'topo_id'])
                else:
                    reference = gp.read_file(reference_file, layer=f'brk_reference', engine="pyogrio", bbox=clip_mask)
            except:
                continue
                                
            if len(reference) == 0:   
                continue
                
            reference['image_id'] = image
            
            img_references = reference.clip(mask=clip_mask) # https://geopandas.org/en/stable/docs/reference/api/geopandas.clip.html                
            img_references = img_references.sjoin(urban, how='left', predicate='intersects')   
                        
            img_references['urban'] = np.where(img_references.BBK_ID.isnull(), False, True)
            
            if type_reference == 'topo':
                img_references['visible'] = True
            else:
                img_references['visible'] = np.where(img_references.visible == 'True', True, False)
                
            img_references = img_references.reindex(columns=['filename_10k', 'image_id', 'visible', 'urban', 'geometry'])
                        
            if len(img_references) > 0:   
                for i, row in enumerate(img_references.itertuples()):
                    if isinstance(row.geometry, LineString):
                        reference_dict['filename_10k'].append(row.filename_10k)
                        reference_dict['image_id'].append(row.image_id)
                        reference_dict['visible'].append(row.visible)
                        reference_dict['urban'].append(row.urban)
                        reference_dict['geometry'].append(row.geometry)
                        reference_dict['split'].append(split)
                        
    references_df = gp.GeoDataFrame.from_dict(reference_dict).set_crs(28992)
    references_df.to_file(os.path.join(out_dir, f'reference_{input_dir}_ps_{pixel_size_patch}_{type_reference}.gpkg'), driver='GPKG')
    
    
    