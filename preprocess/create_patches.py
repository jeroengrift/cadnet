import rasterio
import os
import math
from rasterio.windows import Window
from rasterio.enums import Resampling
from contextlib import contextmanager  
from rasterio.io import MemoryFile

# test tiles must have overlap of at least 50% to create the right predictions
# train and validate tiles do not need this

# https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array
@contextmanager
def resample_raster(image, downscale=0.5):
    profile = image.profile.copy()
    img = image.read(
        out_shape=(
            image.count,
            int(image.height * downscale),
            int(image.width * downscale)
        ),
        resampling=Resampling.nearest)

    transform = image.transform * image.transform.scale(
            (1 / downscale),
            (1 / downscale))

    profile.update({
        "height": img.shape[-2],
        "width": img.shape[-1],
        "transform": transform})
    
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(img)
            del img

        with memfile.open() as dataset:
            yield dataset

class CreatePatches():
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'train', cfg.DATASETS.IMAGES_DIR)
        self.validate_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'validate', cfg.DATASETS.IMAGES_DIR)
        self.test_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.IMAGES_DIR)
        self.root_dirs = [self.train_dir, self.validate_dir, self.test_dir]
        self.root_dirs = [self.test_dir]

        self.image_size = cfg.DATASETS.IMG_SIZE

    @staticmethod
    def write_patch(img_, window, output_dir, filename, i_, n_):
        transform = img_.window_transform(window)
        profile = img_.profile
        profile.update({
            "height": window.height,
            "width": window.width,
            "transform": transform })

        print(f'image_{i_}_{n_}_{filename}')
        output_path = os.path.join(output_dir, f'image_{i_}_{n_}_{filename}')
        if os.path.exists(output_path):
            print('patch already exists')
            return
        else:
            with rasterio.open(os.path.join(output_dir, f'image_{i_}_{n_}_{filename}'), 'w',
                            **profile) as dataset:
                dataset.write(img_.read(window=window))       

    def create_patch(self):
        for root_folder in self.root_dirs:
            split_path = os.path.split(root_folder)[0]
            root_output_folder = os.path.join(split_path, self.cfg.DATASETS.PATCHES_DIR, self.cfg.DATASETS.IMAGES_DIR)
            os.makedirs(root_output_folder, exist_ok=True)
            image_list = os.listdir(root_folder)

            for image_name in image_list:
                image_path = os.path.join(root_folder, image_name)
                with rasterio.open(image_path, 'r', driver='GTiff', dtype='uint16') as img:
                    # patches_x = math.ceil(img.width / self.image_size)
                    # patches_y = math.ceil(img.height / self.image_size)
                    # patch_step_x_modulo = (self.image_size * patches_x) % img.width
                    # patch_step_y_modulo = (self.image_size * patches_y) % img.height
                    # stride_x = self.image_size - int(patch_step_x_modulo / (patches_x - 1))
                    # stride_y = self.image_size - int(patch_step_y_modulo / (patches_y - 1))
                    
                    if 'test' in root_folder:
                        div_factor = 2
                        patches_x = math.floor(img.width / (self.image_size / div_factor))
                        patches_y = math.floor(img.height / (self.image_size / div_factor))
                        stride_x = int((self.image_size / div_factor))
                        stride_y = int((self.image_size / div_factor))

                    if 'train' in root_folder or 'validate' in root_folder:
                        patches_x = math.ceil(img.width / self.image_size)
                        patches_y = math.ceil(img.height / self.image_size)
                        patch_step_x_modulo = (self.image_size * patches_x) % img.width
                        patch_step_y_modulo = (self.image_size * patches_y) % img.height
                        stride_x = self.image_size - int(patch_step_x_modulo / (patches_x - 1))
                        stride_y = self.image_size - int(patch_step_y_modulo / (patches_y - 1))

                    for i in range(patches_x):
                        for n in range(patches_y):
                            if i == 0 and n == 0:
                                w = Window(0, 0, self.image_size, self.image_size)
                                self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                            elif not i == range(patches_x)[-1] and not n == range(patches_y)[-1]:
                                w = Window(i * stride_x, n * stride_y, self.image_size, self.image_size)
                                self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                            elif i == range(patches_x)[-1] and not n == range(patches_y)[-1]:
                                w = Window(img.width - self.image_size, n * stride_y, self.image_size,
                                        self.image_size)
                                self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                            elif not i == range(patches_x)[-1] and n == range(patches_y)[-1]:
                                w = Window(i * stride_x, img.height - self.image_size, self.image_size,
                                        self.image_size)
                                self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                            elif i == range(patches_x)[-1] and n == range(patches_y)[-1]:
                                w = Window(img.width - self.image_size, img.height - self.image_size,
                                        self.image_size, self.image_size)
                                self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)
                                
    def create_patch_resample(self):
        for root_folder in self.root_dirs:
            split_path = os.path.split(root_folder)[0]
            root_output_folder = os.path.join(split_path, self.cfg.DATASETS.PATCHES_DIR, self.cfg.DATASETS.IMAGES_DIR)
            os.makedirs(root_output_folder, exist_ok=True)
            image_list = os.listdir(root_folder)
            
            for image_name in image_list:
                image_path = os.path.join(root_folder, image_name)
                
                with rasterio.open(image_path, 'r', driver='GTiff', dtype='uint16') as image:
                    with resample_raster(image, self.cfg.DATASETS.DOWNSAMPLE_FACTOR) as img: 
                        # patches_x = math.ceil(img.width / self.image_size)
                        # patches_y = math.ceil(img.height / self.image_size)
                        # patch_step_x_modulo = (self.image_size * patches_x) % img.width
                        # patch_step_y_modulo = (self.image_size * patches_y) % img.height
                        # stride_x = self.image_size - int(patch_step_x_modulo / (patches_x - 1))
                        # stride_y = self.image_size - int(patch_step_y_modulo / (patches_y - 1)) 
                        if 'test' in root_folder:
                            div_factor = 2
                            patches_x = math.floor(img.width / (self.image_size / div_factor))
                            patches_y = math.floor(img.height / (self.image_size / div_factor))
                            stride_x = int((self.image_size / div_factor))
                            stride_y = int((self.image_size / div_factor))
                        else:
                            patches_x = math.ceil(img.width / self.image_size)
                            patches_y = math.ceil(img.height / self.image_size)
                            patch_step_x_modulo = (self.image_size * patches_x) % img.width
                            patch_step_y_modulo = (self.image_size * patches_y) % img.height
                            stride_x = self.image_size - int(patch_step_x_modulo / (patches_x - 1))
                            stride_y = self.image_size - int(patch_step_y_modulo / (patches_y - 1))
                
                        for i in range(patches_x):
                            for n in range(patches_y):
                                if i == 0 and n == 0:
                                    w = Window(0, 0, self.image_size, self.image_size)
                                    self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                                elif not i == range(patches_x)[-1] and not n == range(patches_y)[-1]:
                                    w = Window(i * stride_x, n * stride_y, self.image_size, self.image_size)
                                    self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                                elif i == range(patches_x)[-1] and not n == range(patches_y)[-1]:
                                    w = Window(img.width - self.image_size, n * stride_y, self.image_size,
                                            self.image_size)
                                    self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                                elif not i == range(patches_x)[-1] and n == range(patches_y)[-1]:
                                    w = Window(i * stride_x, img.height - self.image_size, self.image_size,
                                            self.image_size)
                                    self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)

                                elif i == range(patches_x)[-1] and n == range(patches_y)[-1]:
                                    w = Window(img.width - self.image_size, img.height - self.image_size,
                                            self.image_size, self.image_size)
                                    self.write_patch(img, w, root_output_folder, image_name, w.col_off, w.row_off)