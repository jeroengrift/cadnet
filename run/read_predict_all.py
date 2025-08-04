import os
import csv
import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix

output_dir = 'cadnet/output/'

def read_raster():
    output_csv_path = os.path.join(output_dir, "predict_raster.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["model", "tile", "type", "vis_type", "precision", "recall", "f1", "select_vis", "overlap_50"])
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    if filename.startswith('raster'):
                        model = dirpath.split('/')[-2].split('-')[2]
                        select_vis = dirpath.split('/')[-2].split('-')[-2].split('_')[0]
                        overlap_50 = dirpath.split('/')[-1].split('_')[-1]
                        try:
                            with open(os.path.join(dirpath, filename), "r") as preds:
                                reader = csv.reader(preds)
                                data = list(reader)[1:]
                                for line in data:
                                    tile = line[0]
                                    type = line[1]
                                    precision = line[2]
                                    recall = line[3]
                                    f1 = line[4]
                                    vis = line[5]
                                    csvwriter.writerow([model, tile, type, vis, precision, recall, f1, select_vis, overlap_50])
                        except: 
                            print('Error reading file')
                            continue

def read_vector(): 
    output_csv_path = os.path.join(output_dir, "predict_vector.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["model", "tile", "type", "vis_type", "vector_precision", "vector_recall", "vector_f1", "norm_gt_discrepancy", "norm_pred_discrepancy", "select_vis", "overlap_50"])
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    if filename.startswith('polis'):
                        model = dirpath.split('/')[-3].split('-')[2]  
                        select_vis = dirpath.split('/')[-3].split('-')[-2].split('_')[0]
                        overlap_50 = dirpath.split('/')[-2].split('_')[-1]
                        tile = filename.split('_')[1].split('.')[0]
                        try:
                            with open(os.path.join(dirpath, filename), "r") as preds:
                                reader = csv.reader(preds)
                                data = list(reader)[1:]         
                                print(data)                      
                                for line in data:                           
                                    type = line[0]
                                    vis = line[1]
                                    vector_precision = line[2]
                                    vector_recall = line[3]
                                    vector_f1 = line[4]
                                    norm_gt_discrepancy = line[5]
                                    norm_pred_discrepancy = line[6]
                                    csvwriter.writerow([model, tile, type, vis, vector_precision, vector_recall, vector_f1, norm_gt_discrepancy, norm_pred_discrepancy, select_vis, overlap_50])
                        except:
                            print('Error reading file')
                            continue    
                      
def evaluate_raster():
    output_csv_path = os.path.join(output_dir, "evaluate_raster.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["dirpath", "tile", "TN", "FP", "FN", "TP", "precision", "recall", "f1"])
        
        for dirpath, dirnames, filenames in os.walk(output_dir):
            if 'model' in dirpath and 'preds' in dirpath and 'concat' not in dirpath:
                total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
                for filename in filenames:
                    if filename in ['tile-120000-430000_brk.tif', 'tile-200000-540000_brk.tif']:
                        print(f"Processing: {dirpath}")
                        print(f"Processing: {filename}")
                        with rasterio.open(os.path.join(dirpath, filename)) as src:
                            pred = src.read().flatten()

                        with rasterio.open(os.path.join(dirpath, filename.replace('.tif', '_gt_.tif'))) as src:
                            gt = src.read().flatten()
                            
                        pred = np.where(np.isnan(pred), 0, pred)
                        pred[pred >= 0.5] = 1.0
                        pred[pred < 0.5] = 0.0
                        pred = pred.astype(np.uint8).flatten()

                        gt = np.where(np.isnan(gt), 0, gt)
                        gt[gt >= 0.5] = 1.0
                        gt[gt < 0.5] = 0.0
                        gt = gt.astype(np.uint8).flatten()

                        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0., 1.]).ravel()
                        total_tn += tn
                        total_fp += fp
                        total_fn += fn
                        total_tp += tp    

                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                csvwriter.writerow([dirpath, total_tn, total_fp, total_fn, total_tp,
                                    f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
                print(f"Saved metrics for: {dirpath}")
                        
    
if __name__ == '__main__':
    read_raster()    
    read_vector()
    evaluate_raster()