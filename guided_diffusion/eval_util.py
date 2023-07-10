import numpy as np
import os
import cv2
import json
from PIL import Image

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def mean_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def dice_score(pred, targs):
    pred = (pred>0).astype(float)
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def get_results(res_dir, gt_dir):

    # the folder: CVC-300, Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB
    #res_dir = "./default_experiment1/ema60k_fast100/ETIS-LaribPolypDB/imagesPred/"
    #gt_dir = "../Data/Kvasir/TestDataset/ETIS-LaribPolypDB/masks"

    files = [f[:-4] for f in os.listdir(gt_dir)]

    summary = dict()
    
    miou_sum = 0
    dice_sum = 0

    for f in files:

        gt = cv2.imread(os.path.join(gt_dir, f + '.png'), cv2.IMREAD_GRAYSCALE)
        gt = np.where(gt >= 127, 1, 0)

        res = cv2.imread(os.path.join(res_dir, f, 'ensem_mask.png'), cv2.IMREAD_GRAYSCALE)
        res = np.where(res >= 127, 1, 0)

        assert gt.shape == res.shape, "The shape of annotation and segmentation mask should match"

        miou = mean_iou(gt, res)
        dice = dice_score(res, gt)

        eval = dict(mIOU = miou,
            Dice=dice)
   
        summary.update({f: eval})

        miou_sum += miou
        dice_sum += dice

    avg_miou = miou_sum / len(summary)
    avg_dice = dice_sum / len(summary)

    eval = dict(Average_mIOU = avg_miou,
            Average_Dice=avg_dice)
    summary.update({"summary": eval})
    
    # convert dictionary to JSON format, serialize the dictionary summary into a JSON formatted string. 
    json_data = json.dumps(summary, indent=4)
    print(os.path.join('/'.join(res_dir.split('/')[:-2]), 'summary.json'))
    
    return json_data
