import os
import numpy as np
import glob
import imageio.v3 as imageio
from metrics_hd95 import hd95
import pandas as pd

def dice_coeff(pred, true):
    smooth=0.001
    pred=np.ndarray.flatten(pred)
    true=np.ndarray.flatten(true)
    intersection = np.sum(true * pred)
    return (2. * intersection + smooth) / (np.sum(true) + np.sum(pred) + smooth)

def tpv(seg):
    return np.sum(seg)

def tpv_diff(pred, true, voxel_size):
    return abs(tpv(pred)-tpv(true))*voxel_size

if __name__=='__main__':
    in_folder='output'
    out_csv=in_folder+'.csv'
    items=os.listdir(in_folder)
    name_list=[]
    for item in items:
        name_list.append(item.split('--')[1])
    name_list=list(set(name_list))
    name_list.sort()
    dice_list=[]
    haus_list=[]
    tpv_list=[]

    tp_num=0
    fp_num=0
    fn_num=0
    pred_num=0
    true_num=0

    for name in name_list:
        pred_name='pred--'+name
        pred_items=glob.glob(os.path.join(in_folder, pred_name+'*'))
        pred_vol=[]
        true_vol=[]
        for pred_item in pred_items:
            true_item=pred_item.replace('pred--', 'true--')
            pred_slice=imageio.imread(pred_item).astype(np.float32)/255
            true_slice=imageio.imread(true_item).astype(np.float32)/255
            if not np.sum(true_slice)==0:
                pred_vol.append(pred_slice)
                true_vol.append(true_slice)
        pred_vol=np.array(pred_vol)
        true_vol=np.array(true_vol)
        dice_score=dice_coeff(pred_vol, true_vol)
        haus_score=hd95(pred_vol, true_vol)
        tpv_score=tpv_diff(pred_vol, true_vol, 15/1000)

        dice_list.append(dice_score)
        haus_list.append(haus_score)
        tpv_list.append(tpv_score)

    print('===============')
    print('DICE:')
    print(np.mean(np.array(dice_list)))
    print(np.std(np.array(dice_list)))
    print('===============')
    print('HAUS:')
    print(np.mean(np.array(haus_list)))
    print(np.std(np.array(haus_list)))
    print('===============')
    print('TPV:')
    print(np.mean(np.array(tpv_list)))
    print(np.std(np.array(tpv_list)))
    print('===============')



    df=pd.DataFrame({'Dice':dice_list, 'Haus':haus_list, 'TPV':tpv_list})
    df.to_csv(out_csv)