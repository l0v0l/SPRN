import numpy as np
from skimage.segmentation import slic
import os
import imageio.v3 as imageio
import copy
from tqdm import tqdm
import copy
import argparse

def img_norm(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))*255

def delete_square(labels):
    labels=copy.deepcopy(labels)
    label_set=list(set(np.ndarray.tolist(np.ndarray.flatten(labels))))
    for i in range(len(label_set)):
        canvas1=np.zeros_like(labels)
        canvas1[labels==label_set[i]]=1
        w_list, h_list=np.where(canvas1!=0)
        max_w=np.max(np.array(w_list))
        min_w=np.min(np.array(w_list))
        max_h=np.max(np.array(h_list))
        min_h=np.min(np.array(h_list))
        canvas2=np.zeros_like(labels)
        canvas2[min_w, min_h]=1
        canvas2[min_w, max_h]=1
        canvas2[max_w, min_h]=1
        canvas2[max_w, max_h]=1
        if np.sum(np.multiply(canvas1, canvas2))==4:
            labels[labels==label_set[i]]=0

    labels_list=np.ndarray.tolist(np.ndarray.flatten(labels))
    labels_set=list(set(labels_list))
    labels_out=copy.deepcopy(labels)
    for i in range(len(labels_set)):
        if not labels_set[i]==0:
            labels_out[labels==labels_set[i]]=i
    return labels_out

def gen_distance_map(in_sp, in_plaque):
    sp=copy.deepcopy(in_sp)
    sp[in_plaque!=0]=0
    plaque_list_x, plaque_list_y=np.where(in_plaque!=0)
    plaque_np_x=np.array(plaque_list_x)
    plaque_np_y=np.array(plaque_list_y)

    sp_id_list=list(set(np.ndarray.tolist(np.ndarray.flatten(sp))))
    sp_id_list.remove(0)
    sp_canvas=np.zeros_like(sp)
    for sp_id in sp_id_list:
        sp_list_x, sp_list_y=np.where(sp==sp_id)
        centroid_x=int(np.median(np.array(sp_list_x)))
        centroid_y=int(np.median(np.array(sp_list_y)))
        square_distance=np.multiply(plaque_np_x-centroid_x, plaque_np_x-centroid_x)+np.multiply(plaque_np_y-centroid_y, plaque_np_y-centroid_y)
        min_square_distance=np.min(square_distance)
        sp_canvas[sp==sp_id]=min_square_distance
    sp=copy.deepcopy(sp_canvas)
    sp_id_list=list(set(np.ndarray.tolist(np.ndarray.flatten(sp))))
    sp_id_list.sort()
    sp_canvas=np.zeros_like(sp)

    for i in range(1, len(sp_id_list)):
        sp_canvas[sp==sp_id_list[i]]=i
    return sp_canvas

def main_fn(args):
    in_folder=args.input
    out_folder=in_folder+'_sp'
    mode=args.mode

    in_img_folder=os.path.join(in_folder, mode+'_slices')
    in_wall_folder=os.path.join(in_folder, mode+'_wall_masks')
    in_plaque_folder=os.path.join(in_folder, mode+'_plaque_masks')
    items=os.listdir(in_img_folder)

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    out_img_folder=os.path.join(out_folder, mode+'_slices')
    out_sp_folder=os.path.join(out_folder, mode+'_sp_masks')
    out_plaque_folder=os.path.join(out_folder, mode+'_plaque_masks')

    if not os.path.exists(out_img_folder):
        os.mkdir(out_img_folder)
    if not os.path.exists(out_sp_folder):
        os.mkdir(out_sp_folder)
    if not os.path.exists(out_plaque_folder):
        os.mkdir(out_plaque_folder)


    for item in tqdm(items):
        img=imageio.imread(os.path.join(in_img_folder, item)).astype(np.float32)
        wall=imageio.imread(os.path.join(in_wall_folder, item)).astype(np.float32)
        plaque=imageio.imread(os.path.join(in_plaque_folder, item)).astype(np.float32)

        plaque=plaque*wall/255

        w_list, h_list=np.where(wall!=0)
        max_w=np.max(np.array(w_list))
        min_w=np.min(np.array(w_list))
        max_h=np.max(np.array(h_list))
        min_h=np.min(np.array(h_list))

        img_crop=img[min_w:max_w, min_h:max_h]
        plaque_crop=plaque[min_w:max_w, min_h:max_h]
        wall_crop=wall[min_w:max_w, min_h:max_h]
        img_crop=img_crop*wall_crop/255
        plaque_crop=plaque_crop*wall_crop/255

        img_crop=np.concatenate([np.expand_dims(img_crop, axis=-1), np.expand_dims(img_crop, axis=-1), np.expand_dims(img_crop, axis=-1)], axis=-1)
        segments=slic(img_crop, n_segments=250, compactness=10, sigma=1, start_label=1)

        segments=delete_square(segments)

        segments_list=list(set(np.ndarray.tolist(np.ndarray.flatten(segments))))
        segments_list.remove(0)

        segments_out=np.zeros_like(segments)

        for i in range(len(segments_list)):
            segments_out[segments==segments_list[i]]=i

        segments[plaque_crop!=0]=np.max(segments)+1

        guide_mask=np.zeros_like(img)
        guide_mask[min_w:max_w, min_h:max_h]=segments[:,:]
        img_out=np.concatenate([np.expand_dims(img, axis=-1), np.expand_dims(wall, axis=-1), np.expand_dims(np.zeros_like(img), axis=-1)], axis=-1)

        imageio.imwrite(os.path.join(out_img_folder, item), img_out.astype(np.uint8))
        imageio.imwrite(os.path.join(out_sp_folder, item), guide_mask.astype(np.uint8))
        imageio.imwrite(os.path.join(out_plaque_folder, item), plaque.astype(np.uint8))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generating MSM')
    parser.add_argument('--input', help='Input folder', default='POM_train_SPIE')
    parser.add_argument('--mode', help='Partition of the dataset', default='train')
    args=parser.parse_args()

    main_fn(args)    