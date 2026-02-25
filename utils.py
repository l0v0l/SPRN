import torch
import os
from PIL import Image
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_path, sp_path, plaque_path, items, transform=None, size=(256, 256)):
        super().__init__()
        assert len(os.listdir(images_path)) == len(os.listdir(sp_path))
        self.items=items
        self.img_path=images_path
        self.sp_path=sp_path
        self.plaque_path=plaque_path
        self.size=size
        self.transform=None
        if transform:
            self.transform=transform
    def __getitem__(self, index):

        image=Image.open(os.path.join(self.img_path, self.items[index]))
        image_slice=np.array(image).astype(np.float32)

        image_slice[:,:,0]=image_slice[:,:,0]*image_slice[:,:,1]/255

        sp=Image.open(os.path.join(self.sp_path, self.items[index]))
        sp_slice=np.array(sp).astype(np.float32)

        plaque=Image.open(os.path.join(self.plaque_path, self.items[index]))
        plaque_slice=np.array(plaque).astype(np.float32)

        plaque_slice[plaque_slice<128]=0
        plaque_slice[plaque_slice>=128]=1

        image_out=np.zeros((2, image_slice.shape[0], image_slice.shape[1]))
        image_out[0,:,:]=image_slice[:,:,0]
        image_out[1,:,:]=image_slice[:,:,1]

        image=torch.from_numpy(image_out)
        image=image/255
        plaque=torch.from_numpy(plaque_slice)
        plaque=plaque
        sp=torch.from_numpy(sp_slice)
        if self.transform:
            image, plaque, sp=self.transform(image, plaque, sp)
        sp_vol=self._gen_sp_vol(sp, sp_slice)
        
        return image.float(), torch.unsqueeze(plaque, dim=0).float(), sp_vol.float()

    def __len__(self):
        return len(self.items)
    
    def _gen_sp_vol(self, sp, sp_np):
        sp_value_list=list(set(np.ndarray.tolist(np.ndarray.flatten(sp_np))))
        sp_value_list.sort()

        # clear the plaque area
        sp_value_list.remove(torch.max(sp))
        # clear the background
        sp_value_list.remove(0)
        pop_num=len(sp_value_list)//10

        out_sp_vol=torch.zeros((10,sp.shape[0], sp.shape[1]))

        for i in range(9):
            for _ in range(pop_num):
                sp_value_list.pop()
                sp_canvas=torch.zeros_like(sp)
                for sp_value in sp_value_list:
                    sp_canvas[sp==sp_value]=1
                sp_canvas[sp==torch.max(sp)]=1
                out_sp_vol[i,:,:]=sp_canvas
        sp_canvas=torch.zeros_like(sp)
        sp_canvas[sp==torch.max(sp)]=1
        out_sp_vol[-1,:,:]=sp_canvas
        return out_sp_vol
    
def dice_coef_for_training_pt(y_true, y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss_pt(y_true, y_pred):
    return 1. - dice_coef_for_training_pt(y_true, y_pred)


def gen_reference(pred, sp):
    pred_tmp=pred.cpu()
    pred_tmp[pred_tmp<0.9]=0
    pred_tmp[pred_tmp>0]=1
    sp_tmp=sp.cpu()
    overlap=pred_tmp*sp_tmp
    sp_list=torch.flatten(overlap.cpu()).tolist()
    sp_set=list(set(sp_list))

    pred_num=len(sp_set)
    pred_num=np.array(pred_num).astype(np.float32)
    pred_num=torch.from_numpy(np.expand_dims(pred_num, axis=0))
    return pred_num.to('cuda')

def milestone_loss_fn(pred, sp):
    if torch.sum(pred)>0:
        pred_num=gen_reference(pred, sp)
        return 1-1/pred_num
    else:
        return 1