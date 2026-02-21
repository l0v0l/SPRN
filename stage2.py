import torch
import os
import numpy as np
import utils

import random

import models


from torchsummary import summary
import argparse
from tqdm import tqdm
import imageio

def train(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_img_path = os.path.join(args.in_folder, 'train_slices')
    train_sp_path = os.path.join(args.in_folder, 'train_sp_masks')
    train_plaque_path=os.path.join(args.in_folder, 'train_plaque_masks')

    train_items=os.listdir(train_img_path)

    val_img_path = os.path.join(args.in_folder, 'val_slices')
    val_sp_path = os.path.join(args.in_folder, 'val_sp_masks')
    val_plaque_path=os.path.join(args.in_folder, 'val_plaque_masks')

    val_items=os.listdir(val_img_path)

    test_img_path=os.path.join(args.in_folder, 'test_slices')
    test_sp_path=os.path.join(args.in_folder, 'test_sp_masks')
    test_plaque_path=os.path.join(args.in_folder, 'test_plaque_masks')
    test_items=os.listdir(test_img_path)
    
    num_train_imgs = len(train_items)
    num_val_imgs=len(val_items)
    num_test_imgs=len(test_items)
    print('training dataset has '+str(num_train_imgs)+' images in training')
    print('training dataset has '+str(num_val_imgs)+' images in training')
    print('training dataset has '+str(num_test_imgs)+' images in training')

    # instantiate dataset
    train_data = utils.Dataset(train_img_path, train_sp_path, train_plaque_path, train_items)
    val_data = utils.Dataset(val_img_path, val_sp_path, val_plaque_path, val_items)
    test_data = utils.Dataset(test_img_path, test_sp_path, test_plaque_path, test_items)

    # instantiate dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    model=models.unet_bn().to(args.device)
    model_save_name='tuned_model.pth'

    model_load_name='base_model.pth'
    model.load_state_dict(torch.load(model_load_name, weights_only=True))

    
    summary(model,(2,256,320))

    # define loss function
    seg_criterion = utils.dice_coef_loss_pt
    mile_criterion=utils.milestone_loss_fn
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    train_iter = 0
    train_epoch = 0
    min_val_loss=100

    print("Starting the training loop...")
    for _ in tqdm(range(args.epoch)):
        for _, (image, mask, sp) in enumerate(train_loader):  
            # Iter predict train
            for i in range(10):
                image_cuda = image.to(args.device)
                mask_cuda = mask.to(args.device)
                sp_cuda=sp.to(args.device)
                optimizer.zero_grad()
                output= model(image_cuda)

                # 3 losses: loss to target, milestone and the stop flag loss
                loss = seg_criterion(mask_cuda, output)+ mile_criterion(output, sp_cuda)  # mask: torch.Size([30, 1, 200, 200]) ; output: torch.Size([30, 1, 200, 200])
                loss.backward()
                optimizer.step()

                
                with torch.no_grad():
                    model.eval()
                    output_tmp= model(image_cuda)
                    image[:,1,:,:]=output_tmp[:,0,:,:]
                    model.train()

            # train log
            if train_iter % args.val_loss_every_iters == 0: 
                val_loss=0
                test_loss=0
                model.eval()
                with torch.no_grad():
                    for _, (val_image, val_mask, val_sp) in enumerate(val_loader):
                        val_image=val_image.to(args.device)
                        val_mask=val_mask.to(args.device)
                        val_sp=val_sp.to(args.device)
                        val_output=models.iter_predict(model, val_image, iter=10, device=args.device)
                        loss = seg_criterion(val_mask, val_output)+ mile_criterion(val_output, val_sp)
                        val_loss=val_loss+loss.item()*val_image.size(0)/num_val_imgs
                    for _, (test_image, test_mask, test_sp) in enumerate(test_loader):
                        test_image=test_image.to(args.device)
                        test_mask=test_mask.to(args.device)
                        test_sp=test_sp.to(args.device)
                        test_output=models.iter_predict(model, test_image, iter=10, device=args.device)
                        loss = seg_criterion(test_mask, test_output)+ mile_criterion(test_output, test_sp)
                        test_loss=test_loss+loss.item()*test_image.size(0)/num_test_imgs
                    if args.iter_save:
                        if min_val_loss>val_loss:
                            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                            print(f'Now the test Loss is--->{test_loss:.6f} \t')
                            min_val_loss=val_loss
                            # Saving State Dict
                            torch.save(model.state_dict(), model_save_name)
                model.train()
            train_iter+=1
        
        val_loss=0
        test_loss=0
        model.eval()
        with torch.no_grad():
            for _, (val_image, val_mask, val_sp) in enumerate(val_loader):
                val_image=val_image.to(args.device)
                val_mask=val_mask.to(args.device)
                val_sp=val_sp.to(args.device)
                val_output=models.iter_predict(model, val_image, iter=10, device=args.device)
                loss = seg_criterion(val_mask, val_output)+ mile_criterion(val_output, val_sp)
                val_loss=val_loss+loss.item()*val_image.size(0)/num_val_imgs
            for _, (test_image, test_mask, test_sp) in enumerate(test_loader):
                test_image=test_image.to(args.device)
                test_mask=test_mask.to(args.device)
                test_sp=test_sp.to(args.device)
                test_output=models.iter_predict(model, test_image, iter=10, device=args.device)
                loss = seg_criterion(test_mask, test_output)+ mile_criterion(test_output, test_sp)
                test_loss=test_loss+loss.item()*test_image.size(0)/num_test_imgs
            if args.epoch_save:
                if min_val_loss>val_loss:
                    print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model obtained at Epoch {train_epoch}.')
                    print(f'Now the test Loss is--->{test_loss:.6f} \t')
                    min_val_loss=val_loss
                    # Saving State Dict
                    torch.save(model.state_dict(), model_save_name)
        model.train()
        train_epoch+=1

    print('Starting the testing loop...')
    # load mdoel 
    model=models.unet_bn().to(args.device)
    model.load_state_dict(torch.load(model_save_name, weights_only=True))
    model.eval()

    out_folder=args.test_out
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    img_num=0
    #test_criterion=utils.dice_coef_final_pt
    test_loss=0
    for test_batch_idx, (test_image, test_mask, test_sp) in enumerate(test_loader):
        test_image=test_image.to(args.device)
        test_mask=test_mask.to(args.device)
        test_sp=test_sp.to(args.device)
        test_output=models.iter_predict(model, test_image, iter=10, device=args.device)
        img_num+=1

        out_pred=test_output.cpu().detach().numpy()
        out_img=test_image.cpu().detach().numpy()
        out_true=test_mask.cpu().detach().numpy()

        # Save the output
        for i in range(out_pred.shape[0]):
            out_pred_slice=out_pred[i,0,:,:]
            out_img_slice=out_img[i,0,:,:]
            out_true_slice=out_true[i,0,:,:]

            imageio.imwrite(os.path.join(out_folder, str(img_num).zfill(5)+'_img.png'), (out_img_slice*255).astype(np.uint8))
            imageio.imwrite(os.path.join(out_folder, str(img_num).zfill(5)+'_pred.png'), (out_pred_slice*255).astype(np.uint8))
            imageio.imwrite(os.path.join(out_folder, str(img_num).zfill(5)+'_true.png'), (out_true_slice*255).astype(np.uint8))
            img_num+=1


    print(f'Testing DSC--->{test_loss:.6f}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='multi-resnet training')
    parser.add_argument('--lr', help='learning rate', default=1e-3)
    parser.add_argument('--in_folder', help='dataset folder', default='POM_train_SPIE_sp')
    parser.add_argument('--is_cca', help='Is cca?', default=True)
    parser.add_argument('--batch_size', help='batch size', default=2)
    parser.add_argument('--epoch', help='epochs', default=5)
    parser.add_argument('--device', help='device', default=torch.device('cuda:0'))
    parser.add_argument('--seed', help='Random seed', default=random.randint(1,10000))
    parser.add_argument('--train_loss_every_iters', help='Logging loss every x iters', default=1000)
    parser.add_argument('--val_loss_every_iters', help='Validation every x iters', default=1000)
    parser.add_argument('--model_mode', help='True for ffrn and False for unet', default='ffrn')
    parser.add_argument('--test_out', help='Test output folder', default='output')
    parser.add_argument('--iter_save', help='Save weights in iter', default=False)
    parser.add_argument('--epoch_save', help='Save weights at epoch', default=True)
    args=parser.parse_args()

    train(args)


