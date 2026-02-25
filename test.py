import torch
import os
import numpy as np
import models
import imageio
import argparse
import utils


def test_main(args):
    print('Starting the testing loop...')

    # load data
    test_img_path=os.path.join(args.in_folder, 'test_slices')
    test_sp_path=os.path.join(args.in_folder, 'test_sp_masks')
    test_plaque_path=os.path.join(args.in_folder, 'test_plaque_masks')
    test_items=os.listdir(test_img_path)

    num_test_imgs=len(test_items)
    print('training dataset has '+str(num_test_imgs)+' images in training')

    test_data = utils.Dataset(test_img_path, test_sp_path, test_plaque_path, test_items)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # load mdoel 
    model=models.unet_bn().to(args.device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    out_folder=args.test_out
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    img_num=0
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Stage 2 training')
    parser.add_argument('--lr', help='learning rate', default=1e-3)
    parser.add_argument('--in_folder', help='dataset folder', default='POM_train_SPIE_sp')
    parser.add_argument('--model', help='model path', default='tuned_model.pth')
    parser.add_argument('--device', help='device', default=torch.device('cuda:0'))
    parser.add_argument('--test_out', help='Test output folder', default='output')
    args=parser.parse_args()

    test_main(args)