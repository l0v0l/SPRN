import torch
import torch.nn as nn
from torchsummary import summary

class conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_bn):
        super(conv2d_block, self).__init__()
        self.with_bn = with_bn
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        if self.with_bn:
            self.bn1=nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = self.activation(x)
        return x

class skip_block(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(skip_block, self).__init__()
        self.conv1=conv2d_block(in_channels, in_channels, kernel_size, with_bn=True)
        self.conv2=conv2d_block(in_channels, in_channels, kernel_size, with_bn=True)
        self.conv3=conv2d_block(in_channels, in_channels, kernel_size, with_bn=True)
    def forward(self, in_x):
        x=self.conv1(in_x)
        x=self.conv2(x)
        x=self.conv3(x)
        return x+in_x

class down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_bn=True, with_pool=True):
        super(down, self).__init__()
        self.conv1 = conv2d_block(in_channels, out_channels, kernel_size, with_bn)
        self.conv2=conv2d_block(out_channels, out_channels, kernel_size, with_bn)
        self.with_pool=with_pool
        if self.with_pool:
            self.max_pool=nn.MaxPool2d(2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.with_pool:
            down_x=self.max_pool(x)
            return down_x, x
        else:
            return x
class up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_bn=True):
        super(up, self).__init__()
        self.trans=nn.ConvTranspose2d(in_channels, out_channels, (2,2), stride=(2,2))
        self.conv1 = conv2d_block(out_channels, out_channels, kernel_size, with_bn)
        self.conv2=conv2d_block(out_channels, out_channels, kernel_size, with_bn)
        self.conv_link1=skip_block(out_channels, (1,1))
        self.conv_link2=skip_block(out_channels, (3,3))
        self.conv_link3=skip_block(out_channels, (5,5))

    def forward(self, x, skip_x):
        x = self.trans(x)
        x = x+skip_x
        x = self.conv1(x)
        x = x+skip_x
        x = self.conv2(x)
        x = x+skip_x
        return x

class out_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(out_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.sigmoid=nn.Sigmoid()
        self.bn=nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x=self.bn(x)
        x = self.sigmoid(x)
        return x

class unet_bn(nn.Module):
    def __init__(self):
        super(unet_bn, self).__init__()

        self.down1=down(2, 32, (3,3), with_bn=True, with_pool=True)
        self.down2=down(32, 64, (3,3), with_bn=True, with_pool=True)
        self.down3=down(64, 128, (3,3), with_bn=True, with_pool=True)
        self.down4=down(128, 256, (3,3), with_bn=True, with_pool=True)
        self.down5=down(256, 512, (3,3), with_bn=True, with_pool=False)
        self.up4=up(512, 256, (3,3))
        self.up3=up(256, 128, (3,3))
        self.up2=up(128,64, (3,3))
        self.up1=up(64, 32, (3,3))
        self.out=out_block(32, 1, (3,3))
        
    def forward(self, in_x):
        #Pyramid Haar transform
        x, down1=self.down1(in_x)
        x, down2=self.down2(x)
        x, down3=self.down3(x)
        x, down4=self.down4(x)
        x=self.down5(x)
        x=self.up4(x, down4)
        x=self.up3(x, down3)
        x=self.up2(x, down2)
        x=self.up1(x, down1)

        x=self.out(x)
        return x 
    
def iter_predict(model, image, iter, device):
    output_list=[]
    for i in range(iter):
        output=model(image)
        image=image.to('cpu')
        output=output.to('cpu')
        output_list.append(output)
        image[:,1,:,:]=output[:,0,:,:]
        image=image.to(device)
    output=torch.concat(output_list, dim=1)
    return output.to(device)
    
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model=unet_bn()
    test_model=test_model.to(device=device)
    summary(test_model,(2,256,256))
