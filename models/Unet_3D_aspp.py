import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DenseASPP_3D import DenseASPP

class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch,mode):
        super(double_conv2, self).__init__()
        self.mode=mode
        self.conv_13 = nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.MY_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))

        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1)
        self.MY_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.conv_13(x)
        x = self.MY_InNorm_Relu_1(x)

        x = self.conv_2(x)
        x = self.MY_InNorm_Relu_2(x)
        return x

        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, mode='MY'):
        super(down, self).__init__()
        self.mode = mode
        self.pooling3d = nn.MaxPool3d(2)
        self.conv = double_conv(in_ch, out_ch, mode=self.mode)
        # self.mpconv = nn.Sequential(
        #     nn.MaxPool3d(2),
        #     double_conv(in_ch, out_ch)
        # )

    def forward(self, x):
        x = self.pooling3d(x)
        self.conv.mode = self.mode
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, mode='MY'):
        super(up, self).__init__()
        self.mode = mode
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                # nn.BatchNorm3d(in_ch * 2 // 3),
                nn.InstanceNorm3d(in_ch * 2 // 3),
                # nn.GroupNorm(16, in_ch * 2 // 3),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = double_conv2(in_ch, out_ch, mode=self.mode)

    def forward(self, x1, x2):  # x1--up , x2 ---down
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,))
        x = torch.cat([x2, x1], dim=1)
        self.conv.mode = self.mode
        x = self.conv(x)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch, mode='MY'):
        super(up3, self).__init__()
        self.mode = mode
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                # nn.BatchNorm3d(in_ch * 2 // 3),
                nn.InstanceNorm3d(in_ch * 2 // 3),
                # nn.GroupNorm(16, in_ch * 2 // 3),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(inplace=True),

            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                # nn.BatchNorm3d(in_ch * 2 // 3),
                nn.InstanceNorm3d(in_ch * 2 // 3),
                # nn.GroupNorm(16, in_ch * 2 // 3),

                nn.ReLU(inplace=True),
                # nn.LeakyReLU(inplace=True),

            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up5, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                # nn.BatchNorm3d(in_ch * 2 // 3),
                nn.InstanceNorm3d(in_ch * 2 // 3),
                # nn.GroupNorm(16,in_ch * 2 // 3),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),

            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4, x5):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        x = torch.cat([x4, x3, x2, x1, x5], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, task='seg'):
        super(outconv, self).__init__()
        self.task = task
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.conv_1 = nn.Conv3d(in_ch,in_ch,3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        # if self.task == 'seg':
        #     x = F.softmax(x, dim=1)
        # else:
        #     x = F.sigmoid(x)
        return x




class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch, mode='MY'):
        super(double_conv, self).__init__()
        self.mode = mode
        self.conv_1 = nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.MY_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))

        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1)
        self.MY_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.conv_1(x)
        x = self.MY_InNorm_Relu_1(x)
        x = self.conv_2(x)
        x = self.MY_InNorm_Relu_2(x)

        return x

class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch, mode):
        super(double_conv_in, self).__init__()
        self.mode = mode
        self.conv_11 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        # self.conv_12 = nn.Conv3d(out_ch, out_ch, 3, padding=2, dilation=2)
        self.MY_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))


        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, stride=1)
        self.MY_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))


        self.pooling3d = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv_11(x)
        # x = self.conv_12(x)

        x = self.MY_InNorm_Relu_1(x)

        x = self.conv_2(x)
        x = self.MY_InNorm_Relu_2(x)

        x = self.pooling3d(x)

        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,mode='MY'):
        super(inconv, self).__init__()
        self.mode=mode
        self.conv = double_conv_in(in_ch, out_ch,mode=self.mode)

    def forward(self, x):
        self.conv.mode=self.mode
        x = self.conv(x)
        return x






cc = 32  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing


class Unet_3D_aspp(nn.Module):
    def __init__(self, n_channels, n_classes, n_counting_class=5, mode='MY', task='seg'):
        super(Unet_3D_aspp, self).__init__()
        self.mode = mode
        self.task = task
        self.inconv = inconv(n_channels, cc,mode=self.mode)
        self.Bridgr_DenseAspp = DenseASPP(inchannels=cc * 4, num_classes=cc * 4)
        # self.task_reconst = reconst_Decoder(cc=cc,n_classes=2,mode=self.mode)
        # self.couting_out = nn.Linear(256,n_counting_class)
        # self.avgpool3D = nn.AvgPool3d(kernel_size=(4,7,7),stride=1)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.down4 = down(8 * cc, 16 * cc)

        self.up1 = up(24 * cc, 8 * cc)
        self.up2 = up(12 * cc, 4 * cc)
        self.up3 = up(6 * cc, 2 * cc)
        self.up4 = up(3 * cc, cc)


        self.outconv = outconv(cc, n_classes)


    def forward(self, x):
        x0 = self.inconv(x)  # cc1
        x1 = self.down1(x0)  # cc2
        x2 = self.down2(x1)  # cc4

        aspp_bridge = self.Bridgr_DenseAspp(x2)

        x3 = self.down3(x2)  # cc8
        x4 = self.down4(x3)  # cc16

        # level 1
        x = self.up1(x4, x3)

        # level 2
        x = self.up2(x, aspp_bridge)  # cc4

        # level 3
        x = self.up3(x, x1)  # cc2

        # level 4
        x = self.up4(x, x0)

        # x_final = self.up4(x, x03, x02, x01, x0)
        # x_edge = self.up4_edge(x, x03, x02, x01, x0)

        y_final = self.outconv(x)
        y_final = F.upsample(y_final, scale_factor=2, mode="trilinear", align_corners=False)

        return y_final


if __name__ == '__main__':
    import time
    import os
    from thop import profile
    from thop import clever_format
    from model.PointRend2.pointrend3_CR import PointHead

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


    input = torch.randn(1, 1, 128, 192, 192).cuda()


    model = Unet_3D(n_channels=1, n_classes=3).cuda()
    flops,params = profile(model,inputs=(input,))

    flops, params = clever_format([flops, params], "%.3f")
    # status = "flops={:.3f},param={:.3f}".format(flops, params)
    print(flops, params)
