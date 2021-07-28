import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch, mode):
        super(double_conv2, self).__init__()
        self.mode = mode
        self.conv_13 = nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.MY_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.GE_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        # self.Syn_InNorm_Relu_1 = nn.Sequential( nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1)
        self.MY_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.GE_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        # self.Syn_InNorm_Relu_2 = nn.Sequential( nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_13(x)
        if self.mode == 'MY':
            x = self.MY_InNorm_Relu_1(x)
        elif self.mode == 'GE':
            x = self.GE_InNorm_Relu_1(x)
        # elif self.mode == "Syn":
        #     x = self.Syn_InNorm_Relu_1(x)
        x = self.conv_2(x)
        if self.mode == 'MY':
            x = self.MY_InNorm_Relu_2(x)
        elif self.mode == 'GE':
            x = self.GE_InNorm_Relu_2(x)
        # elif self.mode == "Syn":
        #     x = self.Syn_InNorm_Relu_2(x)
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
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_1 = nn.Conv3d(in_ch, int(in_ch / 2), 3, padding=1)
        self.InNorm = nn.Sequential(nn.InstanceNorm3d(int(in_ch / 2)), nn.ReLU(inplace=True))
        self.conv = nn.Conv3d(int(in_ch / 2), out_ch, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.InNorm(x)
        x = self.conv(x)
        # x = self.up(x)
        return x


class conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.conv_1 = nn.Conv3d(in_ch,in_ch,3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv_edge(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv_edge, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, dilation=1)
        # self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_1(x)
        # x = self.conv(x)
        # x = F.softmax(x, dim=1)
        return x


class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch, mode='MY'):
        super(double_conv, self).__init__()
        self.mode = mode
        self.conv_1 = nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.MY_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.GE_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        # self.Syn_InNorm_Relu_1 = nn.Sequential( nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1)
        self.MY_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.GE_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        # self.Syn_InNorm_Relu_2 = nn.Sequential( nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_1(x)
        if self.mode == 'MY':
            x = self.MY_InNorm_Relu_1(x)
        elif self.mode == 'GE':
            x = self.GE_InNorm_Relu_1(x)
        # elif self.mode=="Syn":
        #     x = self.Syn_InNorm_Relu_1(x)
        x = self.conv_2(x)
        if self.mode == 'MY':
            x = self.MY_InNorm_Relu_2(x)
        elif self.mode == 'GE':
            x = self.GE_InNorm_Relu_2(x)
        # elif self.mode == "Syn":
        #     x = self.Syn_InNorm_Relu_2(x)
        return x


class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch, mode):
        super(double_conv_in, self).__init__()
        self.mode = mode
        self.conv_11 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        # self.conv_12 = nn.Conv3d(out_ch, out_ch, 3, padding=2, dilation=2)
        self.MY_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.GE_InNorm_Relu_1 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        # self.Syn_InNorm_Relu_1 = nn.Sequential( nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))

        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, stride=1)
        self.MY_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        self.GE_InNorm_Relu_2 = nn.Sequential(nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))
        # self.Syn_InNorm_Relu_2 = nn.Sequential( nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True))

        self.pooling3d = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv_11(x)
        # x = self.conv_12(x)
        if self.mode == 'MY':
            x = self.MY_InNorm_Relu_1(x)
        elif self.mode == 'GE':
            x = self.GE_InNorm_Relu_1(x)
        # elif self.mode == "Syn":
        #     x = self.Syn_InNorm_Relu_1(x)
        x = self.conv_2(x)
        if self.mode == 'MY':
            x = self.MY_InNorm_Relu_2(x)
        elif self.mode == 'GE':
            x = self.GE_InNorm_Relu_2(x)
        # elif self.mode == "Syn":
        #     x = self.Syn_InNorm_Relu_2(x)
        x = self.pooling3d(x)

        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, mode='MY'):
        super(inconv, self).__init__()
        self.mode = mode
        self.conv = double_conv_in(in_ch, out_ch, mode=self.mode)

    def forward(self, x):
        self.conv.mode = self.mode
        x = self.conv(x)
        return x



cc = 32  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing



class Unet_3D(nn.Module):
    def __init__(self, n_channels, n_classes, n_counting_class=5, task='seg', is_infer=False):
        super(Unet_3D, self).__init__()
        self.task = task
        self.is_infer = is_infer
        self.inconv = inconv(n_channels, cc, mode=self.mode)
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

        # self.up20 = up(12 * cc, 4 * cc)
        # self.up2 = up3(16 * cc, 4 * cc)
        #
        # self.up30 = up(6 * cc, 2 * cc)
        # self.up31 = up3(8 * cc, 2 * cc)
        # self.up3 = up4(10 * cc, 2 * cc)
        #
        # self.up40 = up(3 * cc, cc)
        # self.up41 = up3(4 * cc, cc)
        # self.up42 = up4(5 * cc, cc)
        # self.up4 = up5(6 * cc, cc)
        # 输出边界
        # self.up4_edge = up5(6 * cc, cc)
        # self.outconv_edge = outconv(cc,n_classes)


        # self.outconv = outconv(cc, n_classes)
        self.outconv_up4 = outconv(cc, n_classes)

    def forward(self, x):
        # self.inconv.mode = self.mode
        # x = x.unsqueeze(0)
        x0 = self.inconv(x)  # cc1

        # self.down1.mode = self.mode
        x1 = self.down1(x0)  # cc2

        # self.down2.mode = self.mode
        x2 = self.down2(x1)  # cc4

        # self.down3.mode = self.mode
        x3 = self.down3(x2)  # cc8

        # self.down4.mode = self.mode
        x4 = self.down4(x3)  # cc16


        # level 1
        # self.up1.mode = self.mode
        x = self.up1(x4, x3)

        # level 2
        # self.up2.mode = self.mode
        x = self.up2(x, x2)  # cc4

        # level 3
        x = self.up3(x, x1)  # cc2

        # level 4
        x = self.up4(x, x0)
        final = self.outconv_up4(x)


        final = F.upsample(final, scale_factor=2, mode="trilinear", align_corners=False)

        return final


class norm_head(nn.Module):
    def __init__(self, cc=64, n_classes=3):
        super(norm_head, self).__init__()
        # self.up4 = up(3 * cc, cc)
        self.outconv_up4 = outconv(cc, n_classes)

    def forward(self, x, x0):
        # x = self.up4(x, x0)
        out = self.outconv_up4(x)
        return out


# class rendering_head(nn.Module)

if __name__ == '__main__':
    import time
    import os
    from thop import profile
    from thop import clever_format
    from model.PointRend2.pointrend3_CR import PointHead

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'


    input = torch.randn(1, 1, 128, 192, 192).cuda()

    x = torch.rand((1, 64, 32, 48, 48)).cuda()
    x0 = torch.rand((1, 32, 64, 96, 96)).cuda()

    N_head = norm_head().cuda()
    out = N_head(x,x0)
    flops, params = profile(N_head, inputs=(x, x0))


    # model = Unet_3D(n_channels=1, n_classes=3).cuda()
    # flops,params = profile(model,inputs=(input,))

    flops, params = clever_format([flops, params], "%.3f")
    # status = "flops={:.3f},param={:.3f}".format(flops, params)
    print(flops, params)
