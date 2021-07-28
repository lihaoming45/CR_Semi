import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch,):
        super(double_conv2, self).__init__()
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
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.pooling3d = nn.MaxPool3d(2)
        self.conv = double_conv(in_ch, out_ch)
        # self.mpconv = nn.Sequential(
        #     nn.MaxPool3d(2),
        #     double_conv(in_ch, out_ch)
        # )

    def forward(self, x):
        x = self.pooling3d(x)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
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

    def forward(self, x1, x2):  # x1--up , x2 ---down
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
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
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_1 = nn.Conv3d(in_ch, int(in_ch/2), 3, padding=1)
        self.InNorm = nn.Sequential(nn.InstanceNorm3d(int(in_ch/2)), nn.ReLU(inplace=True))
        self.conv = nn.Conv3d(int(in_ch/2), out_ch, 1)
        # self.conv_1 = nn.Conv3d(in_ch,in_ch,3,padding=1)

    def forward(self, x):
        x= self.conv_1(x)
        x= self.InNorm(x)
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
        x = F.softmax(x, dim=1)
        return x


class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
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

    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
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
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class up_reconst(nn.Module):
    def __init__(self, in_ch, out_ch, up_size, mode='MY'):
        super(up_reconst, self).__init__()
        self.mode = mode
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                # nn.BatchNorm3d(in_ch * 2 // 3),
                nn.InstanceNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up_size = up_size
            self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv = double_conv2(in_ch, out_ch, mode=self.mode)

    def forward(self, x1):  # x1--up , x2 ---down
        x1 = self.up(x1)
        diffX = x1.size()[2] - self.up_size[0]
        diffY = self.up_size[1] - x1.size()[3]
        diffZ = self.up_size[2] - x1.size()[4]

        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,))
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        return x


class reconst_Decoder(nn.Module):
    def __init__(self, cc, n_classes=2, mode='MY'):
        super(reconst_Decoder, self).__init__()
        self.mode = mode
        self.up_1 = up_reconst(16 * cc, 8 * cc, up_size=(8, 14, 14), mode=self.mode)
        self.up_2 = up_reconst(8 * cc, 4 * cc, up_size=(16, 28, 28), mode=self.mode)
        self.up_3 = up_reconst(4 * cc, 2 * cc, up_size=(32, 56, 56), mode=self.mode)
        self.up_4 = up_reconst(2 * cc, cc, up_size=(64, 112, 112), mode=self.mode)
        self.outconv = outconv(cc, n_classes, task='resconst')

    def forward(self, x):

        x0 = self.up_1(x)
        x1 = self.up_2(x0)
        x2 = self.up_3(x1)
        x3 = self.up_4(x2)
        y = self.outconv(x3)

        return y


class Encoder(nn.Module):
    def __init__(self, cc, n_channels):
        super(Encoder, self).__init__()
        self.inconv = inconv(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.down4 = down(8 * cc, 16 * cc)

        # self.outconv = outconv(cc, n_classes)

    def forward(self, x):
        x0 = self.inconv(x)  # cc1
        x1 = self.down1(x0)  # cc2
        x2 = self.down2(x1)  # cc4
        x3 = self.down3(x2)  # cc8
        x4 = self.down4(x3)  # cc16

        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, cc, n_classes):
        super(Decoder, self).__init__()

        self.up1 = up(24 * cc, 8 * cc)
        self.up2 = up(12 * cc, 4 * cc)
        self.up3 = up(6 * cc, 2 * cc)
        self.up4 = up(3 * cc, cc)

        self.outconv = outconv(cc, n_classes)

    def forward(self, *x_list):
        x0, x1, x2, x3, x4 = x_list[0]
        # level 1
        x = self.up1(x4, x3)  # cc8

        # level 2
        x = self.up2(x, x2)  # cc4

        # level 3
        x = self.up3(x, x1)  # cc2

        # level 4
        x = self.up4(x, x0)

        # x_final = self.up4(x, x03, x02, x01, x0)
        # x_edge = self.up4_edge(x, x03, x02, x01, x0)

        y_final = self.outconv(x)

        return y_final


cc = 32  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing

class model_MT(nn.Module):
    def __init__(self, cc, n_channels, n_classes, mode='main'):
        super(model_MT, self).__init__()
        self.net_E = Encoder(cc=cc, n_channels=n_channels).to('cuda:0')
        self.net_D = Decoder(cc=cc, n_classes=n_classes).to('cuda:1')
        self.mode = mode

    def forward(self, x):
        out_E = self.net_E(x)
        out_E = [i.to('cuda:1') for i in out_E]
        y = self.net_D(out_E)

        return y


class asym_model(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(asym_model, self).__init__()
        self.result = {}
        self.inconv = inconv(n_channels, cc)
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
        # self.up4 = up(3 * cc, cc)

        self.outconv = outconv(2 * cc, n_classes)


    def forward(self, x):

        x0 = self.inconv(x)  # cc1
        x1 = self.down1(x0)  # cc2
        x2 = self.down2(x1)  # cc4
        x3 = self.down3(x2)  # cc8
        x4 = self.down4(x3)  # cc16

        x = self.up1(x4, x3)
        # level 2
        x = self.up2(x, x2)  # cc4
        # level 3
        x = self.up3(x, x1)  # cc2
        fine_feature = x

        # level 4

        # y_final = self.outconv(x)
        self.result['coarse'] = self.outconv(x)
        # self.result['p2'] = F.interpolate(fine_feature, scale_factor=2, mode='trilinear', align_corners=False)
        self.result['p2_1'] =fine_feature
        self.result['p2_2'] = x0

        return self.result



if __name__ == '__main__':
    import time
    import os
    from thop import profile
    from thop import clever_format
    from model.PointRend2.pointrend3_CR import PointHead,PointRend

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    x = torch.rand((1, 64, 32, 48, 48)).cuda()
    x0 = torch.rand((1, 32, 64, 96, 96)).cuda()

    input = torch.randn(1, 1, 128, 192, 192).cuda()
    p2_1 = torch.randn((1, 64, 32, 48, 48)).cuda()
    p2_2 = torch.randn((1, 32, 64, 96, 96)).cuda()
    coarse = torch.randn((1, 3, 32, 48, 48)).cuda()

    net = PointRend(asym_model(n_channels=1, n_classes=3), PointHead()).cuda()
    net.eval()
    flops, params = profile(net, inputs=(input,))


    # R_head = PointHead().cuda()
    # R_head.training = False
    # R_out = R_head(input, p2_1, p2_2, coarse)
    # flops, params = profile(R_head, inputs=(input, p2_1, p2_2, coarse))

    # N_head = norm_head().cuda()
    # out = N_head(x0)
    # model = Unet_3D(n_channels=1, n_classes=3).cuda()
    # flops, params = profile(N_head, inputs=(x, x0))
    # flops,params = profile(model,inputs=(input))

    flops, params = clever_format([flops, params], "%.3f")
    # status = "flops={:.3f},param={:.3f}".format(flops, params)
    print(flops, params)

