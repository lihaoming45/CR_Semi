import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sampling_point2 import sampling_points, point_sample


class Render_h(nn.Module):
    def __init__(self, in_c=99, num_classes=3, k=3, beta=0.75,num_points=70000):
        super().__init__()
        # self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, 64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(32, num_classes, 1)
        )

        self.k = k
        self.beta = beta
        self.num_points = num_points

    def forward(self, x, p2_1, p2_2, out):

        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """

        if not self.training:
            return self.inference(x, p2_1, p2_2, out)

        points = sampling_points(out, (x.shape[2] // 4) * (x.shape[-1] // 4), self.k, self.beta)

        coarse = point_sample(out, points, align_corners=False)
        fine_1 = point_sample(p2_1, points, align_corners=False)
        fine_2 = point_sample(p2_2, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine_1, fine_2], dim=1)

        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, p2_1, p2_2, out):

        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """

        # num_points = 8096

        while out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode="trilinear", align_corners=True)

            points_idx, points = sampling_points(out, self.num_points, training=self.training)

            coarse = point_sample(out, points, align_corners=False)
            # fine = point_sample(res2, points, align_corners=False)
            fine_1 = point_sample(p2_1, points, align_corners=False)
            fine_2 = point_sample(p2_2, points, align_corners=False)

            # feature_representation = torch.cat([coarse, fine], dim=1)
            feature_representation = torch.cat([coarse, fine_1, fine_2], dim=1)

            rend = self.mlp(feature_representation)

            B, C, D, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)

            if rend.dtype==torch.half:
                out = (out.reshape(B, C, -1).half()
                       .scatter_(2, points_idx, rend)
                       .view(B, C, D, H, W))
            else:
                out = (out.reshape(B, C, -1)
                       .scatter_(2, points_idx, rend)
                       .view(B, C, D, H, W))


        return {"fine": out}


class Rend_m(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        result = self.backbone(x)
        # result.update(self.head(x, result["res2"], result["coarse"]))
        result.update(self.head(x, result["p2_1"], result["p2_2"],result["coarse"]))
        return result

# if __name__ == "__main__":
#     x = torch.randn(3, 3, 256, 512).cuda()
#     from deeplab import deeplabv3
#     net = PointRend(deeplabv3(False), PointHead()).cuda()
#     out = net(x)
#     for k, v in out.items():
#         print(k, v.shape)
