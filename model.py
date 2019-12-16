import torch.nn as nn
import torch
from torch.nn.modules import ModuleList
import torchvision.models as models
import torch.nn.functional as F


class PartSelectNet(nn.Module):
    def __init__(self):
        super(PartSelectNet, self).__init__()
        self.theta = None

    def forward(self, x, orig):

        self.theta = torch.zeros((x.shape[0], 2,3)).to(x.device)
        self.theta[:,0,0] = torch.sigmoid(x[:,0,0])
        self.theta[:,0,2] = torch.tanh(x[:,1,0])
        self.theta[:,1,1] = torch.sigmoid(x[:,0,1])
        self.theta[:,1,2] = torch.tanh(x[:,1,1])
        grid = F.affine_grid(self.theta, size=([x.shape[0], 3, 64, 64]), align_corners=True).to(x.device)
        self.sample = F.grid_sample(input=orig, grid=grid, align_corners=True).to(x.device)
        # assert self.sample.shape == (x.shape[0],3,64,64)
        # theta shape(N, 2, 3)
        return self.sample, self.theta


class FaceSelectNet(torch.nn.Module):
    def __init__(self):
        super(FaceSelectNet, self).__init__()
        self.theta = None

    def forward(self, x, orig):
        self.theta = torch.zeros((x.shape[0], 2, 3)).to(x.device)
        self.theta[:,0,0] = torch.sigmoid(x[:,0,0])
        self.theta[:,0,2] = torch.tanh(x[:,1,0])
        self.theta[:,1,1] = torch.sigmoid(x[:,0,1])
        self.theta[:,1,2] = torch.tanh(x[:,1,1])
        grid = F.affine_grid(self.theta, size=([x.shape[0], 3, 128, 128]), align_corners=True).to(x.device)
        self.sample = F.grid_sample(input=orig, grid=grid, align_corners=True).to(x.device)
        # assert self.sample.shape == (x.shape[0],3,64,64)
        return self.sample, self.theta

class SkinHairModel(nn.Module):
    def __init__(self):
        super(SkinHairModel, self).__init__()
        model = models.resnet50(pretrained=True)
        self.resnet_layer = BackBone(model)

        # Skin and Hair
        self.skin_hair_model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),              # 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3,     # 32 x 32
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3,     # 64 x 64
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3,     # 128 x 128
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3,               # 128 x 128
                      stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.resnet_layer(x)
        out = self.skin_hair_model(out)
        assert out.shape == (x.shape[0], 2, 128, 128)
        return out

# ResNet + FPN
class BackBone(nn.Module):
    def __init__(self, model):
        super(BackBone,self).__init__()
        # num_ftrs= self.resnet_layer.fc.in_features
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        for para in list(self.resnet_layer.parameters())[:-1]:
            para.requires_grad = False

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool2d = nn.MaxPool2d(1, stride=2)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self,x):
        # Bottom-up
        c1 = self.resnet_layer[0:4](x)
        c2 = self.resnet_layer[4](c1)
        c3 = self.resnet_layer[5](c2)
        c4 = self.resnet_layer[6](c3)
        c5 = self.resnet_layer[7](c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)
        # p6 = self.maxpool2d(p5)
        # feature_maps = [p2, p3, p4, p5, p6]
        return torch.tanh(p4)


class NewFaceModel(nn.Module):
    def __init__(self):
        super(NewFaceModel,self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs= self.resnet.fc.in_features
        for para in list(self.resnet.parameters())[:-1]:
            para.requires_grad = False
        self.resnet.fc = nn.Linear(num_ftrs, 9 * 4)
        self.c4_layer = nn.Sequential(*list(self.resnet.children())[:-3])

        self.parts_selectnet = ModuleList([PartSelectNet() for _ in range(8)])
        self.face_selectnet = FaceSelectNet()
        self.parts_model = ModuleList([nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),      # (64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),     # (64, 64)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),          # (32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),     # (32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),     # (32, 32)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),          # (16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),     #  (16, 16)
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3,
                               stride=2, padding=1, output_padding=1),      #  (32, 32)
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3,
                               stride=2, padding=1, output_padding=1),        # (64, 64)
            nn.Tanh()
        )
            for _ in range(8)])
        self.skin_hair_model = SkinHairModel()
        self.bg_model = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.theta = None

    def get_all_theta(self):
        return self.theta

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(-1, 9, 2, 2)
        out_list = []
        theta_list = []
        for i in range(8):
            temp, theta = self.parts_selectnet[i](out[:, i], x)      # (N, 3, 64, 64)
            out_list.append(self.parts_model[i](temp))    # mask (N, 1, 64, 64)
            theta_list.append(theta)                     # theta (N, 2, 3)
        parts_out = torch.cat(out_list, dim=1)     # (N, 8, 64, 64)
        assert parts_out.shape == (x.shape[0], 8, 64, 64)
        del out_list
        parts_theta = (torch.stack(theta_list, dim=0)).transpose(1, 0) # (N, 8, 2, 3)
        del theta_list

        hair_sample, hair_theta = self.face_selectnet(out[:, 8], x)   # sample (N, 8, 128, 128) theta (N, 2, 3)
        hair_theta = torch.unsqueeze(hair_theta, dim=1) # (N, 1, 2, 3)
        hair_out = self.skin_hair_model(hair_sample)   # (N, 2, 128, 128)

        all_theta = torch.cat([parts_theta,hair_theta], dim=1) # (N, 9, 2, 3)
        self.theta = all_theta

        # parts_out Shape(N, 8, 64, 64)
        # face_out Shape(N, 2, 128, 128)

        # Compose up all individual masks

        # prepare for inverse
        temp = torch.Tensor([[0, 0, 1]]).float().to(x.device)  # Shape(1, 3)
        temp = temp.expand(x.shape[0], 9, -1, -1)              # Shape(N, 9, 1, 3)
        r_theta = torch.cat([all_theta, temp], dim=2)          # Shape(N, 9, 3, 3)
        r_theta = r_theta.inverse()
        # Sample all the masks back
        r_theta = r_theta[:, :, 0:2] # Shape(N, 9, 2, 3)
        all_parts = []
        for i in range(8):
            gird = F.affine_grid(theta=r_theta[:, i], size=[x.shape[0], 1, 512, 512], align_corners=True)
            all_parts.append(F.grid_sample(input=torch.unsqueeze(parts_out[:, i], dim=1),
                                           grid=gird, align_corners=True)) # Shape(N, 1, 512, 512)
        all_parts = torch.cat(all_parts, dim=1)
        assert all_parts.shape == (x.shape[0], 8, 512, 512)  # Shape(N, 8, 512, 512)
        all_outer = []
        for i in range(2):
            gird = F.affine_grid(theta=r_theta[:, 8], size=[x.shape[0], 1, 512, 512], align_corners=True)
            all_outer.append(F.grid_sample(input=torch.unsqueeze(hair_out[:, i], dim=1), grid=gird,
                                           align_corners=True))
        all_outer = torch.cat(all_outer, dim=1)
        assert all_outer.shape == (x.shape[0], 2, 512, 512)  # Shape(N, 2, 512, 512)
        all_skin = torch.unsqueeze(all_outer[:, 0], dim=1)   # Shape(N, 1, 512, 512))
        all_hair = torch.unsqueeze(all_outer[:, 1], dim=1)   # Shape(N, 1, 512, 512)))

        # BackGround
        bg = self.c4_layer(x)
        bg = self.bg_model(bg)                               # Shape(N, 1, 512 ,512)
        assert bg.shape == (x.shape[0], 1, 512, 512)
        final_output = torch.tanh(torch.cat([bg, all_skin, all_parts, all_hair], dim=1)) # Shape(N, 11, 512, 512)
        del all_parts, all_hair
        assert final_output.shape == (x.shape[0], 11, 512, 512)

        return final_output


