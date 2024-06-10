import torch
import torch.nn as nn
import torchvision

class TemporalResNet(nn.Module):
    def __init__(self):
        super(TemporalResNet, self).__init__()
        self.base_model = torchvision.models.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1
        )
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.fc = nn.Identity()
        
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fc = nn.Linear(512, 2)

    def forward(self, imgs):
        batch_size, num_frames, C, H, W = imgs.shape
        imgs = imgs.view(batch_size * num_frames, C, H, W)
        cnn_features = self.base_model(imgs)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        attn_output, _ = self.attention(cnn_features, cnn_features, cnn_features)
        avg_features = torch.mean(attn_output, dim=1)
        
        output = self.fc(avg_features)
        return output


def resnet_binary():
    model = torchvision.models.resnet18(
        weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1
    )
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  
    return model


class SiameseResNet(torch.nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        self.base_model = torchvision.models.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1
        )
        self.base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, img1, img2):
        features1 = self.base_model(img1)
        features2 = self.base_model(img2)
        combined_features = torch.cat((features1, features2), dim=1)
        result = self.classifier(combined_features)
        return result
    