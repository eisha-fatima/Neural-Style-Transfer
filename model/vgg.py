import torch
import torch.nn as nn
import torchvision.models as models

from .loss import Normalization, ContentLoss, StyleLoss

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
norm_mean = torch.tensor([0.485, 0.456, 0.406])
norm_std = torch.tensor([0.229, 0.224, 0.225])

class VGG19(nn.Module):
    def __init__(
        self, 
        style_layers:list=style_layers, 
        content_layers:list=content_layers,
        norm_mean:torch.tensor=norm_mean,
        norm_std:torch.tensor=norm_std,
        device:torch.device | str = 'cuda',
    ) -> None:
        super(VGG19, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        
        self.norm_mean = norm_mean.to(device)
        self.norm_std = norm_std.to(device)
        
        self.content_losses = []
        self.style_losses = []
        
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.device = device
    
    
    def build_model(
        self,
        content_img,
        style_img
    ):
        
        """
        Modifies the original VGG19 Architecture to include Normalization, Content Loss and Style Loss layers.

        Returns:
            Modified VGG19 Model with additional layers and without top linear layers.
            List of Layers added for Content and Style Loss.
        """
        
        self.model = nn.Sequential(
            Normalization(self.norm_mean, self.norm_std).to(self.device)
        )
        
        i = 0
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized Layer: {layer.__class__.__name__}')
            
            self.model.add_module(name, layer)

            if name in self.content_layers:
                target = self.model(content_img).detach()
                content_loss = ContentLoss(target)
                self.model.add_module(f'content_loss_{i}', content_loss)
                self.content_losses.append(content_loss)
            
            if name in self.style_layers:
                target_feature = self.model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module(f'style_loss_{i}', style_loss)
                self.style_losses.append(style_loss)
        
        for i in range(len(self.model) - 1 , -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break
        self.model = self.model[:(i+1)]
        return self.model, self.style_losses, self.content_losses