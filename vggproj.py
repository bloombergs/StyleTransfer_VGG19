import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    image = transform(image).unsqueeze(0)
    return image

def imshow(tensor):
    image = tensor.squeeze().detach().cpu()
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = models.vgg19(weights='DEFAULT').features
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        layers = []
        for layer in self.vgg:
            x = layer(x)
            layers.append(x)
        return layers

def get_content_loss(target, content):
    return nn.functional.mse_loss(target, content)

def get_style_loss(target, style):
    def gram_matrix(tensor):
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

    G_target = gram_matrix(target)
    G_style = gram_matrix(style)
    return nn.functional.mse_loss(G_target, G_style)

def style_transfer(content_img, style_img, num_steps=50, style_weight=1000000, content_weight=1):
    vgg = VGG().to(device).eval()
    
    content_layer = 21  
    style_layers = [0, 5, 10, 19, 28] 

    content_target = vgg(content_img)[content_layer]
    style_targets = [vgg(style_img)[i] for i in style_layers]

    generated = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=0.003)

    for step in range(num_steps):
        optimizer.zero_grad()
        
        generated_layers = vgg(generated)
        content_loss = get_content_loss(generated_layers[content_layer], content_target)
        
        style_loss = 0
        for i in range(len(style_layers)):
            style_loss += get_style_loss(generated_layers[style_layers[i]], style_targets[i])

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'Step {step}, Total loss: {total_loss.item()}')

    return generated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_img = load_image('content.jpg').to(device) 
style_img = load_image('style.jpg').to(device)   

output = style_transfer(content_img, style_img)

imshow(output)
