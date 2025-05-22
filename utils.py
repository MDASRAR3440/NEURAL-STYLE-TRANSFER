from PIL import Image
import torch
from torchvision import transforms

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

def save_image(tensor, path):
    image = tensor.squeeze().to('cpu').clone().detach()
    image = transforms.ToPILImage()(image)
    image.save(path)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor((0.229, 0.224, 0.225)).view(3,1,1)
    image = image + torch.tensor((0.485, 0.456, 0.406)).view(3,1,1)
    image = image.clamp(0, 1)
    return image

def get_features(image, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1',
              '10': 'conv3_1', '19': 'conv4_1',
              '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram