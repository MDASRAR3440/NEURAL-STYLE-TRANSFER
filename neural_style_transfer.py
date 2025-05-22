import argparse
import torch
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from utils import load_image, save_image, get_features, gram_matrix, im_convert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.jpg')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight-content', type=float, default=1e4)
    parser.add_argument('--weight-style', type=float, default=1e2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content = load_image(args.content).to(device)
    style = load_image(args.style, shape=content.shape[-2:]).to(device)
    target = content.clone().requires_grad_(True).to(device)

    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    style_weights = {'0': 1.0, '5': 0.8, '10': 0.5, '19': 0.3, '28': 0.1}
    content_weight = args.weight_content
    style_weight = args.weight_style

    optimizer = optim.Adam([target], lr=args.lr)
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for i in range(1, args.epochs+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['21'] - content_features['21'])**2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            _, d, h, w = target_feature.shape
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch {i}, Total loss: {total_loss.item()}")

    save_image(im_convert(target), args.output)

if __name__ == "__main__":
    main()