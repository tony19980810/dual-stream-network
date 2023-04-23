import argparse
import os
import sys

from d2l import torch as d2l
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from model import dual_stream as create_model
from rollout import VITAttentionRollout
from grad_rollout import VITAttentionGradRollout
import matplotlib.pyplot as plt
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default="./data/MEIP_crop_30/02_ha6_di",
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
# def show_mask_on_images(mask):

if __name__ == '__main__':
    devices = [d2l.try_gpu(i) for i in range(1)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_args()
    model = create_model(num_classes=36, has_logits=False).to(device)
    # load model weights
    model = torch.nn.DataParallel(model, device_ids=devices)
    model_weight_path = "./weights/model-2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    imgs = []
    for root, dirs, files in os.walk(args.image_path):
        for f in files:
            img = Image.open(os.path.join(args.image_path, f))


            img = transform(img)

            imgs.append(img)


    imgs = torch.stack(imgs, dim=0).unsqueeze(0)
    print(imgs.shape)
    input_tensor=imgs
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
            args.discard_ratio, args.head_fusion)

    print(mask)
    plt.bar(range(len(mask)), mask)
    plt.show()
