import argparse
import os
import sys

import cv2
import mmcv
import torch
import torchvision.transforms as transforms
import utils.normal_image as normal_image
from torchvision import utils

# yapf: enable
# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa
import agilegan  # isort:skip  # noqa

from mmgen.apis import init_model  # isort:skip  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='AgileGAN Demo')
    parser.add_argument('source_path', help='source image path')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--ckpt', type=str, default=None, help='encoder checkpoint')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/agile_result.png',
        help='path to save image transfer result')
    args = parser.parse_args()
    return args


def load_image(image_path):
    img = cv2.imread(image_path)
    assert img is not None
    normal = normal_image.Normal_Image()
    img = normal.run(img)

    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = img.convert('RGB')
    img = T(img)
    img = img.unsqueeze(0).float()
    return img

def test():
    args = parse_args()
    source_path=args.source_path
    save_path_dir=args.save_path
    for file in os.listdir(source_path):

        file_path = os.path.join(source_path, file)
        # load image
        print('file==',file_path)
        #file_path='faces/fanbingbing_108.png'
        load_image(file_path)

def main():
    args = parse_args()
    source_path=args.source_path
    save_path_dir=args.save_path

    iter=400
    #print(args.config)
    #args.config.stylegan_weights=ckpt
    model = init_model(args.config, checkpoint=args.ckpt,
                    device=args.device).eval()
    cnt=0
    img_save_dir=os.path.join('demo/res',(str)(iter))
    for file in os.listdir(source_path):
        cnt+=1
        if(cnt>200):
            break
        save_path=os.path.join(save_path_dir,file)
        if os.path.exists(save_path):
            continue
        file_path = os.path.join(source_path, file)
        # load image
        print('file==',file_path)
        #file_path='faces/fanbingbing_108.png'
        src_img = load_image(file_path).to(args.device)
        # put throught encoder-decoder
        with torch.no_grad():
            image = model(src_img)
        
        mmcv.mkdir_or_exist(os.path.dirname(save_path))
        utils.save_image(image, save_path, normalize=True)


if __name__ == '__main__':
    main()
