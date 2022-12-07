
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.fid_score import calculate_fid_given_paths
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--path',default=['a','b'], type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
def calculate_FID(source_dir,target_dir):
    path=[source_dir,target_dir]
    batch_size=50
    device=torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    num_workers = 0
    dims=2048

    fid_value = calculate_fid_given_paths(path,
                                          batch_size,
                                          device,
                                          dims,
                                          num_workers)
    return fid_value
    

if __name__ == '__main__':
    source_dir='/home/somebody/MMGEN-FaceStylor/faces_sample'
    target_dir='/home/somebody/MMGEN-FaceStylor/demo/res/images/1_Filisi_600'
    fid_value=calculate_FID(source_dir,target_dir)
    print('FID: ', fid_value)