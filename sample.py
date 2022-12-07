import numpy as np
import os
from PIL import Image
path='faces'
save_path='faces_sample'


imgs=[]
for dir in os.listdir(path):
    imgs.append(dir)
length=len(imgs)
sample_num=50
samples=np.random.choice(length, sample_num, replace=False)
for i in samples:
    file=os.path.join(path,imgs[i])
    im = Image.open(file)
    new_path=os.path.join(save_path,imgs[i])
    im.save(new_path,'png')