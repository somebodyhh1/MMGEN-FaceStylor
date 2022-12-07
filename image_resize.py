import os
from PIL import Image
import glob

def resize(img_path):
    path_save = img_path
    for dir in os.listdir(img_path):
        save_path=os.path.join(path_save,dir)
        dir=os.path.join(img_path, dir)
        file=dir
        #print("file==",file)
        im = Image.open(file)
        im=im.resize((1024,1024))
        #im.thumbnail((200,200))
        #print(im.format, im.size, im.mode)
        im.save(save_path,'png')