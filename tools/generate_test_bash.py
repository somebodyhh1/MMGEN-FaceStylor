import os
import glob

img_path = "data/images"
config_path="configs/test/generates"
save_path_dir="res";
for dir in os.listdir(img_path):
    config_file=os.path.join(config_path,dir)
    config_file=config_file+'.py'
    save_path=os.path.join(save_path_dir,dir)

    cmd='python demo/agilegan_demo_lots.py demo/src_img '+config_file+' --save-path '+save_path+';'

    print(cmd)
    