import os
import glob

img_path = "data/images"
config_path="configs/test/configs"
work_path='./work_dirs/experiments/res'
for dir in os.listdir(img_path):
    config_file=os.path.join(config_path,dir)
    config_file=config_file+'.py'
    work_dir=os.path.join(work_path,dir)

    cmd='bash tools/dist_train.sh '+config_file+' 1 --work-dir '+work_dir+';'

    print(cmd)
    