import argparse
import os
from shutil import rmtree
from eval import calculate_FID
from image_resize import resize
inf=0x7fffffff
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1 2"
def find_proper_config(img_dir):
    return 'configs/sample_config/1_Filisi.py'
    sample_img_dir='data/sample_data'
    config_dir='configs/sample_config'
    min_fid=inf
    config_file=None
    for dir in os.listdir(sample_img_dir):
        fid=calculate_FID(img_dir,os.path.join(sample_img_dir,dir))
        print('fid score to',dir,'is',fid)
        if fid<min_fid:
            min_fid=fid
            config_file=os.path.join(config_dir,dir)+'.py'
    print('the best config should be',config_file)
    return config_file

def main():
    parser = argparse.ArgumentParser(description='style transfer demo')
    parser.add_argument('-i', help='The input path images', required=True, dest='input', type=str)
    parser.add_argument('-cropped', help='Is the images already cropped',default='False', dest='cropped')
    parser.add_argument('-gpu-num', help='num of gpu to train the model', default=1, dest='gpu_num')
    parser.add_argument('-work-dir', help='The work dir of the process',required=True, dest='work_dir')
    args = parser.parse_args()

    assert os.path.exists(args.input), 'The input path does not exists'

    image_path=os.path.abspath(args.input)
    if args.cropped=='False':
        cropped_img_path='data/cropped_img/'
        if os.path.exists(cropped_img_path):
            rmtree(cropped_img_path)
        os.mkdir(cropped_img_path)
        cropped_img_path=os.path.abspath(cropped_img_path)+'/'
        os.chdir("anime-face-detector")
        crop_cmd="python main.py -i {input} -crop-location {output}".format(input=image_path,output=cropped_img_path)
        os.system(crop_cmd)
        os.chdir("./")
        image_path=cropped_img_path
    
    resize(image_path)

    reference_config_file=find_proper_config(image_path)
    config_file='configs/transfer_config/config.py'
    data=''
    with open(reference_config_file,'r+') as f:
        finded=False
        for line in f.readlines():
            if finded==False and line.find('imgs_root')==0:
                line='imgs_root=\''+image_path+'\'\n'
                finded=True
            data+=line
    with open(config_file,'w+') as f:
        f.writelines(data)
    print("current dir==",os.getcwd())
    train_cmd="bash tools/dist_train.sh {config_file} {gpu_num} --work-dir {work_dir}". \
              format(config_file=config_file,gpu_num=args.gpu_num,work_dir=args.work_dir)
    os.system(train_cmd)


if __name__=="__main__":
    main()