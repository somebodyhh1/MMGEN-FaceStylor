import PIL.Image as Image
import os



IMAGE_SIZE = 1000  # 每张小图片的大小
IMAGE_ROW = 3  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列



# 定义图像拼接函数
def image_compose(image_names,IMAGE_SAVE_PATH):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


def main():
    for i in range(6):
        image_name='img'+(str)(i)+'.png'
        images=[]
        for iter in range(400,600,20):
            path=os.path.join('demo/res',(str)(iter))
            file_path=os.path.join(path,image_name)
            images.append(file_path)
        image_save_path=os.path.join('demo/res',image_name)
        image_compose(images,image_save_path)
main()
