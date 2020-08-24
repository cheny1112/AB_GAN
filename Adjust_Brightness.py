from PIL import Image, ImageEnhance
import glob, os
from decimal import Decimal

path = 'C:/Programming/data_new'
out_path = 'C:/Users/chenxi/Desktop/output/'
if (os.path.isdir(out_path)==False):
    os.mkdir(out_path)

#这个是亮度系数，用0.2做对比测试
factor = Decimal('0.8')
#计数
i = 1

for file in glob.glob(path + '/*.jpg'):
    file_path, file_name = os.path.split(file)
    name, exts = os.path.splitext(file_name)
    #按照file一个一个的读取，然后一个一个的处理，然后输出
    img = Image.open(file)
    #先进行size调整，调整成600x400
    #p_w_picpath = Image.open('pythonfan.jpg')
    new_size = (600, 400)
    new_img = img.resize(new_size, Image.ANTIALIAS)
    #new_img.save(out_path + 'normal' + '_' + name + '.png')
    
    #进行增强
    enhancer = ImageEnhance.Brightness(new_img)
    img_output = enhancer.enhance(factor)
    #输出
    img_output.save(out_path + 'light' + str(factor) + '_' + name + '.jpg')#这里要写PNG格式
    i += 1
    if(i % 1 == 0):
        factor -= Decimal('0.2')
        #print(factor)

