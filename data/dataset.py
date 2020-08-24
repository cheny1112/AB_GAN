###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import os
import os.path
import torchvision.transforms as transforms
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

#choose the number of crop for patch discriminator，default=256
fineSize = 256

#是否进行剪裁
#scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
resize_or_crop = 'no'

#是否进行翻转
#if specified, do not flip the images for data augmentation
no_flip = False

#choose the number of crop for patch discriminator
#选择local判别器的patch数量
low_times = 200
high_times = 400

IMG_EXTENSIONS = ['JPG', 'JPEG', 'PNG', 'PPM', 'BMP',
                'jpg', 'jpeg', 'png', 'ppm', 'bmp']

root = 'C:/Programming/train_dataset'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#使用这个函数来读取文件夹里面的图片
def make_dataset(dir):
    images = []
    #Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

'''
#times = random.randint(low_times, high_times)/100. #low_time = 200, high_time = 400
#print(times)
dir_B = os.path.join(root + '/train' + 'B')
B_paths = make_dataset(dir_B)
B_paths = sorted(B_paths)
B_size = len(B_paths)

print(B_size)

index = 10

B_path = B_paths[index % B_size]
B_img = Image.open(B_path).convert('RGB')

transform_list = []
        
transform_list += [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), #这里归一化出现问题，问题原因是因为
                                                        #RGB三通道图只支持0~1的浮点数，和0~255的整数类型
                                        (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

B_img = transform(B_img)

w = B_img.size(2)#获得宽
h = B_img.size(1)#获得高
w_offset = random.randint(0, max(0, w - fineSize - 1))# fineSize = 320
h_offset = random.randint(0, max(0, h - fineSize - 1))

B_img = B_img[:, h_offset:h_offset + fineSize,
               w_offset:w_offset + fineSize]

idx = [i for i in range(B_img.size(2) - 1, -1, -1)]
idx = torch.LongTensor(idx)
B_img = B_img.index_select(2, idx)

idx = [i for i in range(B_img.size(1) - 1, -1, -1)]
idx = torch.LongTensor(idx)
B_img = B_img.index_select(1, idx)

#print(B_img) 似乎是因为进行了一次标准变换，讲值变成-1,1，所以在负数的部分的值出现异常

image_numpy = B_img.data.cpu().float().numpy()
image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0

print(image_numpy)
plt.imshow(image_numpy)
plt.show()
''' 

#读取数据集
class MyDataset(torch.utils.data.Dataset): # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    # 初始化一些需要传入的参数
    def __init__(self, root, transform=True):
        super(MyDataset,self).__init__()
        # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        # fh = open(root + datatxt, 'r')
        #imgs = make_dataset(root)
        self.root = root
        self.dir_A = os.path.join(root + '/train' + 'A')
        self.dir_B = os.path.join(root + '/train' + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        #对训练数据A, B进行排序与总数测量
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        #如果文件夹为空，则返回错误信息
        if self.A_size == 0 or self.B_size == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                            "Supported image extensions are: " +
                            ",".join(IMG_EXTENSIONS)))

        
        #self.imgs = imgs
        self.transform = transform
        #self.target_transform = target_transform
    def __getitem__(self, index):
        global fineSize
        global resize_or_crop
        global no_flip
        global low_times
        global high_times

        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
            
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        w = A_img.size(2)#获得宽
        h = A_img.size(1)#获得高
        w_offset = random.randint(0, max(0, w - fineSize - 1))# fineSize = 320
        h_offset = random.randint(0, max(0, h - fineSize - 1))

        A_img = A_img[:, h_offset:h_offset + fineSize,
            w_offset:w_offset + fineSize]
        B_img = B_img[:, h_offset:h_offset + fineSize,
            w_offset:w_offset + fineSize]


        if  resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            # A_gray = (1./A_gray)/255.
            #若无翻转
            if (not no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if (not no_flip) and random.random() < 0.5:
                times = random.randint(low_times, high_times)/100. #low_time = 200, high_time = 400
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img':input_img,
                'A_paths': A_path, 'B_paths': B_path}
        #return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.A_size

#根据自己定义的那个类MyDataset来创建数据集！注意是数据集！而不是loader迭代器
augmentation = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])


train_data = MyDataset(root,  transform = augmentation)

print(train_data)
#test_data=MyDataset(root,  transform=augmentation)

'''
#测试一下能不能读取数据集
idx = 300
traindata = train_data[idx]
#print(traindata)
tensor1 = traindata['input_img']
tensor2 = traindata['A_gray']
#tensor_cat = torch.cat([tensor1, tensor2], 1)
#img2 = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
#plt.imshow(img2)
#plt.axis('off') # 关掉坐标轴为 off
#print('label：',train_data[idx][1])#train[][0]为图片信息，train[][1]为label

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    #image = image.convert('BGR')
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0)  # pause a bit so that plots are updated

imshow(tensor1)


#输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

img1 = tensor_to_PIL(tensor1)
img2 = tensor_to_PIL(tensor2)

img1.show()
#img2.show()

#img2_new = np.expand_dims(img2, axis=2)
#img2_new = np.concatenate((img2_new, img2_new, img2_new), axis=-1)

#img2_new.show()

#img_cat = np.concatenate([img1, img2_new], 1)

#img_cat.show()
'''