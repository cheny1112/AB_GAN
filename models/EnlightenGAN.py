import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import tools.util as util
from tools.image_pool import ImagePool
from base_model import BaseModel
import networks
import sys


class EnlightenGAN(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self): # , opt
        BaseModel.initialize(self) # , opt
        
        batchSize = 32
        fineSize = 256
        input_nc = 3
        output_nc = 3
        vgg = 0
        skip = 0.8
        ngf = 64
        pool_size = 50
        norm = 'instance'
        lr = 0.0001
        no_dropout = True
        no_lsgan = True
        continue_train = True
        use_wgan = 0.0
        use_mse = True
        beta1 = 0.5
        which_direction = 'AtoB'
        new_lr = True
        niter_decay = 100
        l1 = 10.0

        # batch size
        nb = batchSize
        # 图像size
        size = fineSize
        #self.opt = opt
        self.input_A = self.Tensor(nb, input_nc, size, size)
        self.input_B = self.Tensor(nb, output_nc, size, size)
        self.input_img = self.Tensor(nb, input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        # Default 0, use perceptrual loss
        if vgg > 0:
            self.vgg_loss = networks.PerceptualLoss()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # default=0.8, help='B = net.forward(A) + skip*A'
        skip = True if skip > 0 else False

        # which_model_netG,  default = 'unet-256', selects model to use for netG
        # ngf, default = 64, of gen filters in first conv layer'
        # norm, default = 'instance', instance normalization or batch normalization
        # no_dropout, default = 'True', no dropout for the generator
        self.netG_A = networks.define_G(input_nc, output_nc,
                                        ngf, which_model_netG, norm, not no_dropout, self.gpu_ids, skip=skip)
            
        if not self.isTrain or continue_train:
            #which epoch to load
            which_epoch = 'lastest'
            self.load_network(self.netG_A, 'G_A', which_epoch)

        # --pool_size', default=50, help='the size of image buffer that stores previously generated images'
        # lr, default=0.0001
        if self.isTrain:
            self.old_lr = lr
            self.fake_A_pool = ImagePool(pool_size)
            self.fake_B_pool = ImagePool(pool_size)
            # define loss functions
            if use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                # no_lsgan = True
                self.criterionGAN = networks.GANLoss(use_lsgan=not no_lsgan, tensor=self.Tensor)
            if use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=lr, betas=(beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)

        self.real_B = Variable(self.input_B, volatile=True)

    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if self.skip == 1:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ("latent_real_A", latent_real_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
         # = self.latent_real_A + self.opt.skip * self.real_A
        self.L1_AB = self.criterionL1(self.fake_B, self.real_B) * self.l1
        self.loss_G = self.L1_AB
        self.loss_G.backward()


    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def get_current_errors(self, epoch):
        L1 = self.L1_AB.data[0]
        loss_G = self.loss_G.data[0]
        return OrderedDict([('L1', L1), ('loss_G', loss_G)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.lr / self.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
