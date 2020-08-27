import time
from models.EnlightenGAN import EnlightenGAN
from data.dataset import train_data
from data.data_loader import load_data
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
from generators.Unet_generator import UNet
from discriminators.noNorm_disc import NoNormDiscriminator 

# GPUがあれば、GPUを使って
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
G = model.to(device)

model = UNet()
D = model.to(device)

# Optimizerの設定
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# loss
criterion = nn.BCELoss()

def train_model(G, D, G_optimizer, D_optimizer, epochs):
    # 訓練モードへ
    G.train()
    D.train()

    # 本物のラベルは1
    y_real = Variable(torch.ones(batch_size, 1))
    # 偽物のラベルは0
    y_fake = Variable(torch.zeros(batch_size, 1))

    if torch.cuda.is_available():
        y_fake = y_fake.cuda()
        y_real = y_real.cuda()

    D_running_loss = 0
    G_running_loss = 0