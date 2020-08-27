from torch.utils.data import DataLoader
from data.dataset import train_data

def load_data(train_data):

    t_l = DataLoader(train_data, batch_size=32, shuffle=True)

    return t_l

train_loader = load_data(train_data)

print(len(train_loader))