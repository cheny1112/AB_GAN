import os
import torch

# GPUid
gpu_ids = '0'

# 是否训练
isTrain = True

# 是否有gpu, 若是有就使用
Tensor = torch.cuda.FloatTensor if gpu_ids else torch.Tensor

# 存储路径
checkpoints_dir = 'py_lang/Adjust_brightness/checkpoints'

# 存储的名字
name = 'experiment_name'

class BaseModel():
    #def name(self):
        #return 'BaseModel'
    global gpu_ids

    global isTrain
    #self.opt = opt
    global Tensor

    global save_dir

    global checkpoints_dir

    global name
    def initialize(self):
        
        #gpu_ids 默认是0
        self.gpu_ids = gpu_ids
        #是否训练
        self.isTrain = isTrain
        #是否有gpu
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        #存储路径
        self.save_dir = os.path.join(checkpoints_dir, name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    #def update_learning_rate():
       #pass
