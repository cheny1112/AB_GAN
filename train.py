import time
#from options.train_options import TrainOptions
#from data.data_loader import CreateDataLoader
#from models.models import create_model
#from util.visualizer import Visualizer
from models.Unet_generator import UNet
from data.dataset import train_data
from data.data_loader import load_data


'''
def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)
'''

#opt = TrainOptions().parse()
#config = get_config(opt.config)
#data_loader = CreateDataLoader(opt)
dataset = load_data(train_data)
#dataset_size = len(dataset)
#print('#training images = %d' % dataset_size)

#model = Unet()
#visualizer = Visualizer(opt)

total_steps = 0

# niter 训练回数的总数， 默认100
niter = 100

# niter_decay 以将学习率线性降低为零的数， 默认100
niter_decay = 100

batchSize = 32

# frequency of showing training results on console
print_freq = 100

# frequency of saving the latest results
save_latest_freq = 5000

# frequency of saving checkpoints at the end of epochs
save_epoch_freq = 5

# new learning rate
new_lr = True
'''
for epoch in range(1, niter + niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += batchSize #32
        #epoch_iter = total_steps - dataset_size * (epoch - 1) #32 - 29(1 - 1)
        model.set_input(data)
        model.optimize_parameters(epoch)

        #if total_steps % opt.display_freq == 0:
            #visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % print_freq == 0:
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / batchSize
            #visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #if opt.display_id > 0:
                #visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, total_steps))
            model.save('latest')

    if epoch % save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
        (epoch, niter + niter_decay, time.time() - epoch_start_time))

    if new_lr:
        if epoch == niter:
            model.update_learning_rate()
        elif epoch == (niter + 20):
            model.update_learning_rate()
        elif epoch == (niter + 70):
            model.update_learning_rate()
        elif epoch == (niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > niter:
            model.update_learning_rate()
'''