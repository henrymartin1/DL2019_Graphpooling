from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import numpy as np


class Visualizer():

    def __init__(self, log_dir):

        self.summary_writer = SummaryWriter(log_dir=log_dir)
    
    def write_lr(self, optim, globaliter):
        for i, param_group in enumerate(optim.param_groups):
            self.summary_writer.add_scalar('learning_rate/lr_'+ str(i) , param_group['lr'], globaliter)
        self.summary_writer.flush()
    
    def write_loss_train(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/train', value, globaliter)
        self.summary_writer.flush()

    def write_loss_test(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/test', value, globaliter)
        self.summary_writer.flush()

    def write_loss_validation(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/validation', value, globaliter)
        self.summary_writer.flush()
    
    def write_acc_train(self, value, globaliter):
        self.summary_writer.add_scalar('acc/train', value, globaliter)
        self.summary_writer.flush()

    def write_acc_test(self, value, globaliter):
        self.summary_writer.add_scalar('acc/test', value, globaliter)
        self.summary_writer.flush()

    def write_acc_validation(self, value, globaliter):
        self.summary_writer.add_scalar('acc/validation', value, globaliter)
        self.summary_writer.flush()

    def write_adj_matrix(self, image, globaliter):
        image += image.min()
        image *= 255/image.max()
        self.summary_writer.add_image('adj', image, globaliter)

    def write_hist(self, image, globaliter):
         self.summary_writer.add_histogram('hist', image, globaliter)
  


    def close(self):
        self.summary_writer.close()

if __name__ == "__main__":


    import numpy as np 
    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16
        print(img_batch[i, 0].shape)

        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16
    
    print(img_batch.shape)