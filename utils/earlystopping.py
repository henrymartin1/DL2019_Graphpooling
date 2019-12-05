
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logdir, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.log_dir = logdir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc_val_max = 0
        self.delta = delta

    def __call__(self, acc_val, model):

        score = acc_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc_val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc_val, model)
            self.counter = 0

    def save_checkpoint(self, acc_val, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.acc_val_max:.2f} --> {acc_val:.2f}).  Saving model ...')
        torch.save(model.state_dict(), self.log_dir + '/checkpoint.pt')
        self.acc_val_max = acc_val