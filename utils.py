import numpy as np
from numpy.typing import NDArray as Ndarray

class AverageMeter:
    def __init__(self):
        self.sum = 0.
        self.num = 0
        self.avg = 0.

    def update(self,val,n=1):
        self.sum += val * n
        self.num += n
        self.avg = self.sum / self.num

class Accuracy_averagemeter(AverageMeter):
    def update(self,val,n=1):
        self.sum += int(val)
        self.num += n
        self.avg = self.sum / self.num


def compute_acc(pred:Ndarray,target:Ndarray):
    pred_cls = pred.argmax(axis=1)
    target = target.argmax(axis=1) # 只有一位为1，其他位都是0
    acc_num = (pred_cls == target).sum()
    acc_num = int(acc_num)
    return acc_num

