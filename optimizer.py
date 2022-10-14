import numpy as np
from typing import List
from tensor import Tensor

class Optimizer:
    def __init__(self,param_list: List[Tensor], lr:float):
        self.param_list = param_list
        self.lr = lr

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,params_list,lr):
        super(SGD, self).__init__(params_list,lr)

    def step(self):
        for i in range(len(self.param_list)):
            self.param_list[i].weight.value = self.param_list[i].weight.value - self.param_list[i].weight.grad * self.lr
            self.param_list[i].bias.value = self.param_list[i].bias.value - self.param_list[i].bias.grad * self.lr
            # 释放梯度信息
            self.param_list[i].weight.grad = None
            self.param_list[i].bias.grad = None