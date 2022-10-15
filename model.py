import numpy as np
from layers import Linear,CrossEntropyLoss,Module,ReLU
from tensor import Tensor
from typing import List

class MLP(Module):
    def __init__(self, n_layers: int = 2, n_class: int = 10, dim_list:List[int] = [28*28,512]):
        dim_list = dim_list + [n_class]
        assert n_layers == len(dim_list)-1, 'the length of dim_list param should be equal to n_layers'
        self.layers = []
        for i in range(n_layers):
            self.layers.append(Linear(dim_list[i],dim_list[i+1]))

        self.layers.append(CrossEntropyLoss())
        self.relu = ReLU()
        self.input_list = []
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x: Tensor, target):
        self.input_list.append(x)
        for index, layer in enumerate(self.layers):
            if not isinstance(layer,CrossEntropyLoss):
                x = layer(x)
                if index+2 != len(self.layers):
                    self.input_list.append(x)
                    x = self.relu(x)
                    self.input_list.append(x)
            else:
                loss = layer(x,target)

        if not self.training:
            # 如果是测试模式，不用保存中间节点！
            self.input_list = []

        return x, loss

    def backward(self,target):
        assert self.training, 'only train mode can backward propagation'
        length = len(self.layers)
        input_list_index = -1
        for i in range(length-1,-1,-1):
            if i == length-1: # softmax + nll_loss backward
                in_grad = self.layers[i].backward(target)
            elif i == length-2: # linear layer backward without relu activation
                in_grad = self.layers[i].backward(in_feat=self.input_list[input_list_index].value,in_grad=in_grad)
                input_list_index -= 1
            else:
                in_grad = ReLU.derivative(self.input_list[input_list_index].value) * in_grad
                input_list_index -= 1
                in_grad = self.layers[i].backward(in_feat=self.input_list[input_list_index].value,in_grad=in_grad)
                input_list_index -= 1
        # 释放中间正向传播的计算变量
        self.input_list = []

    def state_dict(self):
        state_dict = {}
        for i,layer in enumerate(self.layers):
            if not isinstance(layer,CrossEntropyLoss):
                name = f'linear_{i+1}'
                state_dict[name+'_weight'] = layer.weight.value
                state_dict[name+'_bias'] = layer.bias.value
        return state_dict

    def load_state_dict(self,state_dict):
        for i,layer in enumerate(self.layers):
            if not isinstance(layer,CrossEntropyLoss):
                name = f'linear_{i+1}'
                assert not state_dict.get(name+'_weight',None) is None,"the architecture of model doesn't match the state_dict"
                assert layer.weight.value.shape == state_dict[name+'_weight'].shape,"the layers of model doesn't match the state_dict"
                self.layers[i].weight = Tensor(value=state_dict[name+'_weight'])
                self.layers[i].bias = Tensor(value = state_dict[name+'_bias'])




if __name__ == '__main__':
    mlp = MLP()
    print(mlp.layers)
    x = np.zeros((2,28*28))
    target = np.ones((2,1))
    x = Tensor(value = x)
    a,b = mlp(x,target)
    print(len(mlp.input_list))
    print(a.value)
    print(b)
    mlp.backward(target)
    print(len(mlp.input_list))
    # from train import save_model
    # save_model(mlp,'test')
    # new_mlp = MLP()
    # import pickle
    # with open('test.qbw','rb') as f:
    #     state_dict = pickle.load(f)
    # new_mlp.load_state_dict(state_dict)


