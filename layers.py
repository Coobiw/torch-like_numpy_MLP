import numpy as np
from tensor import Tensor
from numpy.typing import NDArray as Ndarray

class Module:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,*args,**kwargs):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, n_in: int, n_out: int, initialize:str = 'kaiming', constant: float = None):
        super(Linear,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if initialize == 'kaiming' or 'xavier':
            self.weight = Tensor(initialize=initialize, shape=(n_in,n_out), n_in=n_in, n_out=n_out,requires_grad=True)
        elif initialize == 'constant':
            assert constant, 'constant initialize must be offered "constant" param '
            self.weight = Tensor(initialize='constant',shape=(n_in,n_out),constant=constant)
        else:
            self.weight = Tensor(initialize=initialize,shape=(n_in,n_out),requires_grad=True)
        self.bias = Tensor(initialize='zeros', shape=(1,n_out), requires_grad=True)

    def forward(self,x: Tensor): # x.shpae:(batch_size , n_in)
        # print(x.shape,self.weight.shape,self.bias.shape)
        out_value = (x.value @ self.weight.value) + self.bias.value
        out = Tensor(value=out_value)
        return out

    def backward(self,in_feat:Ndarray,in_grad:Ndarray):
        batch_size = in_feat.shape[0]
        self.bias.grad = in_grad.reshape(batch_size,1,-1)
        self.bias.grad = self.bias.grad.mean(axis=0)
        self.weight.grad = in_feat.reshape(batch_size,-1,1) @ in_grad.reshape(batch_size,1,-1)
        self.weight.grad = self.weight.grad.mean(axis=0)
        hidden_grad = in_grad @ self.weight.value.T
        return hidden_grad

class CrossEntropyLoss(Module): # 连带softmax
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()
        self.prob = None

    def forward(self,pred:Tensor,target:Ndarray):
        pred_value = pred.value
        prob_value = self.softmax(pred_value)
        self.prob = Tensor(value=prob_value)
        return self.nll(prob_value,target)

    def backward(self,target):
        self.prob.grad = self.prob.value - target
        return self.prob.grad

    @staticmethod
    def softmax(pred:Ndarray): # pred_shape : (batch_size,n_class)
        numerator = np.exp(pred)
        denominator = np.sum(numerator,axis=1).reshape(-1,1)
        return numerator/denominator

    @staticmethod
    def nll(prob,target):
        loss = -1 * np.log(prob) * target
        loss = loss.sum(axis=1)
        loss = loss.mean()
        return loss

class ReLU:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,x:Tensor) -> Tensor:
        x_value = x.value
        out = x_value.copy()
        out[out < 0] = 0
        out = Tensor(value = out)
        return out

    @staticmethod
    def derivative(x:Ndarray) -> Ndarray:
        out = x.copy()
        out[out <= 0] = 0.
        out[out > 0] = 1.
        return out


