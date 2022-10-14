import numpy as np
from typing import Tuple,Optional
import math

class Tensor(object):
    def __init__(self,value = None, initialize: str = 'zeros', shape:Optional[Tuple] = None,
                 constant: int = None, n_in: int = None, n_out: int = None, requires_grad = True):
        '''
        :param value: value of the tensor
        :param initialize: the initialization setup
        :param shape: the shape of the tensor
        :param constant: if 'initialize' = constant, this param should be set
        :param n_in: if 'initialize' = xavier or kaiming, this param should be set
        :param n_out: if 'initialize' = xavier or kaiming, this param should be set
        :param requires_grad: whether the tensor need gradient
        '''
        self.grad = None
        if not (value is None):
            self.value = np.array(value,dtype = np.float)
            self._shape = self.value.shape

        else:
            assert shape, "can't initialize the Tensor without shape info and value info"
            self._shape = shape
            if initialize == 'zeros':
                self.value = np.zeros(shape,dtype=np.float)

            if initialize == 'constant':
                assert constant, "constant initialization must be offered the constant nunber"
                self.value = np.ones(shape,dtype=np.float) * constant

            if initialize == 'normal':
                self.value = np.random.randn(*shape)

            if initialize == 'xavier':
                self.value = np.random.normal(loc = 0,scale = math.sqrt(2/(n_in + n_out)),size=shape)

            if initialize == 'kaiming':
                self.value = np.random.normal(loc=0,scale = math.sqrt(4/(n_in+n_out)),size=shape)

    @property
    def shape(self):
        return self._shape