import numpy as np


class Tensor:
    value = None
    shape = None

    def __init__(self, data=None):
        self.value = np.array(data)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):

        if self.iter+1 < self.shape()[0]:
            self.iter += 1
        return self.value[self.iter]

    def __add__(self, tensor):
        return Tensor(np.add(self.value, tensor.value))

    def __mul__(self, tensor):
        return Tensor(np.multiply(self.value, tensor.value))

    def reshape(self, shape):
        self.value = np.reshape(self.value, shape)
        return self

    def shape(self):
        return self.value.shape

    @staticmethod
    def zero(shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape))



if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([[7, 8, 9], [3, 4, 5]])
    c = Tensor.ones([2, 2])
    d = a + b
    e = a.reshape([3, 2])
    for i in a:
        print(i)