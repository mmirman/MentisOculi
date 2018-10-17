import torch as torch
import os
import numbers

if torch.cuda.is_available() and not 'NOCUDA' in os.environ:
    print("using cuda")
    cuda_async = True
    device = torch.device("cuda")
    use_cuda = True

    dtype = lambda *args, **kargs: torch.cuda.FloatTensor(*args, **kargs).cuda(async=cuda_async)
    ltype = lambda *args, **kargs: torch.cuda.LongTensor(*args, **kargs).cuda(async=cuda_async)
    btype = lambda *args, **kargs: torch.cuda.ByteTensor(*args, **kargs, device=device).cuda(async=cuda_async)
    ones = lambda *args, **cargs: torch.ones(*args, **cargs, device=device).cuda(async=cuda_async)
    ones_like = lambda *args, **cargs: torch.ones_like(*args, **cargs, device=device).cuda(async=cuda_async)
    zeros = lambda *args, **cargs: torch.zeros(*args, **cargs, device=device).cuda(async=cuda_async)
    eye = lambda *args, **cargs: torch.eye(*args, **cargs, device=device).cuda(async=cuda_async)
    rand = lambda *args, **cargs: torch.rand(*args, **cargs, device = device).cuda(async=cuda_async)

    linspace = lambda *args, **cargs: torch.linspace(*args, **cargs).cuda(async=cuda_async)

    print("set up cuda")
else:
    print("not using cuda")
    device = torch.device("cpu")
    dtype = lambda *args, **kargs: torch.FloatTensor(*args, **kargs)
    ltype = lambda *args, **kargs: torch.LongTensor(*args, **kargs)
    btype = lambda *args, **kargs: torch.ByteTensor(*args, **kargs)
    linspace = torch.linspace

    rand = torch.rand
    ones = torch.ones
    ones_like = torch.ones_like
    zeros = torch.zeros
    eye = torch.eye
    use_cuda = False

def cudify(x):
    if use_cuda:
        return x.cuda(async=True)
    return x

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return x[cond] 

class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __truediv__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)

    def __add__(self, other):
        if not isinstance(other, vec3):
            return vec3(self.x + other, self.y + other, self.z + other)
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = torch.sqrt(abs(self))
        return self * (1.0 / torch.where(mag == 0, dtype(1), mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(zeros(cond.shape), zeros(cond.shape), zeros(cond.shape))
        r.x[cond] = self.x
        r.y[cond] = self.y
        r.z[cond] = self.z
        return r


rgb = vec3
