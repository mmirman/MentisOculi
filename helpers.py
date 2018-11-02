"""
   Copyright 2018 Matthew Mirman

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import torch as torch
import os
import numbers
import math

import pdb

def pdbAssert(cond):
    if not cond:
        pdb.set_trace()


if torch.cuda.is_available() and not 'NOCUDA' in os.environ:
    print("using cuda")
    cuda_async = True
    device = torch.device("cuda")
    use_cuda = True

    dtype = lambda *args, **kargs: torch.cuda.DoubleTensor(*args, **kargs).cuda(async=cuda_async)
    ltype = lambda *args, **kargs: torch.cuda.LongTensor(*args, **kargs).cuda(async=cuda_async)
    btype = lambda *args, **kargs: torch.cuda.ByteTensor(*args, **kargs, device=device).cuda(async=cuda_async)
    ones = lambda *args, **cargs: torch.ones(*args, **cargs, device=device, dtype=torch.double).cuda(async=cuda_async)
    lones = lambda *args, **cargs: torch.ones(*args, **cargs, device=device).cuda(async=cuda_async)
    ones_like = lambda *args, **cargs: torch.ones_like(*args, **cargs, device=device, dtype=torch.double).cuda(async=cuda_async)
    zeros = lambda *args, **cargs: torch.zeros(*args, **cargs, device=device, dtype=torch.double).cuda(async=cuda_async)
    lzeros = lambda *args, **cargs: torch.zeros(*args, **cargs, device=device).cuda(async=cuda_async)

    eye = lambda *args, **cargs: torch.eye(*args, **cargs, device=device, dtype=torch.double).cuda(async=cuda_async)
    rand = lambda *args, **cargs: torch.rand(*args, **cargs, device = device, dtype=torch.double).cuda(async=cuda_async)
    randn = lambda *args, **cargs: torch.randn(*args, **cargs, device = device, dtype=torch.double).cuda(async=cuda_async)

    linspace = lambda *args, **cargs: torch.linspace(*args, **cargs, dtype=torch.double).cuda(async=cuda_async)

    print("set up cuda")
else:
    print("not using cuda")
    device = torch.device("cpu")
    dtype = lambda *args, **kargs: torch.DoubleTensor(*args, **kargs)
    ltype = lambda *args, **kargs: torch.LongTensor(*args, **kargs)
    btype = lambda *args, **kargs: torch.ByteTensor(*args, **kargs)
    linspace = lambda *args, **cargs: torch.linspace(*args, **cargs, dtype=torch.double)

    rand = lambda *args, **cargs: torch.rand(*args, **cargs, device = device, dtype=torch.double)
    randn = lambda *args, **cargs: torch.randn(*args, **cargs, device = device, dtype=torch.double)
    ones = lambda *args, **cargs: torch.ones(*args, **cargs, device = device, dtype=torch.double) 
    lones = lambda *args, **cargs: torch.ones(*args, **cargs, device = device) 
    ones_like = lambda *args, **cargs: torch.ones_like(*args, **cargs, device = device, dtype=torch.double) 
    zeros = lambda *args, **cargs: torch.zeros(*args, **cargs, device = device, dtype=torch.double) 
    lzeros = lambda *args, **cargs: torch.zeros(*args, **cargs, device = device) 
    eye = lambda *args, **cargs: torch.eye(*args, **cargs, device = device, dtype=torch.double) 
    use_cuda = False

ub_zeros = lambda *args,**kargs: lzeros(*args,**kargs, dtype=torch.uint8)
b_zeros = lambda *args,**kargs: lzeros(*args,**kargs, dtype=torch.int8)
l_zeros = lambda *args,**kargs: lzeros(*args,**kargs, dtype=torch.int32)

def cudify(x):
    if use_cuda:
        return x.cuda(async=True)
    return x

def place(cond, x):
    #pdbAssert(product(x.shape) == int(cond.sum(dtype=torch.long)))
    r = cond.new_zeros(size = cond.shape) # TODO:  looks like this is bugging out nondeterministically!
    r[cond] = x
    #pdbAssert(int(r.sum(dtype=torch.long)) == int(x.sum(dtype=torch.long)))
    return r

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return x[cond] 

def product(it):
    if isinstance(it,int):
        return it
    product = 1
    for x in it:
        if x >= 0:
            product *= x
    return product

def max_shape(l,r):
    return [ max(x,y) for x,y in zip(l,r) ]

class vec3(object):
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        if isinstance(other, vec3):
            return vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        if isinstance(other, vec3):
            return vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
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
    def rgbNorm(self):
        return self * (1 / (self.x + self.y + self.z))
    
    def cross(self, other):
        ax,ay,az = self.x, self.y, self.z
        bx,by,bz = other.x, other.y, other.z
        return vec3(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    def norm(self):
        l = abs(self)
        mag = torch.sqrt(l) if isinstance(l, torch.Tensor) else math.sqrt(l)
        if isinstance(mag, torch.Tensor):
            mag =  torch.where(mag == 0, dtype(1), mag.double())
        else:
            mag = 1 if mag == 0 else mag
        return self * (1.0 / mag)

    def luminance(self):
        return (self.x + self.y + self.z) / 3

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

    def div_or(self, b, alt):
        return vec3(one_or_div(self.x, b, alt.x), one_or_div(self.y, b, alt.y), one_or_div(self.z, b, alt.z))
rgb = vec3


def one_or_div(a,b, o = None):
    if isinstance(b, numbers.Number):
        return a / b if b > 0 else 1
    gtz = b > 0

    if o is None:
        o = ones(b.shape)
    return torch.where(gtz, a / b , o)


def vec3u(x,s):
    return vec3(ones(s),ones(s),ones(s)) * x

def vec3uCPU(x,s):
    return vec3(ones(s).cpu(),ones(s).cpu(),ones(s).cpu()) * x
