import torch as torch
import os

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
