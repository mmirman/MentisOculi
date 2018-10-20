from PIL import Image
import numpy as np
import torch as tr
from helpers import *
import os
import time
import math
import itertools
import pdb

from functools import reduce

def pdbAssert(cond):
    if not cond:
        pdb.set_trace()

def save_img(args, color, nm):
    file_nm = os.path.join(args.SAVE_DIR,nm)
    print("\tsaving:", file_nm)
    color = color * 8
    rgb = [Image.fromarray(np.array(c.clamp(0, 1).reshape((args.h, args.w)).float() * 255), "F").resize((args.WIDTH, args.HEIGHT), Image.ANTIALIAS).convert("L") for c in color.components()]
    Image.merge("RGB", rgb).save(file_nm)


def random_spherical(u, v):
    theta  = u * 2 * math.pi
    phi = v * math.pi 

    # Switch to cartesian coordinates
    sphi = phi.sin()
    x = theta.cos() * sphi
    y = theta.sin() * sphi

    return vec3(x, y, phi.cos())

class Sphere:
    def __init__(self, center, r, diffuse, mirror = None):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, args, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = tr.sqrt(tr.relu(disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = tr.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > args.NEAREST)
        return tr.where(pred, h, ones_like(h) * args.FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def sampleDiffuse(self, args, getRand, M, N, newO, bounce):
        rayDiff = random_spherical(getRand(), getRand())
        should_flip = N.dot(rayDiff).lt(0).double()
        rayDiff = rayDiff * (1 - 2 * should_flip)

        return raytrace(args, getRand, newO, rayDiff , bounce + 2) * rayDiff.dot(N) * self.diffusecolor(M) * 2 

    def sampleMirror(self, args, getRand, D, N, newO, bounce):
        rayRefl = (D - N * 2 * D.dot(N)).norm()  # reflection            
        return raytrace(args, getRand, newO, rayRefl, bounce + 1) * self.mirror * rayRefl.dot(N)        


    def light(self, args, getRand, O, D, d, bounce):
        # D is direction
        # O is previous origin
        M = (O + D * d)                         # new intersection point
        N = (M - self.c) / self.r        # normal

        toO = (O - M).norm()                    # direction to ray origin
        newO = M + N * args.NUDGE               # M nudged to avoid itself

        if self.mirror is not None:
            diffcol = self.diffusecolor(M)
            refl_prob = self.mirror / (self.mirror + diffcol.luminance()) if isinstance(self.mirror, numbers.Number) else self.mirror.luminance()
            reflect = getRand() <= refl_prob
            diffuse = 1 - reflect
            
            colorDiff = self.sampleDiffuse(args, getNewRand(getRand, diffuse, 0), M.extract(diffuse), N.extract(diffuse), newO.extract(diffuse), bounce) * (1 / (1 - refl_prob)) if diffuse.any() else rgb(0,0,0)

            colorRefl = self.sampleMirror(args, getNewRand(getRand, reflect, 1), D.extract(reflect), N.extract(reflect), newO.extract(reflect), bounce) * (1 / refl_prob) if reflect.any() else rgb(0,0,0)

            color = colorDiff.place(diffuse) + colorRefl.place(reflect)
        else:
            color = self.sampleDiffuse(args, getRand, M, N, newO, bounce)
        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).int() % 2) == ((M.z * 2).int() % 2)
        return self.diffuse * checker.double() + rgb(0.8, 0.6, 0.6) * (1 - checker.double())

class Light(Sphere):
    def light(self, *args, **kargs):
        return self.diffuse
  

def raytrace(args, getRand, O, D, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays
    color = rgb(0, 0, 0)
    if bounce > args.MAX_BOUNCE:
        return color

    distances = [dtype(s.intersect(args, O, D)) for s in args.scene]
    nearest, nearest_idx = tr.min(torch.stack(distances), dim=0)

    for (s, i) in zip(args.scene, range(len(args.scene))):
        hit = (nearest < args.FARAWAY) & (nearest_idx == i) & (nearest > args.NUDGE) # d == nearest is hacky af
        probStop = args.STOP_PROB if bounce >= 1 else 0
        rd = getRand()
        pdbAssert( rd.shape == hit.shape)
        rgp = (rd >= probStop)

        hit = hit & rgp

        if hit.any():
            Oc = O.extract(hit)
            dc = extract(hit, nearest)
            Dc = D.extract(hit)
            cc = s.light(args, getNewRand(getRand, hit, i), Oc, Dc, dc, bounce)
            color += cc.place(hit) / (1 - probStop)

    return color


def getNewRand(getRand, mask, curr_idx):
    pdbAssert( (mask <= 1).all() )
    pdbAssert( (mask >= 0).all() )

    if mask.all():
        return getRand
    mshape = [int(mask.sum(dtype=torch.long))]
    def newRand(arg = None):
        if arg is None:
            arg = (mshape, mask, [curr_idx])
        else:
            (sN, hitN, sub_idx) = arg
            pdbAssert( (hitN <= 1).all() )
            pdbAssert( (hitN >= 0).all() )
            arg = (sN, place(mask, hitN), [curr_idx] + sub_idx)
        return getRand(arg)
    return newRand

def circ(a):
    a.sub_(a.floor())

def getMCRand(top_shape):
    def getRand(arg = None):
        if arg is None:
            mask = lones(top_shape, dtype=torch.uint8)
            maskShape = top_shape
            idx = []
        else:
            maskShape,mask, idx = arg
        return rand(size = maskShape)
    return getRand

def getPermuteRand(top_shape, mcmc_best):
    mcmc_generator = {}
    num_calls = {}

    for k,v in mcmc_best.items():  # save old random values for when new things get mixed in
        mcmc_generator[k] = v
    
    def getRand(arg = None):
            if arg is None:
                mask = lones(top_shape, dtype=torch.uint8)
                maskShape = top_shape
                idx = []
            else:
                maskShape, mask, idx = arg
            tidx = tuple(idx)    
            
            if tidx not in num_calls:
                num_calls[tidx] = 0
            else:
                num_calls[tidx] += 1
            tidx = tuple(idx + [num_calls[tidx]])

            if tidx not in mcmc_best:
                r = rand(size = maskShape)
                pdbAssert(product(r.shape) == product(maskShape))
            else: 
                # could be done way quicker in handwritten cuda.
                # sadly, pseudorandoms are slow enough that we want to do as few of them as possible.
                
                # can theoretically optimize this a bit here by using a subset of top_shape corresponding to the max in mask
                
                bestIndxs, bestRand = mcmc_best[tidx]

                bestMask = lzeros(top_shape, dtype=torch.uint8)
                bestMask[bestIndxs] = 1
                
                relevantBestMask = bestMask & mask 
                
                newRands = zeros(top_shape) # if these are different sizes then something went very significantly wrong
                pdbAssert(bestMask.sum() == product(bestRand.shape))
                newRands[bestMask] = torch.normal(mean = bestRand, std = 0.003)
                newRands[(1 - bestMask) & mask] = rand(size = newRands[(1 - bestMask) & mask].shape)

                r = newRands[mask]
                pdbAssert(product(r.shape) == product(maskShape))
                circ(r)
                pdbAssert(product(r.shape) == product(maskShape))
            mcmc_generator[tidx] = (mask.nonzero().cpu(),r)
            return r
    
    return getRand, mcmc_generator

def mixSamples(top_shape, mix, sa, sb):
    res = {}
            
    for k in set().union(sa.keys(), sb.keys()):
        if k not in sa.keys():
            res[k] = sb[k]
        elif k not in sb.keys():
            res[k] = sa[k]
        else:
            aI, aR = sa[k]
            pdbAssert(len(aR.shape) == 1)
            pdbAssert(product(aI.shape) == product(aR.shape))
            
            bI, bR = sb[k]
            pdbAssert(len(bR.shape) == 1)
            pdbAssert(product(bI.shape) == product(bR.shape))

            aM = lzeros(top_shape, dtype=torch.uint8)
            bM = lzeros(top_shape, dtype=torch.uint8)
            
            aM[aI] = 1
            bM[bI] = 1

            aRes = zeros(top_shape)
            bRes = zeros(top_shape)

            aRes[aM] = aR
            bRes[bM] = bR
            
            abM = aM | bM

            # be wary of what happens when mixing something in which was not there before!
            res[k] = (abM.nonzero().cpu(), (aRes * mix + bRes * (1 - mix))[abM])
    return res

def shoot(args, getRand, S, pixels):
    x_sz = (S[2] - S[0]) / args.w
    y_sz = (S[3] - S[1]) / args.h
    
    pixel_mod = pixels + vec3(getRand() * x_sz, getRand() * y_sz, 0)    
    return raytrace(args, getRand, args.eye, (pixel_mod - args.eye).norm(), bounce = 0) 


def multiSamp(args, samp_shape, samp_cast, num_mc_samples):
    total_time = 0
    estimate = vec3u(0,samp_shape)
    for i in range(1,num_mc_samples + 1):
        tPass = time.time()

        mcRand = getMCRand(samp_shape)
        new_estimate = raytrace(args, mcRand, args.eye, (samp_cast - args.eye).norm(), bounce = 0) 
        estimate = (new_estimate / float(num_mc_samples))  + estimate

        tCurr = time.time()
        pass_time = tCurr - tPass
        total_time += pass_time

        print("\nMCPass:", i)
        print("\tElapsed Time:", total_time)
        print("\tPass Time:", pass_time)
        print("\tAvg Pass Time:",  total_time / i)

        print("\tTotal Samples:", args.w * args.h * i)
        print("\tSamples Per Pixel:", args.OVERSAMPLE * i)

        print("\tsamp/sec:", (args.w * args.h) / pass_time )
        print("\tAvg samp/sec:",  (args.w * args.h * i) / total_time, "\n")

    return estimate

def mctrace(args, S, pixels, k):
    top_shape = pixels.x.shape
    img = 0
    total_time = 0
    for i in range(1,k+1):
        tPass = time.time()
        getRand = getMCRand(top_shape)
        img = shoot(args, getRand, S, pixels) + img
        
        tCurr = time.time()
        pass_time = tCurr - tPass
        total_time += pass_time

        print("\nMCPass:", i)
        print("\tElapsed Time:", total_time)
        print("\tPass Time:", pass_time)
        print("\tAvg Pass Time:",  total_time / i)

        print("\tTotal Samples:", args.w * args.h * i)
        print("\tSamples Per Pixel:", args.OVERSAMPLE * i)

        print("\tsamp/sec:", (args.w * args.h) / pass_time )
        print("\tAvg samp/sec:",  (args.w * args.h * i) / total_time, "\n")

    img /= k
    return img

def one_or_div(a,b, o = 1):
    if isinstance(b, numbers.Number):
        return a / b if b > 0 else 1
    gtz = b > 0
    return torch.where(gtz, a / torch.where(gtz, b, ones(b.shape) * o) , ones(b.shape) * o)

def pathtrace(args, S, pixels):

    samp_shape = pixels.x.shape
    img_shape = pixels.x.shape

    img = vec3u(0, img_shape)

    total_time = 0
        

    x_sz = (S[2] - S[0])
    y_sz = (S[3] - S[1])


    m = 0
    k = 0
    mcestim = vec3u(0, img_shape)
    for i in itertools.count(1,1):
        restart = i % args.restart_freq == 1     
        if restart:
            best_sample = vec3u(0, samp_shape)
            best_sample_params = {}

        getRand, new_sample_params = getPermuteRand(samp_shape, best_sample_params)
        samp_coords = vec3(getRand(), getRand(), 0)

        samp_cast = vec3(S[0], S[1], 0) + samp_coords * vec3(x_sz, y_sz, 0)

        if restart:
            k += 1
            best_samp_coords = samp_coords
            best_sample_params = new_sample_params

            estimate = multiSamp(args, samp_shape, samp_cast, args.num_mc_samples)

            im_locs = best_samp_coords * vec3(args.w, args.h, 0)
            im_locs = [im_locs.y.long(), im_locs.x.long()]
            estim = vec3u(0, img_shape)
            estim.x.reshape(args.h, args.w)[im_locs] += estimate.x
            estim.y.reshape(args.h, args.w)[im_locs] += estimate.y
            estim.z.reshape(args.h, args.w)[im_locs] += estimate.z

            mcestim = estim + mcestim
            save_img(args, mcestim / k, "estimate"+str(k)+".png")
            continue
        m += 1

        tPass = time.time()

        new_samp = raytrace(args, getRand, args.eye, (samp_cast - args.eye).norm(), bounce = 0) 

        accept_var = rand(samp_shape)
        accept_prob = one_or_div(new_samp.luminance(), best_sample.luminance())
        accept_prob.clamp_(0,1)

        def addS(s, p):
            im_locs = s * vec3(args.w, args.h, 0)
            im_locs = [im_locs.y.long(), im_locs.x.long()]

            img.x.reshape(args.h, args.w)[im_locs] += p.x
            img.y.reshape(args.h, args.w)[im_locs] += p.y
            img.z.reshape(args.h, args.w)[im_locs] += p.z
        addS(best_samp_coords, (best_sample * estimate.luminance()).div_or(best_sample.luminance(), estimate) * (1 - accept_prob) )
        addS(samp_coords, (new_samp * estimate.luminance()).div_or(new_samp.luminance(), estimate) * accept_prob)

        should_accept = (accept_var <= accept_prob).double()
        best_sample_params = mixSamples(samp_shape, should_accept, new_sample_params, best_sample_params)
        best_sample = new_samp * should_accept + best_sample * (1 - should_accept)
        best_samp_coords = samp_coords * should_accept + best_samp_coords * (1 - should_accept)


        tCurr = time.time()
        pass_time = tCurr - tPass
        total_time += pass_time

        print("\n\nPass:", i)
        print("\tElapsed Time:", total_time)
        print("\tPass Time:", pass_time)
        print("\tAvg Pass Time:",  total_time / i)

        print("\n\tTotal Samples:", args.w * args.h * i)
        print("\tSamples Per Pixel:", args.OVERSAMPLE * i)

        print("\n\tsamp/sec:", (args.w * args.h) / pass_time )
        print("\tAvg samp/sec:",  (args.w * args.h * i) / total_time, "\n")

        save_img(args, img / m, "img"+str(i)+".png")
        save_img(args, img / m, "img.png")


def render(args):

    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)

    args.w = args.WIDTH * args.OVERSAMPLE
    args.h = args.HEIGHT * args.OVERSAMPLE

    r = float(args.WIDTH) / args.HEIGHT
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    x =  linspace(S[0], S[2], args.w).repeat(args.h)
    y = linspace(S[1], S[3], args.h).view(-1,1).expand(args.h,args.w).contiguous().view(-1)

    Q = vec3(x, y, 0)

    pathtrace(args, S, Q)


class StaticArgs:
    SAVE_DIR="out_erpt_red_save"
    OVERSAMPLE = 4
    WIDTH = 400
    HEIGHT = 300

    scene = [
        Light(vec3(5, 2, 1.2), 2.0, rgb(1, 1, 1)),
        Sphere(vec3(0, 205, 1), 197, rgb(0.99, 0.96, 0.99)),
        Sphere(vec3(.3, .1, 1.3), .6, rgb(0.1, 0.1, 0), rgb(0.5, 0.95, 1)),
        Sphere(vec3(-.4, .2, 0.8), .4, rgb(1, .8, .9).rgbNorm() * 3 * 0.4, 0.7),
        CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.96, .99, .99)),
    ]

    eye = vec3(0., 0.35, -1.)     # Eye position
    FARAWAY = 1.0e36            # an implausibly huge distance
    MAX_BOUNCE = 12
    NUDGE = 0.0000001
    STOP_PROB = 0.8

    NEAREST = 0.000000001
    restart_freq = 30
    num_mc_samples = 3

render(StaticArgs)
