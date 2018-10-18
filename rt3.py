from PIL import Image
import numpy as np
import torch as tr
from helpers import *
import os
import time
import math
import itertools

from functools import reduce

def save_img(args, color, nm):
    file_nm = os.path.join(args.SAVE_DIR,nm)
    print("\tsaving:", file_nm)
    color = color * 20
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
            refl_prob = self.mirror / (self.mirror + diffcol.luminance())if isinstance(self.mirror, numbers.Number) else self.mirror.luminance()
            reflect = getRand() <= refl_prob
            diffuse = 1 - reflect
            
            
            colorDiff = self.sampleDiffuse(args, getNewRand(getRand, diffuse, 0), M.extract(diffuse), N.extract(diffuse), newO.extract(diffuse), bounce) * (1 / refl_prob) if tr.sum(diffuse) > 0 else rgb(0,0,0)

            colorRefl = self.sampleMirror(args, getNewRand(getRand, reflect, 1), D.extract(reflect), N.extract(reflect), newO.extract(reflect), bounce) * (1 / (1 - refl_prob)) if tr.sum(reflect) > 0 else rgb(0,0,0)

            color = colorDiff.place(diffuse) + colorRefl.place(reflect)
        else:
            color = self.sampleDiffuse(args, getRand, M, N, newO, bounce)

        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).int() % 2) == ((M.z * 2).int() % 2)
        return self.diffuse * checker.double() + rgb(0.8, 0.7, 0.7) * (1 - checker.double())

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
    nearest = reduce(tr.min, distances)

    for (s, d, i) in zip(args.scene, distances, range(len(args.scene))):
        hit = (nearest < args.FARAWAY) & (d == nearest) & (d > args.NUDGE) # d == nearest is hacky af
        probStop = args.STOP_PROB if bounce >= 1 else 0
        hit = hit & (getRand() >= probStop)

        if tr.sum(hit).data > 0:
            Oc = O.extract(hit)
            dc = extract(hit, d)
            Dc = D.extract(hit)
            cc = s.light(args, getNewRand(getRand, hit, i), Oc, Dc, dc, bounce)
            color += cc.place(hit) / (1 - probStop)
    return color



def getNewRand(getRand, mask, curr_idx):
    if mask.all():
        return getRand
    def newRand(shape = None):
        if shape is None:
            shape = (mask[mask].shape, mask, [curr_idx])
        else:
            (sN, hitN, sub_idx) = shape
            shape = (sN, place(mask, hitN), [curr_idx] + sub_idx)
        return getRand(shape)
    return newRand

def getOriginalRand(pshape, mcmc_best):
    mcmc_generator = {}
    
    def getRand(shape = None):
            if shape is None:
                mask = lones(pshape, dtype=torch.uint8)
                shape = pshape
                idx = []
            else:
                shape,mask, idx = shape

            if tuple(idx) not in mcmc_best:
                r = rand(shape)
            else:
                bestShape, bestIndxs, bestRand = mcmc_best[tuple(idx)]
                
                #idxs = torch.sparse_coo_tensor(bestIndxs.cuda().long(), cudify(bestRand))
                
                bestMask = lzeros(bestShape, dtype=torch.uint8)
                bestMask[bestIndxs] = 1
                
                relevantBestMask = bestMask & mask 
                
                newRands = zeros(max_shape(bestShape, shape)) # if these are different sizes then something went very significantly wrong

                newRands[bestMask] = bestRand 
                newRands[relevantBestMask] += rand(newRands[relevantBestMask].shape)
                newRands[relevantBestMask].clamp_(0,1)
                newRands[(1 - bestMask) & mask] = rand(newRands[(1 - bestMask) & mask].shape)
                
                r = newRands[mask]

            mcmc_generator[tuple(idx)] = (mask.shape, mask.nonzero().cpu(),r)
            return r

    return getRand, mcmc_generator

def pathtrace(args, S, pixels):

    img = 0

    x_sz = (S[2] - S[0]) / args.w
    y_sz = (S[3] - S[1]) / args.h

    mcmc_best = {}
    total_time = 0

    for i in itertools.count(1,1):
        if i % 20 == 0:
            mcmc_best = []

        tPass = time.time()
        
        getRand, mcmc_generator = getOriginalRand(pixels.x.shape, mcmc_best)
        mcmc_best = mcmc_generator
        
        pixel_mod = pixels + vec3(getRand() * x_sz, getRand() * y_sz, 0)
        sub_img = raytrace(args, getRand, args.eye, (pixel_mod - args.eye).norm(), bounce = 0) 

        img = sub_img + img

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

        save_img(args, sub_img, "sub_img"+str(i)+".png")
        save_img(args, img / (i + 1), "img"+str(i)+".png")
        save_img(args, img / (i + 1), "img.png")


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

    color = pathtrace(args, S, Q)

    save_img(color, "img.png")


class StaticArgs:
    SAVE_DIR="out_met"
    OVERSAMPLE = 4
    WIDTH = 400
    HEIGHT = 300

    scene = [
        Light(vec3(5, 2, 1.2), 2.0, rgb(1, 1, 1)),
        Sphere(vec3(0, 205, 1), 197, rgb(0.99, 0.99, 0.99)),
        Sphere(vec3(.3, .1, 1.3), .6, rgb(0.1, 0.1, 0), rgb(0.5, 0.95, 1)),
        Sphere(vec3(-.4, .2, 0.8), .4, rgb(1, .8, .9).rgbNorm() * 3 * 0.4, 0.7),
        CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.95, .95, .95)),
    ]

    eye = vec3(0., 0.35, -1.)     # Eye position
    FARAWAY = 1.0e36            # an implausibly huge distance
    MAX_BOUNCE = 6
    NUDGE = 0.0000001
    STOP_PROB = 0.7

    NEAREST = 0.000000001

render(StaticArgs)
