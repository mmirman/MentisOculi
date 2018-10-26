from PIL import Image
import numpy as np
import torch as tr
from helpers import *
import os
import time
import math
import itertools

from functools import reduce

def mulF(a,m):
    a,b = a
    return a * m, b

def no_repeats(l):
    t = l.sort()[0]
    return not (t[1:] == t[:-1]).any()

def save_img(args, img, nm):
    file_nm = os.path.join(args.SAVE_DIR,nm)
    print("\tsaving:", file_nm)
    img = (img * 22.0 ).clamp(0,1) * 255
    img = img.transpose(0,2)
    
    rgb = [Image.fromarray(np.array(c), "F").resize((args.WIDTH, args.HEIGHT), Image.ANTIALIAS).convert("L") for c in img.float()]
    Image.merge("RGB", rgb).save(file_nm)


def getFirstOcc(locs, p):
    is_first = torch.cat([btype([1]), locs[1:] != locs[:-1]])
    is_repeat = 1 - is_first
    return locs[is_first], p[is_first], locs[is_repeat], p[is_repeat]

def addS(args, img, s, p):
    im_locs = s * vec3(args.w, args.h, 0)

    im_locs, im_locs_ind = torch.sort(im_locs.y.long() + im_locs.x.long() * args.h  )
    p = torch.stack([p.x, p.y, p.z], dim=1)[im_locs_ind]
    
    imger = img.reshape(args.h * args.w, 3)
    while product(p.shape) > 0:
        im_locs_cont, p_cont, im_locs, p = getFirstOcc(im_locs, p)
        imger[im_locs_cont] += p_cont


def new_img(args):
    img_shape  = [args.w, args.h, 3]
    return zeros(img_shape)

def random_spherical(u, v):
    theta  = u * 2 * math.pi
    phi = v * math.pi 

    # Switch to cartesian coordinates
    sphi = phi.sin()
    x = theta.cos() * sphi
    y = theta.sin() * sphi

    return vec3(x, y, phi.cos())

class Sphere:
    def __init__(self, center, r, diffuse, mirror = None, semi = None, semi_low = 0, semi_high = 1):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

        self.semi_low = semi_low
        self.semi_high = semi_high
        self.absCmR2 =  abs(self.c) - r * r
        self.inv_r = 1 / self.r
        self.semi = semi.norm() * self.inv_r if semi is not None else None
        self.c2 = self.c * 2
    def intersect(self, args, O, D):
        Omc = O - self.c
        b = D.dot(Omc)
        c = self.absCmR2 + abs(O) - self.c2.dot(O)
        disc = b * b - c
        sq = tr.sqrt(tr.relu(disc)) # can postpone the sqrt here for a speedup
        h0 = -b - sq  # dot(O - self.c) < r * r 
        h1 = -b + sq
        if self.semi is not None:
            n0 = (Omc + D * h0).dot(self.semi)
            n1 = (Omc + D * h1).dot(self.semi)

            h = tr.where((h0 > 0) & (n0 >= self.semi_low) & (n0 <= self.semi_high), h0, tr.where((n1 >= self.semi_low) & (n1 <= self.semi_high), h1, zeros(h1.shape) - 1) )
        else:
            h = tr.where(h0 > 0, h0, h1)
            
        pred = (disc > 0) & (h > args.NEAREST)

        return tr.where(pred, h, ones_like(h) * args.FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse, 0

    def sampleDiffuse(self, args, getRand, M, N, newO, bounce):
        rayDiff = random_spherical(getRand(), getRand())
        nrdiff = N.dot(rayDiff)
        
        sflip = 1 - 2 * nrdiff.lt(0).double()
        rayDiff = rayDiff * sflip
        nrdiff = nrdiff * sflip

        dm, did = self.diffusecolor(M)
        return raytrace(args, getRand, newO, rayDiff , bounce + 1.5)[0] * nrdiff * dm * 2 , did

    def sampleMirror(self, args, getRand, D, N, newO, bounce):
        rayRefl = (D - N * 2 * D.dot(N)).norm()  # reflection            
        col, mid = raytrace(args, getRand, newO, rayRefl, bounce + 0.5)
        return col * self.mirror, mid


    def light(self, args, getRand, O, D, d, bounce):
        # D is direction
        # O is previous origin
        M = O + D * d                         # new intersection point
        N = (M - self.c) * self.inv_r        # normal
        if self.semi is not None:
            N = N * (2 * (D.dot(N) < 0).double() - 1)

        toO = (O - M).norm()                    # direction to ray origin
        newO = M + N * args.NUDGE               # M nudged to avoid itself

        sid = l_zeros(D.x.shape)

        if self.mirror is not None:
            diffcol = self.diffusecolor(M)[0]
            refl_prob = self.mirror / (self.mirror + diffcol.luminance()) if isinstance(self.mirror, numbers.Number) else self.mirror.luminance()
            reflect = tri(getRand()) <= refl_prob
            diffuse = 1 - reflect
            
            colorDiff, did = mulF(self.sampleDiffuse(args, getNewRand(getRand, diffuse, 0), M.extract(diffuse), N.extract(diffuse), newO.extract(diffuse), bounce), 1 / (1 - refl_prob)) if diffuse.any() else (rgb(0,0,0), 0)

            colorRefl, mid = mulF(self.sampleMirror(args, getNewRand(getRand, reflect, 1), D.extract(reflect), N.extract(reflect), newO.extract(reflect), bounce), 1 / refl_prob) if reflect.any() else (rgb(0,0,0), 0)

            color = colorDiff.place(diffuse) + colorRefl.place(reflect)
            sid[diffuse] = did * 2
            sid[reflect] = mid * 2 + 1
        else:
            color, sid = self.sampleDiffuse(args, getRand, M, N, newO, bounce)
        return color, sid


class CheckeredSphere(Sphere):
    def __init__(self,*args, diffuse2 = vec3(0,0,0), **kargs):
        self.diffuse2= diffuse2
        super(CheckeredSphere, self).__init__(*args, **kargs)
    def diffusecolor(self, M):
        checker = (((M.x * 4).floor() + (M.z * 4).floor()).int()  % 2) > 0.5
        return self.diffuse * checker.double() + self.diffuse2 * (1 - checker.double()), checker.to(dtype=torch.int32)

class Light(Sphere):
    def light(self, *args, **kargs):
        return self.diffuse, 0
  

def raytrace(args, getRand, O, D, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays
    color = rgb(0, 0, 0)
    ids = l_zeros(D.x.shape)
    if bounce > args.MAX_BOUNCE:
        return color, ids

    distances = [dtype(s.intersect(args, O, D)) for s in args.scene]
    nearest, nearest_idx = tr.min(tr.stack(distances), dim=0)

    ls = len(args.scene)
    for (s, i) in zip(args.scene, range(ls)):
        hit = (nearest < args.FARAWAY) & (nearest_idx == i) & (nearest > args.NUDGE) # d == nearest is hacky af
        probStop = args.STOP_PROB if bounce >= 1 and not isinstance(s,Light) else 0
        rd = tri(getRand())
        rgp = (rd >= probStop)

        hit = hit & rgp

        if hit.any():
            Oc = O.extract(hit)
            dc = extract(hit, nearest)
            Dc = D.extract(hit)
            cc,sid = s.light(args, getNewRand(getRand, hit, i), Oc, Dc, dc, bounce)
            color += cc.place(hit) / (1 - probStop)

            ids[hit] = sid * (ls + 1) + i + 1
    return color, ids


def getNewRand(getRand, mask, curr_idx):
    if mask.all():
        return getRand
    mshape = [int(mask.sum(dtype=tr.long))]
    def newRand(arg = None):
        if arg is None:
            arg = (mshape, mask, [curr_idx])
        else:
            (sN, hitN, sub_idx) = arg
            maskN = place(mask, hitN)
            arg = (sN, maskN, [curr_idx] + sub_idx)
        return getRand(arg)
    return newRand

def getMCRand(top_shape):
    def getRand(arg = None):
        if arg is None:
            mask = lones(top_shape, dtype=tr.uint8)
            maskShape = top_shape
            idx = []
        else:
            maskShape,mask, idx = arg
        return rand(size = maskShape)
    return getRand

def getPermuteRand(should_jump, args, top_shape, mcmc_best):
    mcmc_generator = {}
    num_calls = {}

    for k,v in mcmc_best.items():  # save old random values for when new things get mixed in
        mcmc_generator[k] = v

    def getRand(arg = None):
            if arg is None:
                mask = lones(top_shape, dtype=tr.uint8)
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
            else: 
                bestIndxs, bestRand = mcmc_best[tidx]

                newRands = zeros(top_shape) # if these are different sizes then something went very significantly wrong
                newRands[mask] = rand(size = maskShape)

                newRands[cudify(bestIndxs)] = (cudify(bestRand) + randn(bestRand.shape) * args.jump_size) if should_jump else cudify(bestRand)
                
                r = newRands[mask]

            ids = mask.nonzero().squeeze(dim=1)
            mcmc_generator[tidx] = (ids.cpu(),r.cpu())
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
            
            bI, bR = sb[k]

            aM = lzeros(top_shape, dtype=tr.uint8)
            bM = lzeros(top_shape, dtype=tr.uint8)
            
            aM[cudify(aI)] = 1
            bM[cudify(bI)] = 1

            aRes = zeros(top_shape)
            bRes = zeros(top_shape)

            aRes[aM] = cudify(aR)
            bRes[bM] = cudify(bR)
            
            abM = aM | bM

            # be wary of what happens when mixing something in which was not there before!
            abMn = abM.nonzero().squeeze(dim=1)
            res[k] = (abMn.cpu(), (aRes * mix + bRes * (1 - mix))[abM].cpu())
    return res

def multiSamp(args, samp_shape, samp_cast, num_mc_samples):
    total_time = 0
    estimate = vec3u(0,samp_shape)
    samps_per_pass = product(samp_shape)
    for i in range(1,num_mc_samples + 1):
        tPass = time.time()

        mcRand = getMCRand(samp_shape)
        new_estimate = raytrace(args, mcRand, args.eye, (samp_cast - args.eye).norm(), bounce = 0)[0]
        estimate = (new_estimate / float(num_mc_samples))  + estimate

        tCurr = time.time()
        pass_time = tCurr - tPass
        total_time += pass_time

        print("\nMCPass:", i)
        print("\tElapsed Time:", total_time)
        print("\tPass Time:", pass_time)
        print("\tAvg Pass Time:",  total_time / i)

        print("\tTotal Samples:", samps_per_pass * i)
        print("\tSamples Per Pixel:", args.OVERSAMPLE * i)

        print("\tsamp/sec:", samps_per_pass / pass_time )
        print("\tAvg samp/sec:",  samps_per_pass * i / total_time, "\n")

    return estimate

def one_or_div(a,b, o = 1):
    if isinstance(b, numbers.Number):
        return a / b if b > 0 else 1
    gtz = b > 0
    return tr.where(gtz, a / tr.where(gtz, b, ones(b.shape) * o) , ones(b.shape) * o)



def wrap(r):
    return r - r.floor()

def tri(r):
    return 1 - (1 - r.abs().fmod(2)).abs()

def erpt(args, S):

    samp_shape = [args.WIDTH * args.SUBSAMPLE * args.HEIGHT * args.SUBSAMPLE]
    samps_per_pass = product(samp_shape)

    histogram = new_img(args)
    mc_histogram = new_img(args)

    total_time = 0

    x_sz = (S[2] - S[0])
    y_sz = (S[3] - S[1])

    m = 0
    k = 0

    samp_mul = args.SUBSAMPLE * args.SUBSAMPLE / (args.OVERSAMPLE * args.OVERSAMPLE)

    did_restart = False

    eye_dir = args.eye.norm() * (-1)
    x_dir = vec3(0,1,0).cross(eye_dir).norm()
    y_dir = eye_dir.cross(x_dir).norm()
        
    lower_left = (args.eye + eye_dir * args.focal) - x_dir * (x_sz * 0.5) - y_dir * (y_sz * 0.5)
    
    for i in itertools.count(1,1):
        restart = i % args.restart_freq == 1     
        if restart or (args.mut_restart_freq > 1 and i % args.mut_restart_freq == 1):
            best_samp = vec3u(0, samp_shape)
            best_id = -1 + l_zeros(samp_shape)
            best_samp_params = {}
            if not restart: # refresh
                best_samp_coords = original_samp_coords
                best_samp_params = original_samp_params
            did_restart = True

        getRand, new_samp_params = getPermuteRand(not did_restart, args, samp_shape, best_samp_params)

        samp_coords = vec3(tri(getRand()),tri(getRand()), 0)
        
        samp_cast = lower_left + x_dir * samp_coords.x * x_sz + y_dir * samp_coords.y * y_sz

        if restart:
            k += 1
            did_restart = True
            original_samp_coords = samp_coords
            original_samp_params = new_samp_params

            best_samp_coords = samp_coords
            best_samp_params = new_samp_params

            estimate = multiSamp(args, samp_shape, samp_cast, args.num_mc_samples)

            addS(args, mc_histogram, best_samp_coords, estimate)
            save_img(args, mc_histogram / (k * samp_mul), "estimate.png")
            continue
        did_restart = False

        m += 1

        tPass = time.time()
        
        
        new_samp, new_id = raytrace(args, getRand, args.eye, (samp_cast - args.eye).norm(), bounce = 0) 

        filt = ((best_id < 0) | (new_id == best_id)).double()
        accept_prob = one_or_div(new_samp.luminance(), best_samp.luminance()) * filt
        accept_prob.clamp_(0,1)

        addS(args, histogram, best_samp_coords, (best_samp * estimate.luminance()).div_or(best_samp.luminance(), estimate)* (1 - accept_prob) )
        addS(args, histogram, samp_coords, (new_samp * estimate.luminance() ).div_or(new_samp.luminance(), estimate) * accept_prob)

        accept_var = rand(samp_shape)
        should_accept = (accept_var <= accept_prob)
        shouldnt_accept = 1 - should_accept
        best_samp_params = mixSamples(samp_shape, should_accept.double(), new_samp_params, best_samp_params)
        
        best_samp =           new_samp * should_accept.double() + best_samp        * shouldnt_accept.double()
        best_samp_coords = samp_coords * should_accept.double() + best_samp_coords * shouldnt_accept.double()
        best_id =               new_id * should_accept.int() + best_id          * shouldnt_accept.int()

        #addS(args, histogram, best_samp_coords, (best_samp * estimate.luminance()).div_or(best_samp.luminance(), estimate))

        tCurr = time.time()
        pass_time = tCurr - tPass
        total_time += pass_time

        print("\n\nPass:", m)
        print("\tElapsed Time:", total_time)
        print("\tPass Time:", pass_time)
        print("\tAvg Pass Time:",  total_time / m)

        print("\n\tTotal Samples:", samps_per_pass * (m + k * args.num_mc_samples) )
        print("\tSamples Per Pixel:", args.SUBSAMPLE * m)

        print("\n\tsamp/sec:", samps_per_pass / pass_time )
        print("\tAvg samp/sec:",  samps_per_pass * m / total_time, "\n")

        save_img(args, histogram / (m * samp_mul), "img.png")


def render(args):

    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)

    args.w = args.WIDTH * args.OVERSAMPLE
    args.h = args.HEIGHT * args.OVERSAMPLE

    r = float(args.WIDTH) / args.HEIGHT
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    erpt(args, S)



