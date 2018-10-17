from PIL import Image
import numpy as np
import torch as tr
from helpers import *

import time
import math

from functools import reduce


OVERSAMPLE = 4
SUBSAMPLE = 32
WIDTH = 400
HEIGHT = 300
(w, h) = (WIDTH * OVERSAMPLE, HEIGHT * OVERSAMPLE)

def save_img(color, nm):
    print("saving: ", nm)
    color = color 
    rgb = [Image.fromarray(np.array(c.clamp(0, 1).reshape((h, w)).float() * 255), "F").resize((WIDTH, HEIGHT), Image.ANTIALIAS).convert("L") for c in color.components()]
    Image.merge("RGB", rgb).save("out/" + nm)


L = vec3(1.5, 0.5, 0.8)        # Point light position
E = vec3(0., 0.35, -1.)     # Eye position
FARAWAY = 1.0e36            # an implausibly huge distance
MAX_BOUNCE = 4
NUDGE = 0.0000001

def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [dtype(s.intersect(O, D)) for s in scene]
    nearest = reduce(tr.min, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest < FARAWAY) & (d == nearest) & (d > NUDGE) # d == nearest is hacky af
        if tr.sum(hit).data > 0:
            Oc = O.extract(hit)
            dc = extract(hit, d)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, bounce)
            color += cc.place(hit)
    return color


mcmc_generator = []

def getRand(s):
    global mcmc_generator
    
    r = rand(s.shape)
    mcmc_generator += [r]
    return r

def pathtrace(origin, S, pixels, scene):
    global mcmc_generator
    img = 0

    mcmc_best = list(mcmc_generator)

    for i in range(SUBSAMPLE):
        mcmc_generator = []

        x_sz = (S[2] - S[0]) / w
        y_sz = (S[3] - S[1]) / h

        pixel_mod = pixels + vec3(getRand(pixels.x) * x_sz, getRand(pixels.y) * y_sz, 0)
        sub_img = raytrace(origin, (pixel_mod - origin).norm(), scene, bounce = 0) 

        img = sub_img + img


        save_img(sub_img, "sub_img"+str(i)+".png")
        save_img(img / (i + 1), "img"+str(i)+".png")
        save_img(img / (i + 1), "img.png")

    return img / SUBSAMPLE

def random_spherical(u, v):
    theta  = u * 2 * math.pi
    phi = v * math.pi 

    # Switch to cartesian coordinates
    sphi = phi.sin()
    x = theta.cos() * sphi
    y = theta.sin() * sphi

    return vec3(x, y, phi.cos())

class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5, phong_pow = 50, phong_col = rgb(1, 1, 1)):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror
        self.phong_pow = phong_pow
        self.phong_col = phong_col

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = tr.sqrt(tr.relu(disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = tr.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0.000000001)
        return tr.where(pred, h, ones_like(h) * FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        # D is direction
        # O is previous origin
        M = (O + D * d)                         # new intersection point
        N = (M - self.c) * (1. / self.r)        # normal

        toL = (L - M).norm()                    # direction to light
        toO = (O - M).norm()                    # direction to ray origin
        newO = M + N * NUDGE                    # M nudged to avoid itself


        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(newO, toL) for s in scene]
        light_nearest = reduce(tr.min, light_distances)
        seelight = ((light_nearest < 0)  + (light_nearest * light_nearest >= abs(L - M))).double()

        # Ambient
        color = rgb(0.0, 0.0, 0.0)

        # Lambert shading (diffuse)
        lv = N.dot(toL).relu()
        x = self.diffusecolor(M) * lv * seelight
        color += x

        # Reflection
        if bounce < MAX_BOUNCE:

            
            rayDiff = random_spherical(getRand(N.x), getRand(N.x))
            should_flip = N.dot(rayDiff).lt(0).double()
            rayDiff = rayDiff * (1 - 2 * should_flip)

            color += raytrace(newO, rayDiff , scene, bounce + 1) * rayDiff.dot(N) * self.diffusecolor(M)

            rayRefl = (D - N * 2 * D.dot(N)).norm()  # reflection            
            color += raytrace(newO, rayRefl, scene, bounce + 1) * self.mirror * rayDiff.dot(N)

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += self.phong_col * tr.pow(tr.clamp(phong, 0, 1), self.phong_pow) * seelight
        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).int() % 2) == ((M.z * 2).int() % 2)
        return self.diffuse * checker.double() + rgb(0.6, 0.5, 0.5) * (1 - checker.double())

class Light(Sphere):
    def light(self, O, D, d, scene, bounce):
        return self.diffuse

scene = [
    Light(vec3(1.7, 0.6, 0.7), 0.1, rgb(3, 3, 3), 0, 0, 0),
    Sphere(vec3(0, 200, 1), 197, rgb(0.9, 0.9, 0.9), 0.1, phong_pow = 1, phong_col=rgb(0,0,0)),
    Sphere(vec3(.3, .1, 1.3), .6, rgb(0.1, 0.1, 0), rgb(0.1, 0.9, 1), phong_pow = 1000, phong_col=rgb(0,0.6,0)),
    Sphere(vec3(-.4, .2, 0.8), .4, rgb(1, .8, .9).rgbNorm() * 3 * 0.4, 0.6, phong_pow = 1000, phong_col=rgb(0.2,0,0)),
    CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .85), 0, phong_pow = 1, phong_col=rgb(0,0,0)),
    ]

t0 = time.time()



r = float(WIDTH) / HEIGHT
# Screen coordinates: x0, y0, x1, y1.
S = (-1., 1. / r + .25, 1., -1. / r + .25)
x =  linspace(S[0], S[2], w).repeat(h)
y = linspace(S[1], S[3], h).view(-1,1).expand(h,w).contiguous().view(-1)

Q = vec3(x, y, 0)

color = pathtrace(E, S, Q, scene)
print("Took", time.time() - t0)

save_img(color, "img.png")
