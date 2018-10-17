from PIL import Image
import numpy as np
import torch as tr
from helpers import *

import time

from functools import reduce


OVERSAMPLE = 4
SUBSAMPLE = 4
WIDTH = 600
HEIGHT = 450


def save_img(color, nm):
    print("saving: ", nm)
    rgb = [Image.fromarray(np.array(c.clamp(0, 1).reshape((h, w)) * 255), "F").resize((WIDTH, HEIGHT), Image.ANTIALIAS).convert("L") for c in color.components()]
    Image.merge("RGB", rgb).save("out/" + nm)


L = vec3(5, 5., -10)        # Point light position
E = vec3(0., 0.35, -1.)     # Eye position
FARAWAY = 1.0e36            # an implausibly huge distance
MAX_BOUNCE = 3
NUDGE = 0.002

def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(lambda x,y: tr.min(dtype(x),dtype(y)), distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if tr.sum(hit).data > 0:
            Oc = O.extract(hit)
            dc = extract(hit, d)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, bounce)
            color += cc.place(hit)
    return color


def pathtrace(origin, S, pixels, scene):
    img = 0
    for i in range(SUBSAMPLE):
        x_sz = (S[2] - S[0]) / w
        y_sz = (S[3] - S[1]) / h
        x_diffs = rand(pixels.x.shape) * x_sz
        y_diffs = rand(pixels.y.shape) * y_sz
        pixel_mod = pixels + vec3(x_diffs, y_diffs, 0)
        sub_img = raytrace(origin, (pixel_mod - origin).norm(), scene, bounce = 0) 
        img = sub_img + img
        save_img(sub_img, "sub_img"+str(i)+".png")
        save_img(img / (i + 1), "img"+str(i)+".png")
    return img / SUBSAMPLE

class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = tr.sqrt(tr.relu(disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = tr.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return tr.where(pred, h, ones_like(h) * FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        # D is direction
        # O is previous origin
        M = (O + D * d)                         # new intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        newO = M + N * NUDGE                    # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(newO, toL) for s in scene]
        light_nearest = reduce(tr.min, light_distances)
        seelight = (light_distances[scene.index(self)] == light_nearest).float()

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = N.dot(toL).relu()
        x = self.diffusecolor(M) * lv * seelight
        color += x

        # Reflection
        if bounce < MAX_BOUNCE:
            rayD = (D - N * 2 * D.dot(N)).norm()  # reflection
            color += raytrace(newO, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * tr.pow(tr.clamp(phong, 0, 1), 50) * seelight
        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).int() % 2) == ((M.z * 2).int() % 2)
        return self.diffuse * checker.float()

scene = [
    Sphere(vec3(.75, .1, 1.), .6, rgb(0, 0, 1)),
    Sphere(vec3(-.75, .1, 2.25), .6, rgb(.5, .223, .5)),
    Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1., .572, .184)),
    CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .75), 0),
    ]

t0 = time.time()

(w, h) = (WIDTH * OVERSAMPLE, HEIGHT * OVERSAMPLE)

r = float(WIDTH) / HEIGHT
# Screen coordinates: x0, y0, x1, y1.
S = (-1., 1. / r + .25, 1., -1. / r + .25)
x =  linspace(S[0], S[2], w).repeat(h)
y = linspace(S[1], S[3], h).view(-1,1).expand(h,w).contiguous().view(-1)

Q = vec3(x, y, 0)

color = pathtrace(E, S, Q, scene)
print("Took", time.time() - t0)

save_img(color, "img.png")
