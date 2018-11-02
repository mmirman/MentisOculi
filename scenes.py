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
from rayt import *

def cornell():
  class StaticArgs:
    SAVE_DIR="cornell"
    OVERSAMPLE = 2

    SUBSAMPLE = 10

    WIDTH = 300
    HEIGHT = 300

    scene = [
        Light(vec3(0, 2.1, 0.5), 0.4, rgb(1, 1, 1)),
        #Light(vec3(-1.3, 1.7, 0.7), 0.5, rgb(1, 1, 1)),
        #Sphere(vec3(.3, .1, 1.3), .6, rgb(0.1, 0.1, 0), rgb(0.9, 0.95, 1)),
        Sphere(vec3(.42, .35, 0.8), 1.1, rgb(0.1, .1, .02), 0.98, mir=0.99, semi=vec3(-1, -0.7,0.45), semi_low = .45),
        Sphere(vec3(.3, -0.35, 0.9), .4, rgb(0.0, 0.0, 0), rgb(1, 1, 0.8), mir=0.99, semi=vec3(1, 1.5,-0.45), semi_low = .7),
        CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.99, .99, .99), diffuse2 = rgb(0.3, 0.3, 0.8)),
        Sphere(vec3(0, 100000.8, 0), 99999, rgb(0.99, 0.99, 0.99)),
        Sphere(vec3(0, 0, 100001.), 99999, rgb(0.99, 0.99, 0.99)),
        Sphere(vec3(100000.2, 0, 0), 99999, rgb(0.99, 0.6, 0.6)),
        Sphere(vec3(-100000.2, 0, 0), 99999, rgb(0.6, 0.99, 0.6)),
    ]

    eye = vec3(0., 0.35, -1.)     # Eye position
    img_sz = 1
    FARAWAY = 1.0e36            # an implausibly huge distance
    MAX_BOUNCE = 5
    NUDGE = 0.0000001
    STOP_PROB = 0.7

    NEAREST = 0.000000001
    restart_freq = 60
    mut_restart_freq = 20
    num_mc_samples = 20
    jump_size = 0.006
    focal = 1
    eye_focus = vec3(0, 0, 0)     # Eye focus
  return StaticArgs



def ring():
  class StaticArgs:
    SAVE_DIR="ring"
    OVERSAMPLE = 2

    SUBSAMPLE = 8

    WIDTH = 600
    HEIGHT = 400

    scene = [
        Light(vec3(4, 1, -0.8), 0.2, rgb(40, 40, 40)),
        Cylinder(vec3(0, 0, 0.2), 0.8, rgb(0.8, 0.3, 0.0), rgb(1, 0.8, 0.4), mir=0.8, semi=vec3(0, 1, 0), semi_low = -0.1, semi_high = 0.2),
        CheckeredSphere(vec3(0,-100000.05, 0), 100000, rgb(.99, .99, .99), diffuse2 = rgb(0.5, 0.5, 0.7)),
        #Sphere(vec3(0, 0, 100001.), 99999, rgb(0.99, 0.99, 0.99)),
    ]

    eye = vec3(0.2, 1.2, -2) * 2    # Eye position
    img_sz = 1/3.0
    eye_focus = vec3(0, 0.2, 0)     # Eye focus
    FARAWAY = 1.0e36            # an implausibly huge distance
    MAX_BOUNCE = 4
    NUDGE = 0.0000001
    STOP_PROB = 0.8

    NEAREST = 0.000000001
    restart_freq = 60
    mut_restart_freq = 30
    num_mc_samples = 40
    jump_size = 0.008

    focal = 1.5
  return StaticArgs
