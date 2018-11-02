MentisOculi Pytorch Path Tracer
======================================

 ![example](https://raw.githubusercontent.com/mmirman/MentisOculi/master/cyl.png)

* A very simple and small path tracer written in pytorch meant to be run on the GPU
* Why use pytorch and not some other cuda library or shaders?  To enable arbitrary automatic differentiation. And because I can.

Features
--------

* Can trace reflective spheres and open cylinders with flat and checkered materials.
* [Energy Redistribution Path Tracing](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.5938&rep=rep1&type=pdf) with transitions limited to rays intersecting same initial diffuse material.
* Backwards (camera to light) path tracing
* [Ray permutations in the hypercube of random numbers](http://sirkan.iit.bme.hu/~szirmay/paper50_electronic.pdf)
* Nearly everything done in large batches on the GPU.

Future Directions
-----------------

* Langevin Metropolis and HMC both use gradients to increase the efficiency of sampling.  [This paper outlines how to do these for ray tracing.](https://cseweb.ucsd.edu/~ravir/h2mc_clean.pdf)
* [This paper demonstrates how to make ray tracing even more differentiable for the benefit of inverse rendering](https://people.csail.mit.edu/tzumao/diffrt/)
* Triangles and a GPU tailored ray acceleration datastructure.
* [Metropolis SPPM?](https://dl.acm.org/citation.cfm?id=2383509)  With HMC?

Credits
-------

* While the code has been significantly morphed, it was originally a fork [James Bowmans' python raytracer](http://www.excamera.com/sphinx/article-ray.html)
* This was inspired by my ongoing work on secure differentiable programming, i.e. adversarial examples in AI at at the [ETH SRI Lab](https://www.sri.inf.ethz.ch/).  
