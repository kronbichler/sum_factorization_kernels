# Sum factorization kernels for high-order finite element methods

This small project collects basic sum factorization kernels for high
performance execution on modern cache-based CPUs. This work has been partly
used to run the basic kernel in the publication
```
@article{Kronbichler2019,
author = "Kronbichler, M. and Kormann, K.", 
title  = "Fast matrix-free evaluation of discontinuous {G}alerkin finite element operators",
journal = {ACM Trans. Math. Softw.},
volume = {45},
number = {3},
pages = {29:1--29:40},
year = {2019},
doi = {10.1145/3325864}
}
```

Parts of the implementation have been taken from the deal.II finite element,
https://github.com/dealii/dealii.
