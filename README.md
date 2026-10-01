# ggbpl

JAX implementation of the Generalized Geometry Block Proximal Linearized method (GGBPL) and its inertial version (iGGBPL) for l0-constrained sparse nonnegative matrix factorization.

SNCP is not implemented. The authors released MATLAB code at https://github.com/Weifeng-Yang/GGBPL and this repository is not a port of that code.

The U metric is the diagonal of the row sums of V @ V.T plus (alpha1 + eps). The V metric is the diagonal of the row sums of U.T @ U plus (alpha2 + eps), so both metrics are r by r. The metric step uses gamma = 2.0, hence sigma = 1/gamma. The inertial restart threshold is rho1 = 1e-4. The diagonal metric shift is eps = 1e-8.

Install and run:

    pip install -e .
    pytest
    python train.py
