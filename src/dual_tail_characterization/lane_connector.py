"""Exact midpoint, endpoint-preserving amplitude connector and slope inversion."""
import numpy as np
from scipy.optimize import brentq
from .shared_endpoints import logistic_weight

def validate(kappa, center):
    if not np.isfinite(kappa) or not 0 < kappa <= 1e6:
        raise ValueError("kappa must be finite in (0, 1e6]")
    if not np.isfinite(center) or not 1e-9 <= center <= 1-1e-9:
        raise ValueError("midpoint must be within [1e-9, 1-1e-9] UI")

def connector(u, kappa, center=.5):
    validate(kappa, center)
    u=np.asarray(u,dtype=float)
    if not np.all(np.isfinite(u)) or np.any((u<0)|(u>1)):
        raise ValueError("phase must lie in [0,1]")
    q=u*(1-center)/(center*(1-u)+u*(1-center))
    return np.where(u==center,.5,logistic_weight(q,kappa))

def midpoint_derivative(kappa, center=.5):
    validate(kappa,center)
    ratio=4. if kappa<1e-6 else kappa/np.tanh(kappa/4)
    return float(ratio/(16*center*(1-center)))

def solve_kappa(magnitude, delta, center=.5):
    validate(1.,center)
    minimum=abs(delta)/(4*center*(1-center))
    maximum=abs(delta)*midpoint_derivative(1e6,center)
    if not np.isfinite(magnitude) or delta==0 or not minimum < magnitude <= maximum:
        raise ValueError(f"slope magnitude must exceed {minimum:g} and be <= {maximum:g}; zero nominal amplitude difference cannot be inverted")
    return float(brentq(lambda k: abs(delta)*midpoint_derivative(k,center)-magnitude,1e-9,1e6))
