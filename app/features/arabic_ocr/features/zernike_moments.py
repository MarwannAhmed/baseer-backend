import numpy as np
import math
from arabic_ocr.config import NORM_SIZE
import mahotas


def zernike_features(norm_img: np.ndarray, degree: int = 8):
    try:
        radius = NORM_SIZE // 2
        feats = mahotas.features.zernike_moments(
            (norm_img < 128).astype(np.uint8), radius=radius, degree=degree
        )
        return feats.astype(np.float32)
    except ImportError:
        return _zernike_manual(norm_img, degree)



def _zernike_manual(img: np.ndarray, degree: int):
    binary = (img < 128).astype(float)
    height, width = binary.shape
    y_index, x_index = np.indices((height, width))
    x_norm = (x_index - width / 2.0) / (width / 2.0)
    y_norm = (y_index - height / 2.0) / (height / 2.0)
    rho   = np.hypot(x_norm, y_norm)
    theta = np.arctan2(y_norm, x_norm)

    inside = rho <= 1.0 #which pixels are inside unit circle, only those contribute to Zernike moments
    moments: list[float] = []

    #multiplies with the basis function and sums (projecton onto that basis)
    for n in range(degree + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 != 0:
                continue
            R = _radial_poly(n, abs(m), rho)
            V = R * np.exp(1j * m * theta)
            Z = np.sum(binary[inside] * np.conj(V[inside]))
            Z *= (n + 1) / np.pi
            moments.append(abs(Z)) 

    return np.array(moments, dtype=np.float32)

def _radial_poly(n: int, m: int, rho: np.ndarray):
    result = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        coeff = (
            ((-1) ** s)
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + m) // 2 - s)
                * math.factorial((n - m) // 2 - s)
            )
        )
        result += coeff * rho ** (n - 2 * s)
    return result

