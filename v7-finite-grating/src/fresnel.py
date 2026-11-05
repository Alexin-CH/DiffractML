"""Analytic Fresnel reflection/transmission for a planar interface.

Used to validate the MEEP finite-grating machinery (source launch, flux
normalisation, reflection/transmission signs) against an exact solution before
trusting it on the grating.  A grating with zero corrugation and zero base
thickness reduces to a planar ``n1`` (superstrate) / ``n2`` (substrate)
interface.
"""

from __future__ import annotations

import numpy as np


def fresnel_RT(n1: float, n2: float, angle_deg: float, pol: str):
    """Return analytic ``(R, T)`` for a planar interface, lossless media.

    ``pol='s'`` (TE, E out of plane) or ``pol='p'`` (TM, E in plane).
    ``T`` is the transmitted power fraction (``R + T = 1`` for real ``n``).
    """
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")

    theta_i = np.deg2rad(np.asarray(angle_deg, dtype=float))
    sin_t = (n1 / n2) * np.sin(theta_i)
    sin_t = np.clip(sin_t, -1.0, 1.0)  # evanescent beyond critical angle
    theta_t = np.arcsin(sin_t)
    cos_i, cos_t = np.cos(theta_i), np.cos(theta_t)

    if pol == "s":
        r = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        t = (2.0 * n1 * cos_i) / (n1 * cos_i + n2 * cos_t)
    else:
        r = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
        t = (2.0 * n1 * cos_i) / (n2 * cos_i + n1 * cos_t)

    R = np.abs(r) ** 2
    T = np.real((n2 * cos_t) / (n1 * cos_i)) * np.abs(t) ** 2
    return R, T
