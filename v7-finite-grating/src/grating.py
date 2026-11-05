"""Finite 1D grating geometry + realistic defect model (pure NumPy).

Units
-----
All lengths are in micrometres (µm), matching MEEP's default unit distance
``a = 1 µm`` (so frequencies are in 1/µm and ``freq = 1 / wavelength_um``).

The grating is a *finite aperture* of ``n_ridges`` sine-corrugated TiN ridges on
a substrate.  It is translation-invariant along ``y`` (a 2D x-z problem) and
truncated along ``x``: outside the aperture there is no TiN, which is the
finite-edge situation that periodic RCWA cannot represent without padding.

Two realistic defect types are supported, matching the project brief:

* **height jitter** -- each ridge's corrugation amplitude is multiplied by a
  per-ridge factor drawn from a configurable distribution;
* **missing ridge** -- selected ridges are removed entirely down to the
  substrate (a ``h = 0`` gap).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

_TWO_PI = 2.0 * np.pi


@dataclass
class GratingSpec:
    """Description of a finite sine-corrugated 1D grating.

    Parameters
    ----------
    n_ridges : int
        Number of ridges in the finite aperture (>= 1).
    period : float
        Grating period in µm.
    amplitude : float
        Corrugation amplitude in µm.  The local TiN height above the uniform
        base is ``amplitude * (1 + sin(...))``, so the peak-to-valley
        corrugation is ``2 * amplitude`` (matching the v6 layer stack).
    base_thickness : float
        Uniform TiN thickness in µm under every ridge (v6 ``uni_layer_h``).
    substrate_index : float
        Refractive index of the semi-infinite substrate below the TiN.
    superstrate_index : float
        Refractive index of the incident medium above the grating.
    ridge_scale : np.ndarray, optional
        Per-ridge multiplier applied to ``amplitude`` (height-jitter defects).
        Shape ``(n_ridges,)``; defaults to all ones.
    missing : np.ndarray, optional
        Boolean mask ``(n_ridges,)``; ``True`` removes the whole ridge (base +
        corrugation) within its period, leaving a bare-substrate gap.
    phase : float
        Phase offset (radians) of the sine profile.
    """

    n_ridges: int = 10
    period: float = 1.5
    amplitude: float = 0.045
    base_thickness: float = 0.03
    substrate_index: float = 1.5
    superstrate_index: float = 1.0
    ridge_scale: Optional[np.ndarray] = None
    missing: Optional[np.ndarray] = None
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.n_ridges < 1:
            raise ValueError("n_ridges must be >= 1")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if self.base_thickness < 0:
            raise ValueError("base_thickness must be non-negative")

        if self.ridge_scale is None:
            self.ridge_scale = np.ones(self.n_ridges, dtype=float)
        else:
            self.ridge_scale = np.asarray(self.ridge_scale, dtype=float).reshape(-1)
            if self.ridge_scale.size != self.n_ridges:
                raise ValueError("ridge_scale must have length n_ridges")

        if self.missing is None:
            self.missing = np.zeros(self.n_ridges, dtype=bool)
        else:
            self.missing = np.asarray(self.missing, dtype=bool).reshape(-1)
            if self.missing.size != self.n_ridges:
                raise ValueError("missing must have length n_ridges")

    # ------------------------------------------------------------------ #
    # Derived geometry
    # ------------------------------------------------------------------ #
    @property
    def aperture_width(self) -> float:
        """Total lateral extent of the finite grating (µm)."""
        return self.n_ridges * self.period

    @property
    def ridge_centers(self) -> np.ndarray:
        """x-coordinate of each ridge centre (aperture centred on x=0)."""
        half = 0.5 * self.aperture_width
        edges = np.linspace(-half, half, self.n_ridges + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def max_height(self) -> float:
        """Largest possible TiN top above the substrate (µm)."""
        scale = float(np.max(np.abs(self.ridge_scale))) if self.ridge_scale.size else 0.0
        return self.base_thickness + 2.0 * self.amplitude * scale

    # ------------------------------------------------------------------ #
    # Height profile h(x): top of the TiN above the substrate plane (z=0)
    # ------------------------------------------------------------------ #
    def height_profile(self, x: np.ndarray) -> np.ndarray:
        """TiN height above the substrate at lateral positions ``x`` (µm).

        Outside the finite aperture ``h = 0`` (bare substrate).  Within each
        ridge period the top follows ``base + amp * (1 + sin(2π x'/period + φ))``
        scaled by that ridge's jitter factor, unless the ridge is missing, in
        which case ``h = 0`` over its full period.
        """
        x = np.asarray(x, dtype=float)
        h = np.zeros_like(x)

        half = 0.5 * self.aperture_width
        inside = (x >= -half) & (x < half)
        if not np.any(inside):
            return h

        xi = x[inside]
        # local coordinate inside each ridge's period: [-period/2, period/2)
        local = (xi + half) % self.period - 0.5 * self.period
        # ridge index (0 .. n_ridges-1)
        idx = np.floor((xi + half) / self.period).astype(int)
        np.clip(idx, 0, self.n_ridges - 1, out=idx)

        corrugation = self.amplitude * (
            1.0 + np.sin(_TWO_PI * local / self.period + self.phase)
        )
        corrugation *= self.ridge_scale[idx]
        h_inside = self.base_thickness + corrugation
        h_inside[self.missing[idx]] = 0.0

        h[inside] = h_inside
        return h

    # ------------------------------------------------------------------ #
    # Defect sampling
    # ------------------------------------------------------------------ #
    def with_defects(
        self,
        *,
        jitter_std: float = 0.0,
        n_missing: int = 0,
        rng: Optional[np.random.Generator] = None,
        min_scale: float = 0.0,
    ) -> "GratingSpec":
        """Return a copy with random height jitter and/or missing ridges.

        Parameters
        ----------
        jitter_std : float
            Standard deviation of the per-ridge height multiplier (Gaussian,
            centred on 1).  ``0`` disables height jitter.
        n_missing : int
            Number of ridges to remove entirely (uniformly chosen without
            replacement).  ``0`` disables missing ridges.
        rng : np.random.Generator, optional
            Random generator for reproducibility.
        min_scale : float
            Lower clip for the jittered multiplier (keeps heights physical).
        """
        if rng is None:
            rng = np.random.default_rng()

        scale = np.ones(self.n_ridges, dtype=float)
        if jitter_std > 0.0:
            scale = rng.normal(1.0, jitter_std, size=self.n_ridges)
            scale = np.clip(scale, min_scale, None)

        missing = np.zeros(self.n_ridges, dtype=bool)
        if n_missing > 0:
            n_missing = min(int(n_missing), self.n_ridges)
            chosen = rng.choice(self.n_ridges, size=n_missing, replace=False)
            missing[chosen] = True

        return replace(self, ridge_scale=scale, missing=missing)


def sample_grating_spec(
    rng: np.random.Generator,
    *,
    n_ridges: int = 10,
    period_range=(0.6, 2.0),
    amplitude_range=(0.02, 0.08),
    base_thickness: float = 0.03,
    substrate_index: float = 1.5,
    jitter_std: float = 0.1,
    max_missing: int = 2,
) -> GratingSpec:
    """Draw a random finite-grating spec with defects for dataset generation."""
    period = float(rng.uniform(*period_range))
    amplitude = float(rng.uniform(*amplitude_range))
    n_missing = int(rng.integers(0, max_missing + 1)) if max_missing > 0 else 0
    spec = GratingSpec(
        n_ridges=n_ridges,
        period=period,
        amplitude=amplitude,
        base_thickness=base_thickness,
        substrate_index=substrate_index,
    )
    return spec.with_defects(jitter_std=jitter_std, n_missing=n_missing, rng=rng)
