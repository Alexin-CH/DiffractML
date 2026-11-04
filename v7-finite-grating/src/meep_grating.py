"""MEEP (FDTD) ground-truth simulator for a finite, defective 1D grating.

This is the M0 deliverable of ``v7``: a full-wave reference for
transmission/reflection spectra of a *finite* grating (normal or oblique
incidence), which periodic RCWA can only approximate via a padded supercell.

Why FDTD here
-------------
A finite aperture with edges and defects is not periodic, so RCWA's unit-cell
assumption does not apply.  MEEP solves Maxwell's equations on the full
finite structure with PML absorbing boundaries, giving an independent
reference for the total reflected/transmitted power.

Coordinate mapping
------------------
MEEP's native 2D plane is x-y (with ``z`` invariant), so the physical x-z
problem (invariant along physical y) is mapped as::

    physical x -> MEEP x        (lateral, finite aperture)
    physical z -> MEEP y        (propagation)
    invariant y -> MEEP z       (out of plane)

Consequently the physical s-polarisation (E along physical y) is MEEP ``Ez``,
and p-polarisation (E in physical x-z) is MEEP ``Ex``/``Ey``.

Definitions
-----------
*Total power across a plane*: ``R`` and ``T`` are the power fluxes integrated
over the plane above / below the structure, normalised by the incident power
measured in an empty-cell control run.  This is the quantity the RCWA baseline
must reproduce (sum of all propagating-order powers).
"""

from __future__ import annotations

import argparse
import cmath
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from grating import GratingSpec
from tin_dispersion import get_or_fit

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIT_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, "..", "data"))


@dataclass
class FDTDResult:
    """Broadband R/T result for one finite grating and incidence condition."""

    wl_um: np.ndarray
    R: np.ndarray
    T: np.ndarray
    angle_deg: float
    pol: str
    spec: GratingSpec
    meta: dict = field(default_factory=dict)

    @property
    def absorptance(self) -> np.ndarray:
        return 1.0 - self.R - self.T

    def to_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        header = "wl_um,R,T,A"
        data = np.column_stack([self.wl_um, self.R, self.T, self.absorptance])
        np.savetxt(path, data, delimiter=",", header=header, comments="")


# --------------------------------------------------------------------------- #
# Source construction
# --------------------------------------------------------------------------- #
def _oblique_amp_func(kx: float, scale: float = 1.0):
    """Amplitude function ``scale * exp(i kx x)`` for a planewave source.

    Sources are centred at ``x = 0`` in this module, so ``r.x`` is already the
    global coordinate (cf. MEEP's ``pw-source.py`` example).
    """

    def amp(r):
        return scale * cmath.exp(1j * kx * r.x)

    return amp


def _make_sources(
    meep,
    fcen: float,
    fwidth: float,
    *,
    angle_deg: float,
    pol: str,
    z_src: float,
    source_width: float,
    superstrate_index: float,
    is_integrated: bool = True,
):
    """Build the incident planewave source list for the requested polarization.

    In MEEP coordinates the source is a line along ``x`` at ``y = z_src``.
    ``pol="s"`` uses the out-of-plane ``Ez``; ``pol="p"`` uses in-plane
    ``Ex``/``Ey``.  Normal incidence is a full-width integrated source; oblique
    incidence uses the approximate planewave construction from MEEP's
    ``pw-source.py``.
    """
    theta = math.radians(angle_deg)
    k0 = 2.0 * math.pi * fcen * superstrate_index
    kx = k0 * math.sin(theta)
    normal = abs(angle_deg) < 1e-9
    center = meep.Vector3(0, z_src, 0)
    size = meep.Vector3(source_width, 0, 0)

    if normal:
        src = meep.GaussianSource(fcen, fwidth=fwidth, is_integrated=is_integrated)
        component = meep.Ez if pol == "s" else meep.Ex
        return [meep.Source(src, component=component, center=center, size=size)]

    # Oblique incidence: approximate planewave via a phase-ramped current sheet.
    src = meep.GaussianSource(fcen, fwidth=fwidth)
    if pol == "s":
        return [
            meep.Source(
                src, component=meep.Ez, center=center, size=size,
                amp_func=_oblique_amp_func(kx),
            )
        ]
    # p-pol: current parallel to the in-plane E field (cosθ x̂ + ... ).
    return [
        meep.Source(
            src, component=meep.Ex, center=center, size=size,
            amp_func=_oblique_amp_func(kx, scale=math.cos(theta)),
        ),
        meep.Source(
            src, component=meep.Ey, center=center, size=size,
            amp_func=_oblique_amp_func(kx, scale=-math.sin(theta)),
        ),
    ]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _build_geometry(meep, spec: GratingSpec, tin_material, *, resolution: float,
                    sub_depth: float):
    """Return MEEP geometry: substrate slab + TiN columns of the finite grating."""
    substrate = meep.Medium(index=spec.substrate_index)

    # Substrate slab from the bottom of the cell up to the grating base (y=0).
    geom = [
        meep.Block(
            size=meep.Vector3(meep.inf, sub_depth, meep.inf),
            center=meep.Vector3(0, -0.5 * sub_depth, 0),
            material=substrate,
        )
    ]

    # Column decomposition of the arbitrary (defective) height profile.  One
    # column per FDTD pixel keeps the staircase aligned with the Yee grid.
    dx_pix = 1.0 / resolution
    n_col = max(1, int(round(spec.aperture_width / dx_pix)))
    dx = spec.aperture_width / n_col
    edges = np.linspace(-0.5 * spec.aperture_width, 0.5 * spec.aperture_width, n_col + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    heights = spec.height_profile(centers)

    overlap = 0.0  # columns start exactly at the substrate plane (z=0)
    for xc, h in zip(centers, heights):
        if h <= 0.0:
            continue
        geom.append(
            meep.Block(
                size=meep.Vector3(dx, h + overlap, meep.inf),
                center=meep.Vector3(float(xc), 0.5 * (h - overlap), 0),
                material=tin_material,
            )
        )
    return geom


# --------------------------------------------------------------------------- #
# One simulation (structure + empty reference)
# --------------------------------------------------------------------------- #
def _run_until_decayed(sim, meep, pol: str, point, decay_by: float, dT: float):
    comp = meep.Ez if pol == "s" else meep.Ex
    sim.run(until_after_sources=meep.stop_when_fields_decayed(dT, comp, point, decay_by))


def _run_once(
    meep,
    spec: GratingSpec,
    *,
    wl_range,
    n_freq: int,
    angle_deg: float,
    pol: str,
    resolution: float,
    dpml: float,
    pad: float,
    air_above: float,
    sub_depth: float,
    tin_material,
    decay_by: float,
    dT: float,
    freq_center: Optional[float] = None,
    freq_width: Optional[float] = None,
    periodic: bool = False,
):
    """Run structure + empty control and return ``(wl, R, T, meta)``."""
    wl_min, wl_max = sorted(wl_range)
    fmin, fmax = 1.0 / wl_max, 1.0 / wl_min
    fcen = 0.5 * (fmin + fmax) if freq_center is None else float(freq_center)
    fwidth = (fmax - fmin) if freq_width is None else float(freq_width)

    H = spec.max_height
    sx = spec.aperture_width if periodic else (spec.aperture_width + 2.0 * pad + 2.0 * dpml)
    s_bottom = -(sub_depth + dpml)
    s_top = H + air_above + dpml
    sy = s_top - s_bottom
    cell = meep.Vector3(sx, sy, 0)  # 2D x-y cell

    y_src = H + 0.7 * air_above
    y_refl = H + 0.3 * air_above
    y_tran = -0.5 * sub_depth

    # Normal incidence: integrate a full-width source.  Oblique: approximate
    # planewave spanning only the non-PML region (MEEP pw-source.py recipe).
    normal = abs(angle_deg) < 1e-9
    source_width = sx if (normal or periodic) else (sx - 2.0 * dpml)

    def _sim(geometry):
        sources = _make_sources(
            meep, fcen, fwidth,
            angle_deg=angle_deg, pol=pol, z_src=y_src,
            source_width=source_width,
            superstrate_index=spec.superstrate_index,
            is_integrated=not periodic,
        )
        boundaries = ([meep.PML(dpml, direction=meep.Y)] if periodic
                      else [meep.PML(dpml)])
        extra = {"k_point": meep.Vector3()} if periodic else {}
        return meep.Simulation(
            cell_size=cell,
            boundary_layers=boundaries,
            geometry=geometry,
            sources=sources,
            default_material=meep.Medium(index=spec.superstrate_index),
            resolution=resolution,
            **extra,
        )

    nfreq = max(1, int(n_freq))

    # ---- control run: empty cell (no grating, no substrate) ---- #
    sim = _sim([])
    refl_fr = meep.FluxRegion(center=meep.Vector3(0, y_refl, 0), size=meep.Vector3(sx, 0, 0))
    tran_fr = meep.FluxRegion(center=meep.Vector3(0, y_tran, 0), size=meep.Vector3(sx, 0, 0))
    refl = sim.add_flux(fcen, fwidth, nfreq, refl_fr)
    tran = sim.add_flux(fcen, fwidth, nfreq, tran_fr)
    _run_until_decayed(sim, meep, pol, meep.Vector3(0, y_refl, 0), decay_by, dT)
    empty_refl_data = sim.get_flux_data(refl)
    empty_refl_flux = np.asarray(meep.get_fluxes(refl), dtype=float)
    empty_tran_flux = np.asarray(meep.get_fluxes(tran), dtype=float)
    # Incident wave travels -y, so the signed fluxes are negative; normalise
    # each plane by the incident flux measured at that same plane.
    inc_refl = -empty_refl_flux
    inc_tran = -empty_tran_flux
    if np.any(inc_refl <= 0) or np.any(inc_tran <= 0):
        raise RuntimeError("non-positive incident flux in control run")
    sim.reset_meep()

    # ---- structure run ---- #
    sim = _sim(
        _build_geometry(
            meep, spec, tin_material,
            resolution=resolution, sub_depth=sub_depth + dpml,
        )
    )
    refl = sim.add_flux(fcen, fwidth, nfreq, refl_fr)
    tran = sim.add_flux(fcen, fwidth, nfreq, tran_fr)
    sim.load_minus_flux_data(refl, empty_refl_data)
    _run_until_decayed(sim, meep, pol, meep.Vector3(0, y_refl, 0), decay_by, dT)

    refl_flux = np.asarray(meep.get_fluxes(refl), dtype=float)
    tran_flux = np.asarray(meep.get_fluxes(tran), dtype=float)
    freqs = np.asarray(meep.get_flux_freqs(refl), dtype=float)

    R = refl_flux / inc_refl
    T = -tran_flux / inc_refl
    wl_um = 1.0 / freqs

    meta = {
        "fcen": fcen,
        "fwidth": fwidth,
        "incident_flux": inc_refl,
        "sx": sx,
        "sy": sy,
        "resolution": resolution,
        "dpml": dpml,
        "pad": pad,
        "y_src": y_src,
        "y_refl": y_refl,
        "y_tran": y_tran,
    }
    return wl_um, R, T, meta


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_grating(
    spec: GratingSpec,
    *,
    wl_range=(0.5, 1.5),
    n_freq: int = 60,
    angle_deg: float = 0.0,
    pol: str = "s",
    resolution: float = 40.0,
    dpml: float = 0.5,
    pad: float = 0.6,
    air_above: float = 0.6,
    sub_depth: float = 0.8,
    tin_params: Optional[dict] = None,
    tin_dataset: str = "beliaev",
    tin_fit_path: Optional[str] = None,
    tin_material=None,
    decay_by: float = 1e-6,
    dT: float = 20.0,
    verbose: bool = False,
) -> FDTDResult:
    """Compute the R/T spectrum of a finite grating with MEEP.

    ``angle_deg == 0`` runs a single broadband simulation.  For oblique
    incidence the correct fixed-angle spectrum requires one narrowband run per
    requested wavelength (a fixed spatial phase ramp only matches one
    frequency), so the cost scales with ``n_freq``.

    ``tin_material`` can be supplied to override the fitted dispersive TiN
    (e.g. a lossless test dielectric).
    """
    import meep

    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")

    if tin_material is None:
        if tin_params is None:
            cache = tin_fit_path or os.path.join(
                _DEFAULT_FIT_DIR, f"tin_fit_{tin_dataset}.json"
            )
            tin_params = get_or_fit(tin_dataset, cache_path=cache)
        from tin_dispersion import build_meep_medium

        tin_material = build_meep_medium(tin_params, meep)

    common = dict(
        spec=spec,
        angle_deg=angle_deg,
        pol=pol,
        resolution=resolution,
        dpml=dpml,
        pad=pad,
        air_above=air_above,
        sub_depth=sub_depth,
        tin_material=tin_material,
        decay_by=decay_by,
        dT=dT,
    )

    if abs(angle_deg) < 1e-9:
        wl, R, T, meta = _run_once(meep, wl_range=wl_range, n_freq=n_freq, **common)
        return FDTDResult(wl, R, T, angle_deg, pol, spec, meta)

    # Oblique: one narrowband run per target wavelength.  A wider air pad is
    # needed so the finite planewave source approximates an infinite plane wave
    # over the aperture (otherwise lateral spill biases R).
    common["pad"] = max(common["pad"], 1.5)
    common["air_above"] = max(common["air_above"], 1.0)
    wl_targets = np.linspace(min(wl_range), max(wl_range), max(1, int(n_freq)))
    Rs, Ts, out_meta = [], [], []
    for wl_t in wl_targets:
        f_t = 1.0 / wl_t
        fwidth = 0.05 * f_t  # narrowband so the angle is well defined
        _, R, T, meta = _run_once(
            meep,
            wl_range=(wl_t * 0.995, wl_t * 1.005),
            n_freq=1,
            freq_center=f_t,
            freq_width=fwidth,
            **common,
        )
        Rs.append(float(np.atleast_1d(R)[0]))
        Ts.append(float(np.atleast_1d(T)[0]))
        out_meta.append(meta)
        if verbose:
            print(f"  wl={wl_t:.3f} µm  R={Rs[-1]:.4f}  T={Ts[-1]:.4f}")

    return FDTDResult(
        np.asarray(wl_targets),
        np.asarray(Rs),
        np.asarray(Ts),
        angle_deg,
        pol,
        spec,
        {"runs": out_meta},
    )


def run_grating_periodic(
    spec: GratingSpec,
    *,
    wl_range=(0.5, 1.5),
    n_freq: int = 60,
    pol: str = "s",
    resolution: float = 40.0,
    dpml: float = 0.5,
    air_above: float = 0.6,
    sub_depth: float = 0.8,
    tin_params: Optional[dict] = None,
    tin_dataset: str = "beliaev",
    tin_fit_path: Optional[str] = None,
    tin_material=None,
    decay_by: float = 1e-6,
    dT: float = 20.0,
) -> FDTDResult:
    """FDTD reference for a *periodic* unit cell (Bloch PBC in x, PML in y).

    Requires ``spec.n_ridges == 1`` (a single period).  This is the periodic
    counterpart of :func:`run_grating` and is used to validate the RCWA
    backbone independently of finite-size effects: for ``pad=0`` the RCWA
    supercell is exactly this unit cell.
    """
    import meep

    if spec.n_ridges != 1:
        raise ValueError("periodic reference requires n_ridges == 1")
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")

    if tin_material is None:
        if tin_params is None:
            cache = tin_fit_path or os.path.join(
                _DEFAULT_FIT_DIR, f"tin_fit_{tin_dataset}.json"
            )
            tin_params = get_or_fit(tin_dataset, cache_path=cache)
        from tin_dispersion import build_meep_medium

        tin_material = build_meep_medium(tin_params, meep)

    wl, R, T, meta = _run_once(
        meep, spec, wl_range=wl_range, n_freq=n_freq, angle_deg=0.0, pol=pol,
        resolution=resolution, dpml=dpml, pad=0.0, air_above=air_above,
        sub_depth=sub_depth, tin_material=tin_material, decay_by=decay_by,
        dT=dT, periodic=True,
    )
    return FDTDResult(wl, R, T, 0.0, pol, spec, meta)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse():
    p = argparse.ArgumentParser(description="Finite-grating MEEP R/T spectrum")
    p.add_argument("--n-ridges", type=int, default=10)
    p.add_argument("--period", type=float, default=1.5, help="period (µm)")
    p.add_argument("--amplitude", type=float, default=0.045, help="amplitude (µm)")
    p.add_argument("--base-thickness", type=float, default=0.03, help="uniform TiN (µm)")
    p.add_argument("--substrate-index", type=float, default=1.5)
    p.add_argument("--jitter-std", type=float, default=0.0, help="ridge height jitter std")
    p.add_argument("--n-missing", type=int, default=0, help="number of missing ridges")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wl-range", type=float, nargs=2, default=[0.5, 1.5], help="µm")
    p.add_argument("--n-freq", type=int, default=60)
    p.add_argument("--angle", type=float, default=0.0, help="incidence angle (deg)")
    p.add_argument("--pol", choices=["s", "p"], default="s")
    p.add_argument("--resolution", type=float, default=40.0, help="pixels/µm")
    p.add_argument("--dpml", type=float, default=0.5)
    p.add_argument("--pad", type=float, default=0.6)
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--tin-fit", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main(argv=None):
    a = _parse()
    spec = GratingSpec(
        n_ridges=a.n_ridges,
        period=a.period,
        amplitude=a.amplitude,
        base_thickness=a.base_thickness,
        substrate_index=a.substrate_index,
    )
    if a.jitter_std > 0 or a.n_missing > 0:
        spec = spec.with_defects(
            jitter_std=a.jitter_std,
            n_missing=a.n_missing,
            rng=np.random.default_rng(a.seed),
        )

    result = run_grating(
        spec,
        wl_range=tuple(a.wl_range),
        n_freq=a.n_freq,
        angle_deg=a.angle,
        pol=a.pol,
        resolution=a.resolution,
        dpml=a.dpml,
        pad=a.pad,
        tin_dataset=a.tin_dataset,
        tin_fit_path=a.tin_fit,
        verbose=a.verbose,
    )
    print(f"angle={a.angle} deg  pol={a.pol}  ridges={a.n_ridges}")
    for wl, R, T in zip(result.wl_um, result.R, result.T):
        print(f"  wl={wl:.3f} µm  R={R:.4f}  T={T:.4f}  A={1-R-T:.4f}")
    if a.output:
        result.to_csv(a.output)
        print(f"saved: {a.output}")


if __name__ == "__main__":
    main()
