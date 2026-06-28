#!/usr/bin/env python
"""
populate_abacus_lrg_xirppi.py
=============================

Populate the 25 AbacusSummit base boxes (c000, z = 0.5) with a DESI-LRG HOD
using AbacusHOD, then measure the redshift-space 2PCF xi(rp, pi) on a linear
140 x 140 grid with 1 Mpc/h bins:

    0 <= rp < 140 Mpc/h
    0 <= pi < 140 Mpc/h

Per-phase output:
    xirppi_ph{NNN}.npy          shape (140, 140), axis0 = rp, axis1 = pi

Combined output:
    xirppi_all.npy              shape (Nphase, 140, 140)
    xirppi_mean.npy             shape (140, 140)
    xirppi_deviations.npy       shape (Nphase, 19600)
    rp_edges.npy
    pi_edges.npy

Optional:
    xirppi_cov.npy              shape (19600, 19600), about 3 GB

Usage
-----
  # One phase:
  python populate_abacus_lrg_xirppi.py --phase 0 --skip-prepare

  # All phases serially:
  python populate_abacus_lrg_xirppi.py --phase all --skip-prepare

  # One-time heavy extraction step:
  python populate_abacus_lrg_xirppi.py --phase all --prepare-only

  # Combine per-phase outputs:
  python populate_abacus_lrg_xirppi.py combine

  # Combine and also write the full 19600 x 19600 covariance (~3 GB):
  python populate_abacus_lrg_xirppi.py combine --full-cov

Requirements
------------
abacusutils (abacusnbody), Corrfunc, numpy, pyyaml.
"""

import os
import sys
import time
import glob
import yaml
import argparse
import numpy as np


# -----------------------------------------------------------------------------
# 1. PATHS
# -----------------------------------------------------------------------------

SIM_DIR = "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit"

# Keep the trailing slash. Some AbacusHOD/prepare_sim versions concatenate
# this string directly with sim_name rather than using os.path.join().
SUBSAMPLE_DIR = "/pscratch/sd/l/lado/ZeroCrossing/abacus_subs/"

OUTPUT_DIR = "/pscratch/sd/l/lado/ZeroCrossing/abacus_lrg_mocks"

Z_MOCK = 0.5
BOXSIZE = 2000.0
NPHASES = 25

# Uses the SLURM allocation when available; otherwise defaults to 128.
NTHREAD = int(os.environ.get("SLURM_CPUS_PER_TASK", "128"))


# -----------------------------------------------------------------------------
# 2. LRG HOD PARAMETERS
# -----------------------------------------------------------------------------

LRG_PARAMS = dict(
    logM_cut=12.64,
    logM1=13.71,
    sigma=0.09,
    alpha=1.18,
    kappa=0.60,
    alpha_c=0.19,
    alpha_s=0.95,

    # Satellite-profile flexibility: disabled.
    s=0.0,
    s_v=0.0,
    s_p=0.0,
    s_r=0.0,

    # Assembly bias: disabled.
    Acent=0.0,
    Asat=0.0,
    Bcent=0.0,
    Bsat=0.0,

    # Full completeness for the periodic box.
    ic=0.62,
)


# -----------------------------------------------------------------------------
# 3. CLUSTERING GRID
# -----------------------------------------------------------------------------

RP_EDGES = np.arange(141, dtype=float)
RP_EDGES[0] = 1e-4

PIMAX = 140
PI_BIN_SIZE = 1
PI_EDGES = np.arange(0, PIMAX + PI_BIN_SIZE, PI_BIN_SIZE)

def sim_name_for(ph):
    return f"AbacusSummit_base_c000_ph{ph:03d}"


def build_config(sim_name):
    """Build the AbacusHOD configuration dictionaries for one simulation."""

    return {
        "sim_params": {
            "sim_name": sim_name,
            "sim_dir": SIM_DIR,
            "output_dir": OUTPUT_DIR,
            "subsample_dir": SUBSAMPLE_DIR,
            "z_mock": Z_MOCK,
            "cleaned_halos": True,
        },

        "prepare_sim": {
            "Nparallel_load": 5,
        },

        "HOD_params": {
            "want_ranks": False,
            "want_rsd": True,
            "want_AB": False,
            "Ndim": 1024,
            "density_sigma": 3.0,
            "write_to_disk": False,
            "tracer_flags": {
                "LRG": True,
                "ELG": False,
                "QSO": False,
            },
            "LRG_params": LRG_PARAMS,
        },

        # These are retained because AbacusHOD expects clustering parameters,
        # but the actual xi(rp, pi) measurement below uses Corrfunc directly.
        "clustering_params": {
            "clustering_type": "xirppi",
            "bin_params": {
                "logmin": -1.0,
                "logmax": np.log10(140.0),
                "nbins": 140,
            },
            "pimax": PIMAX,
            "pi_bin_size": PI_BIN_SIZE,
        },
    }


def prepare_phase(sim_name, force=False):
    """
    Run prepare_sim for one phase.

    This is the expensive one-time extraction of halo and particle subsamples.
    """

    from abacusnbody.hod import prepare_sim

    subdir = os.path.join(SUBSAMPLE_DIR, sim_name, f"z{Z_MOCK:.3f}")

    already_prepared = (
        os.path.isdir(subdir)
        and len(os.listdir(subdir)) > 0
    )

    if already_prepared and not force:
        print(f"[prepare] {sim_name}: subsamples already present; skipping",
              flush=True)
        return

    cfg = build_config(sim_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tmp_yaml = os.path.join(OUTPUT_DIR, f"_cfg_{sim_name}.yaml")

    with open(tmp_yaml, "w") as fh:
        yaml.safe_dump(cfg, fh)

    print(f"[prepare] {sim_name}: extracting subsamples...", flush=True)

    t0 = time.time()
    prepare_sim.main(tmp_yaml)

    print(f"[prepare] {sim_name}: finished in {time.time() - t0:.1f} s",
          flush=True)


def compute_xirppi_periodic(mock, rp_edges, pimax, pi_bin_size,
                            boxsize, nthreads):
    """
    Return xi(rp, pi) for rp >= 0 and 0 <= pi < pimax,
    with shape (N_rp, N_pi).

    This Corrfunc build returns signed-pi bins. We request the full
    [-pimax, +pimax] interval at the desired resolution and keep pi >= 0.
    """
    from Corrfunc.theory import DDrppi

    x = np.asarray(mock["LRG"]["x"], dtype=np.float64) % boxsize
    y = np.asarray(mock["LRG"]["y"], dtype=np.float64) % boxsize
    z = np.asarray(mock["LRG"]["z"], dtype=np.float64) % boxsize

    nrpbins = len(rp_edges) - 1

    # Corrfunc returns bins over [-pimax, +pimax].
    # Therefore this must be 2*pimax/dpi, not pimax/dpi.
    npibins_signed = int(round(2.0 * pimax / pi_bin_size))

    if not np.isclose(npibins_signed * pi_bin_size, 2.0 * pimax):
        raise ValueError("2*pimax must be an integer multiple of pi_bin_size")

    print(
        f"[xi] Corrfunc DDrppi: {nrpbins} rp bins x "
        f"{npibins_signed} signed-pi bins; keeping positive half...",
        flush=True,
    )

    result = DDrppi(
        autocorr=1,
        nthreads=nthreads,
        binfile=rp_edges,
        pimax=pimax,
        npibins=npibins_signed,
        X1=x,
        Y1=y,
        Z1=z,
        periodic=True,
        boxsize=boxsize,
    )

    dd_signed = result["npairs"].reshape(nrpbins, npibins_signed).astype(float)

    # Keep pi in [0, 140): 140 bins of width 1 Mpc/h.
    dd = dd_signed[:, npibins_signed // 2:]

    # For one signed-pi half, the volume of each bin is:
    #   dV = pi (rp_hi^2-rp_lo^2) * d_pi
    #
    # Corrfunc auto-pair counts are double-counted, hence N(N-1).
    nobj = len(x)
    volume = boxsize**3

    rp_lo = rp_edges[:-1]
    rp_hi = rp_edges[1:]
    annulus_area = np.pi * (rp_hi**2 - rp_lo**2)

    rr = (
        nobj * (nobj - 1)
        * annulus_area[:, None]
        * pi_bin_size
        / volume
    )

    xi = dd / rr - 1.0
    return xi


def run_phase(ph, skip_prepare=False):
    """Populate one phase, measure xi(rp, pi), and save the result."""

    from abacusnbody.hod.abacus_hod import AbacusHOD

    sim_name = sim_name_for(ph)

    if not skip_prepare:
        prepare_phase(sim_name)

    cfg = build_config(sim_name)

    print(f"[load] {sim_name}: initializing AbacusHOD...", flush=True)

    ball = AbacusHOD(
        cfg["sim_params"],
        cfg["HOD_params"],
        cfg["clustering_params"],
    )

    print(f"[hod] {sim_name}: populating galaxies...", flush=True)

    t0 = time.time()

    mock = ball.run_hod(
        ball.tracers,
        want_rsd=True,
        write_to_disk=False,
        Nthread=NTHREAD,
    )

    ngal = len(mock["LRG"]["x"])
    nbar = ngal / BOXSIZE**3

    print(
        f"[hod] {sim_name}: N_LRG = {ngal:,}; "
        f"nbar = {nbar:.4e} (h/Mpc)^3; "
        f"finished in {time.time() - t0:.1f} s",
        flush=True,
    )

    t0 = time.time()

    xi = compute_xirppi_periodic(
        mock=mock,
        rp_edges=RP_EDGES,
        pimax=PIMAX,
        pi_bin_size=PI_BIN_SIZE,
        boxsize=BOXSIZE,
        nthreads=NTHREAD,
    )

    print(
        f"[xi] {sim_name}: measurement finished in "
        f"{time.time() - t0:.1f} s",
        flush=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outfile = os.path.join(OUTPUT_DIR, f"xirppi_ph{ph:03d}.npy")
    np.save(outfile, xi)

    print(
        f"[xi] {sim_name}: saved {outfile}; shape={xi.shape}",
        flush=True,
    )

    return xi


def combine(write_full_cov=False):
    """
    Combine per-phase xi(rp, pi) arrays.

    With 25 phases, the covariance has rank at most 24. The complete
    19600 x 19600 covariance is about 3 GB in float64, so by default we save
    the mean and deviations rather than materializing it.
    """

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "xirppi_ph*.npy")))

    if not files:
        sys.exit(f"No xirppi_ph*.npy files found in {OUTPUT_DIR}")

    all_xi = np.array([np.load(fname) for fname in files])

    expected_shape = (len(RP_EDGES) - 1, len(PI_EDGES) - 1)

    if all_xi.shape[1:] != expected_shape:
        raise ValueError(
            f"Unexpected xi shape {all_xi.shape[1:]}; "
            f"expected {expected_shape}. "
            "Remove old files from a previous binning scheme before combining."
        )

    nphase = len(all_xi)

    print(
        f"[combine] stacked {nphase} phases; array shape = {all_xi.shape}",
        flush=True,
    )

    xi_mean = all_xi.mean(axis=0)
    flat = all_xi.reshape(nphase, -1)
    deviations = flat - flat.mean(axis=0, keepdims=True)

    np.save(os.path.join(OUTPUT_DIR, "xirppi_all.npy"), all_xi)
    np.save(os.path.join(OUTPUT_DIR, "xirppi_mean.npy"), xi_mean)
    np.save(os.path.join(OUTPUT_DIR, "xirppi_deviations.npy"), deviations)
    np.save(os.path.join(OUTPUT_DIR, "rp_edges.npy"), RP_EDGES)
    np.save(os.path.join(OUTPUT_DIR, "pi_edges.npy"), PI_EDGES)

    print("[combine] wrote xirppi_all, xirppi_mean, xirppi_deviations, "
          "rp_edges, and pi_edges", flush=True)

    if write_full_cov:
        print("[combine] constructing full covariance; this is ~3 GB...",
              flush=True)

        xi_cov = np.cov(flat, rowvar=False)

        np.save(os.path.join(OUTPUT_DIR, "xirppi_cov.npy"), xi_cov)

        print("[combine] wrote xirppi_cov.npy", flush=True)
    else:
        print(
            "[combine] full covariance not written. With 25 phases it has "
            "rank <= 24 and would occupy about 3 GB. Use --full-cov only "
            "if you explicitly need it.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "combine"],
        help="run (default): populate + measure; combine: aggregate phase files.",
    )

    parser.add_argument(
        "--phase",
        default="all",
        help="Phase index 0-24, or 'all' (default).",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run the one-time prepare_sim extraction.",
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume prepared subsamples already exist.",
    )

    parser.add_argument(
        "--full-cov",
        action="store_true",
        help="When combining, write the full 19600 x 19600 covariance.",
    )

    args = parser.parse_args()

    if args.mode == "combine":
        combine(write_full_cov=args.full_cov)
        return

    phases = range(NPHASES) if args.phase == "all" else [int(args.phase)]

    for ph in phases:
        if ph < 0 or ph >= NPHASES:
            raise ValueError(f"Phase must be in [0, {NPHASES - 1}], got {ph}")

    if args.prepare_only:
        for ph in phases:
            prepare_phase(sim_name_for(ph))
        return

    for ph in phases:
        run_phase(ph, skip_prepare=args.skip_prepare)

    if args.phase == "all":
        combine(write_full_cov=args.full_cov)


if __name__ == "__main__":
    main()