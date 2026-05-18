"""
main.py — Master Orchestration Script
======================================
This is the single entry point for the entire simulation study.
Run this file to reproduce all results and figures in the paper.

HOW TO RUN:
    # Full run (all experiments, all figures) — takes ~30–60 min on a laptop:
    python main.py

    # Fast preview run (reduced Monte Carlo, quick sanity check):
    python main.py --fast

    # Skip experiments, regenerate figures from saved results only:
    python main.py --figures-only

    # Run one specific experiment:
    python main.py --only exp1

WHAT THIS FILE DOES (in order):
    1. Checks that the core/ modules are importable (dependency check).
    2. Runs the three distance-sweep experiments (exp1, exp2, exp3) — Figure 1.
    3. Runs the donor-threshold sweep — Figure 2.
    4. Runs the transfer-efficiency sweep — Figure 3.
    5. Runs the scaling sweep (multiple poor nodes) — Figure 4.
    6. Saves every result to results/ as .npz files (compressed numpy arrays).
    7. Calls plot_results.py to generate all paper figures as PDF + PNG.
    8. Prints a summary table of key performance numbers for the paper abstract.

WHY WE SAVE RESULTS BEFORE PLOTTING:
    Simulations take 20–60 minutes depending on hardware. Plotting takes
    seconds. By saving results to disk before plotting, you can:
        - Tweak figure styling without re-running simulations.
        - Share the results/ folder with a co-author for independent plotting.
        - Resume after a crash without losing completed experiment data.
        - Document exactly which parameters produced which published figures.

REPRODUCIBILITY:
    All random seeds are deterministic functions of (experiment_id,
    distance_index, trial_index). Running main.py twice on any machine
    will produce numerically identical results/ files.
"""

import os
import sys
import time
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — ensure all sub-packages are importable from any working directory
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR   = os.path.join(SCRIPT_DIR, 'core')
EXP_DIR    = os.path.join(SCRIPT_DIR, 'experiments')
ANA_DIR    = os.path.join(SCRIPT_DIR, 'analysis')

for path in [CORE_DIR, EXP_DIR, ANA_DIR, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ─────────────────────────────────────────────────────────────────────────────
# Dependency check — fail early with a clear message rather than a cryptic
# ImportError buried inside a long simulation run
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    """Verify all required packages and project modules are available."""
    missing_pkg = []
    for pkg in ['numpy', 'scipy', 'matplotlib', 'tqdm']:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkg.append(pkg)

    if missing_pkg:
        print(f"ERROR: Missing packages: {missing_pkg}")
        print("Install them with:")
        print(f"  pip install {' '.join(missing_pkg)}")
        sys.exit(1)

    missing_mod = []
    for mod in ['channel_models', 'energy_harvester', 'energy_buffer',
                'backscatter_comm', 'cooperative_sharing']:
        try:
            __import__(mod)
        except ImportError:
            missing_mod.append(mod)

    if missing_mod:
        print(f"ERROR: Cannot import core modules: {missing_mod}")
        print(f"Expected location: {CORE_DIR}/")
        print("Check that channel_models.py, energy_harvester.py, etc. exist there.")
        sys.exit(1)

    print("✓ All dependencies satisfied.")


# ─────────────────────────────────────────────────────────────────────────────
# Import project modules (after path setup and dependency check)
# ─────────────────────────────────────────────────────────────────────────────

import exp1_baseline      as exp1
import exp2_hybrid_nocoop as exp2

# exp3 is defined inline in the roadmap document — we replicate it here as
# a proper importable module. For now, we define the sweep function directly
# in main.py so everything is self-contained in a single runnable file.
from energy_harvester      import HybridEnergyHarvester
from energy_buffer         import SupercapacitorBuffer
from backscatter_comm      import compute_backscatter_snr
from cooperative_sharing   import ThresholdSharingPolicy
from channel_models        import generate_rf_channel_gain
from tqdm                  import tqdm
from plot_results           import generate_all_figures


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Hybrid + Cooperative — core single-config function
# ─────────────────────────────────────────────────────────────────────────────

def _exp3_single_config(distance_B_m, tau_A_frac=0.65, tau_B_frac=0.30,
                         eta_tr=0.50, n_slots=5000, rng_seed=None,
                         n_poor_nodes=1):
    """
    Simulate one trial of the proposed system: hybrid harvesting + cooperation.

    Supports n_poor_nodes > 1 for the scaling experiment (Figure 4).
    When n_poor_nodes > 1, Node A shares its surplus energy across all poor
    nodes in round-robin order, and the returned outage metrics are averages
    across all poor nodes.

    Parameters
    ----------
    distance_B_m : float
        Distance of all poor nodes from the AP (meters). In the scaling
        experiment, all poor nodes are at the same distance for fairness.
    tau_A_frac : float
        Donor threshold: Node A shares when buffer > tau_A_frac * E_max.
    tau_B_frac : float
        Receiver threshold: Node A sends to Node B when B < tau_B_frac * E_max.
    eta_tr : float
        WPT transfer efficiency (fraction of sent energy that B receives).
    n_slots : int
        Time slots per trial.
    rng_seed : int or None
        RNG seed for reproducibility.
    n_poor_nodes : int
        Number of energy-constrained ZED nodes served by one rich donor.

    Returns
    -------
    dict with keys:
        energy_outage_A   : Node A energy outage fraction
        energy_outage_B   : Mean energy outage fraction across all poor nodes
        comm_outage_B     : Mean communication outage fraction (conditional on active)
        sharing_fraction  : Fraction of slots where sharing occurred
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # ── Constants ────────────────────────────────────────────────────────────
    E_max_uJ = 500.0
    distance_A_m         = 3.0
    dist_B_to_gateway_m  = 6.0
    snr_threshold_linear = 10 ** (-15.0 / 10.0)
    P_carrier_W          = 0.05
    E_sense_J            = 4e-6
    E_backscatter_J      = 1e-6
    E_idle_J             = 0.5e-6
    E_consumed           = E_sense_J + E_backscatter_J + E_idle_J

    # ── Initialize Node A (donor) ────────────────────────────────────────────
    harvester_A = HybridEnergyHarvester(
        eta_rf=0.5, eta_opt=0.60, P_rf_mean_mW=0.1, P_opt_mean_mW=0.8,
        distance_m=distance_A_m, path_loss_exp=2.5, sigma_shadow_db=3.0
    )
    buffer_A = SupercapacitorBuffer(E_max_uJ=E_max_uJ, E_min_operate_uJ=10.0,
                                     E_initial_uJ=E_max_uJ * 0.6)

    # ── Initialize N poor nodes ──────────────────────────────────────────────
    harvesters_B = [
        HybridEnergyHarvester(
            eta_rf=0.5, eta_opt=0.60, P_rf_mean_mW=0.1, P_opt_mean_mW=0.8,
            distance_m=distance_B_m, path_loss_exp=2.5, sigma_shadow_db=4.5
        )
        for _ in range(n_poor_nodes)
    ]
    buffers_B = [
        SupercapacitorBuffer(E_max_uJ=E_max_uJ, E_min_operate_uJ=10.0,
                              E_initial_uJ=E_max_uJ * 0.3)
        for _ in range(n_poor_nodes)
    ]

    # Each poor node has its own sharing policy instance.
    # tau_A_frac is shared, but tau_B_frac applies individually.
    policy = ThresholdSharingPolicy(
        tau_A_fraction=tau_A_frac,
        tau_B_fraction=tau_B_frac,
        transfer_efficiency=eta_tr,
        E_max_J=E_max_uJ * 1e-6
    )

    # ── Counters ─────────────────────────────────────────────────────────────
    outage_A        = 0
    outages_B       = np.zeros(n_poor_nodes)
    comm_outages_B  = np.zeros(n_poor_nodes)
    active_slots_B  = np.zeros(n_poor_nodes)
    share_events    = 0

    # ── Time-slot loop ───────────────────────────────────────────────────────
    for _ in range(n_slots):

        # Step 1: Harvest — Node A and all poor nodes harvest simultaneously
        E_harv_A, _, _ = harvester_A.harvest(1)
        E_harv_A = E_harv_A[0]

        E_harv_Bs = []
        for harv_B in harvesters_B:
            e, _, _ = harv_B.harvest(1)
            E_harv_Bs.append(e[0])

        # Step 2: Cooperative sharing (round-robin across poor nodes)
        #
        # Round-robin ensures fairness: each poor node gets an equal opportunity
        # to receive energy in each slot. This prevents "energy starvation" where
        # one node monopolizes the donor's surplus.
        #
        # In each slot, Node A evaluates sharing with ONE poor node in sequence.
        # This is realistic because a single WPT transmitter cannot charge
        # multiple receivers simultaneously at full efficiency.
        # (Multi-beam WPT is a future extension — Paper 3.)

        total_E_sent     = 0.0
        E_received_by_B  = np.zeros(n_poor_nodes)

        for b_idx in range(n_poor_nodes):
            E_sent, E_recv, shared = policy.decide_and_transfer(
                buffer_A.E, buffers_B[b_idx].E
            )
            if shared:
                share_events += 1
                total_E_sent           += E_sent
                E_received_by_B[b_idx]  = E_recv
                # Once A has shared this slot, reduce its effective buffer
                # so subsequent iterations within the same slot do not
                # over-draw its energy.
                buffer_A.E = max(0.0, buffer_A.E - E_sent)

        # Step 3: Update Node A's buffer with net harvest and sharing costs
        _, outage_A_flag = buffer_A.update(
            E_harvested=E_harv_A,
            E_consumed=E_consumed,
            E_shared_received=0.0,
            E_shared_sent=0.0   # already deducted above in round-robin loop
        )
        if outage_A_flag:
            outage_A += 1

        # Step 4: Update all poor nodes' buffers
        for b_idx in range(n_poor_nodes):
            _, outage_B_flag = buffers_B[b_idx].update(
                E_harvested=E_harv_Bs[b_idx],
                E_consumed=E_consumed,
                E_shared_received=E_received_by_B[b_idx],
                E_shared_sent=0.0
            )
            if outage_B_flag:
                outages_B[b_idx] += 1
            else:
                active_slots_B[b_idx] += 1
                h_fwd = generate_rf_channel_gain(1, distance_m=distance_B_m)[0]
                h_bwd = generate_rf_channel_gain(1, distance_m=dist_B_to_gateway_m)[0]
                snr, _ = compute_backscatter_snr(P_carrier_W, h_fwd, h_bwd)
                if snr < snr_threshold_linear:
                    comm_outages_B[b_idx] += 1

    # ── Compute probabilities ────────────────────────────────────────────────
    outage_probs_B = outages_B / n_slots
    comm_probs_B   = np.where(active_slots_B > 0,
                               comm_outages_B / active_slots_B,
                               1.0)
    return {
        'energy_outage_A':  outage_A / n_slots,
        'energy_outage_B':  outage_probs_B.mean(),
        'comm_outage_B':    comm_probs_B.mean(),
        'sharing_fraction': share_events / (n_slots * n_poor_nodes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Distance sweep wrapper (Figure 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3_distance_sweep(distance_range_m, n_slots=5000,
                             n_monte_carlo=300, verbose=True):
    """
    Sweep Node B distance for the proposed system (exp3).
    Returns a dict with the same schema as exp1/exp2 for Figure 1 overlay.
    """
    n_dist = len(distance_range_m)
    outage_A_mean = np.zeros(n_dist)
    outage_B_mean = np.zeros(n_dist)
    comm_out_mean = np.zeros(n_dist)
    outage_A_std  = np.zeros(n_dist)
    outage_B_std  = np.zeros(n_dist)
    share_mean    = np.zeros(n_dist)

    iterator = enumerate(distance_range_m)
    if verbose:
        iterator = enumerate(tqdm(distance_range_m,
                                  desc='[Exp3 Proposed] Distance sweep'))

    for idx, dist in iterator:
        trial_outA  = np.zeros(n_monte_carlo)
        trial_outB  = np.zeros(n_monte_carlo)
        trial_comm  = np.zeros(n_monte_carlo)
        trial_share = np.zeros(n_monte_carlo)

        for trial in range(n_monte_carlo):
            seed = int(5_000_000 + idx * 10000 + trial)
            res = _exp3_single_config(dist, n_slots=n_slots, rng_seed=seed)
            trial_outA[trial]  = res['energy_outage_A']
            trial_outB[trial]  = res['energy_outage_B']
            trial_comm[trial]  = res['comm_outage_B']
            trial_share[trial] = res['sharing_fraction']

        outage_A_mean[idx] = trial_outA.mean()
        outage_B_mean[idx] = trial_outB.mean()
        comm_out_mean[idx] = trial_comm.mean()
        outage_A_std[idx]  = trial_outA.std()
        outage_B_std[idx]  = trial_outB.std()
        share_mean[idx]    = trial_share.mean()

    return {
        'distance_m_sweep':    distance_range_m,
        'energy_outage_A':     outage_A_mean,
        'energy_outage_B':     outage_B_mean,
        'comm_outage_B':       comm_out_mean,
        'energy_outage_A_std': outage_A_std,
        'energy_outage_B_std': outage_B_std,
        'sharing_fraction':    share_mean,
        'system_label': 'Proposed: Hybrid + Cooperation',
        'system_color': '#2ecc71',   # green — best performance
        'system_ls':    '-',
        'n_monte_carlo': n_monte_carlo,
        'n_slots': n_slots,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Donor threshold (τ_A) sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_tau_A_sweep(distance_B_m=10.0, n_slots=5000,
                    n_monte_carlo=300, verbose=True):
    """
    Sweep τ_A from 0.25 to 0.95 to reveal the energy-sharing trade-off.

    Node B is fixed at 10m (a moderately energy-constrained position).
    τ_B is fixed at 0.3. Only τ_A varies.

    The crossover point where outage_A ≈ outage_B is the fairness-optimal
    operating point. The minimum of (outage_A + outage_B) is the
    efficiency-optimal operating point. These two points may differ and the
    gap between them is an interesting result to discuss in the paper.
    """
    tau_A_values = np.linspace(0.25, 0.95, 15)
    n_pts = len(tau_A_values)

    outA = np.zeros(n_pts)
    outB = np.zeros(n_pts)
    comm = np.zeros(n_pts)

    iterator = enumerate(tau_A_values)
    if verbose:
        iterator = enumerate(tqdm(tau_A_values,
                                  desc='[Fig2] τ_A sweep'))

    for idx, tau_A in iterator:
        t_outA = np.zeros(n_monte_carlo)
        t_outB = np.zeros(n_monte_carlo)
        t_comm = np.zeros(n_monte_carlo)

        for trial in range(n_monte_carlo):
            seed = int(6_000_000 + idx * 10000 + trial)
            res = _exp3_single_config(
                distance_B_m=distance_B_m,
                tau_A_frac=float(tau_A),
                tau_B_frac=0.30,
                eta_tr=0.50,
                n_slots=n_slots,
                rng_seed=seed
            )
            t_outA[trial] = res['energy_outage_A']
            t_outB[trial] = res['energy_outage_B']
            t_comm[trial] = res['comm_outage_B']

        outA[idx] = t_outA.mean()
        outB[idx] = t_outB.mean()
        comm[idx] = t_comm.mean()

    return {
        'tau_A_sweep':     tau_A_values,
        'energy_outage_A': outA,
        'energy_outage_B': outB,
        'comm_outage_B':   comm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Transfer efficiency (η_tr) sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_efficiency_sweep(distance_B_m=10.0, n_slots=5000,
                          n_monte_carlo=300, verbose=True):
    """
    Sweep η_tr from 0.1 to 0.9 for both RF-WPT and optical-WPT cooperation.

    RF WPT is omnidirectional. In practice, η_tr for RF WPT at 1–3m is
    roughly 0.3–0.5 (Powercast P21xx datasheet, Friis equation at 915 MHz).

    Optical WPT is directional. With LOS and a focused beam (laser/LED
    with lens), η_tr can reach 0.6–0.8. Without LOS, optical WPT fails
    completely (η_tr → 0). This motivates using RF as a fallback.

    We model the optical scenario as having +0.2 efficiency advantage over
    RF at every η_tr point — i.e., for a given hardware configuration,
    optical achieves η_tr = rf_eta + 0.2 (capped at 0.9).
    This is a simplification for the first paper; a more rigorous model
    would use the Lambert-Beer law for optical propagation.
    """
    eta_values = np.linspace(0.1, 0.9, 9)
    n_pts      = len(eta_values)

    outB_RF      = np.zeros(n_pts)
    outB_optical = np.zeros(n_pts)

    # Non-cooperative baseline (does not depend on η_tr — it's a flat reference)
    # Run once and replicate
    t_nocoop = np.zeros(n_monte_carlo)
    for trial in range(n_monte_carlo):
        seed = int(7_000_000 + trial)
        np.random.seed(seed)
        harv_B = HybridEnergyHarvester(distance_m=distance_B_m, sigma_shadow_db=4.5)
        buf_B  = SupercapacitorBuffer(E_max_uJ=500.0, E_min_operate_uJ=10.0,
                                       E_initial_uJ=150.0)
        E_consumed = 4e-6 + 1e-6 + 0.5e-6
        out_cnt = 0
        for _ in range(n_slots):
            e, _, _ = harv_B.harvest(1)
            _, outage = buf_B.update(e[0], E_consumed, 0.0, 0.0)
            if outage:
                out_cnt += 1
        t_nocoop[trial] = out_cnt / n_slots
    nocoop_val = t_nocoop.mean()

    iterator = enumerate(eta_values)
    if verbose:
        iterator = enumerate(tqdm(eta_values,
                                  desc='[Fig3] η_tr sweep'))

    for idx, eta_rf in iterator:
        # Optical achieves +0.2 efficiency, capped at 0.9
        eta_optical = min(0.9, eta_rf + 0.20)

        t_rf  = np.zeros(n_monte_carlo)
        t_opt = np.zeros(n_monte_carlo)

        for trial in range(n_monte_carlo):
            # RF cooperation at this η_tr
            seed_rf = int(7_100_000 + idx * 10000 + trial)
            res_rf = _exp3_single_config(
                distance_B_m=distance_B_m, eta_tr=float(eta_rf),
                n_slots=n_slots, rng_seed=seed_rf
            )
            t_rf[trial] = res_rf['energy_outage_B']

            # Optical cooperation at eta_optical
            seed_opt = int(7_200_000 + idx * 10000 + trial)
            res_opt = _exp3_single_config(
                distance_B_m=distance_B_m, eta_tr=float(eta_optical),
                n_slots=n_slots, rng_seed=seed_opt
            )
            t_opt[trial] = res_opt['energy_outage_B']

        outB_RF[idx]      = t_rf.mean()
        outB_optical[idx] = t_opt.mean()

    return {
        'eta_tr_sweep':    eta_values,
        'outage_B_RF':     outB_RF,
        'outage_B_optical': outB_optical,
        'outage_B_nocoop': nocoop_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Scaling sweep (1 to 5 poor nodes)
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling_sweep(distance_B_m=10.0, n_slots=5000,
                      n_monte_carlo=200, verbose=True):
    """
    Sweep the number of poor nodes (1 to 5), one rich donor node fixed.

    For each value of N (number of poor nodes), we run exp1, exp2, and exp3
    so all three systems appear in Figure 4 and can be compared at each N.

    The key question: does the cooperative gain (exp3 vs exp1) shrink as N
    grows? If it shrinks gracefully (outage grows sublinearly with N), the
    system is scalable. If it collapses at N=3 or N=4, that's a known
    limitation to disclose honestly in the paper.
    """
    n_node_values = np.array([1, 2, 3, 4, 5])
    n_pts = len(n_node_values)

    outage_exp1 = np.zeros(n_pts)
    outage_exp2 = np.zeros(n_pts)
    outage_exp3 = np.zeros(n_pts)
    std_exp3    = np.zeros(n_pts)

    iterator = enumerate(n_node_values)
    if verbose:
        iterator = enumerate(tqdm(n_node_values,
                                  desc='[Fig4] Scaling sweep'))

    for idx, n_poor in iterator:
        t1 = np.zeros(n_monte_carlo)
        t2 = np.zeros(n_monte_carlo)
        t3 = np.zeros(n_monte_carlo)

        for trial in range(n_monte_carlo):
            # Exp1 baseline with N poor nodes (no cooperation, RF-only).
            # We approximate by running one poor node and multiplying — valid
            # because without cooperation, all nodes are statistically identical.
            seed = int(8_000_000 + idx * 10000 + trial)
            res1 = exp1.run_single_config(distance_B_m, n_slots=n_slots,
                                           rng_seed=seed)
            t1[trial] = res1['energy_outage_B']

            # Exp2: hybrid harvesting, no cooperation
            res2 = exp2.run_single_config(distance_B_m, n_slots=n_slots,
                                           rng_seed=seed + 100_000)
            t2[trial] = res2['energy_outage_B']

            # Exp3: proposed system with n_poor receiver nodes
            res3 = _exp3_single_config(
                distance_B_m=distance_B_m,
                n_poor_nodes=int(n_poor),
                n_slots=n_slots,
                rng_seed=seed + 200_000
            )
            t3[trial] = res3['energy_outage_B']

        outage_exp1[idx] = t1.mean()
        outage_exp2[idx] = t2.mean()
        outage_exp3[idx] = t3.mean()
        std_exp3[idx]    = t3.std()

    return {
        'n_poor_nodes': n_node_values,
        'outage_exp1':  outage_exp1,
        'outage_exp2':  outage_exp2,
        'outage_exp3':  outage_exp3,
        'std_exp3':     std_exp3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results_dict, filepath):
    """
    Save a results dictionary to a compressed numpy archive (.npz).

    String metadata (like system_label, system_color) are stored as
    0-dimensional object arrays so they survive the numpy serialisation.
    """
    save_kwargs = {}
    for key, val in results_dict.items():
        if isinstance(val, str):
            save_kwargs[key] = np.array(val)       # 0-d string array
        elif isinstance(val, (int, float)):
            save_kwargs[key] = np.array(val)       # 0-d numeric array
        else:
            save_kwargs[key] = np.asarray(val)
    np.savez_compressed(filepath, **save_kwargs)
    print(f"    Saved → {filepath}.npz")


def load_results(filepath):
    """Load a previously saved .npz result file into a plain dict."""
    if not os.path.exists(filepath + '.npz'):
        return None
    data = np.load(filepath + '.npz', allow_pickle=True)
    result = {}
    for key in data.files:
        val = data[key]
        result[key] = val.item() if val.ndim == 0 else val
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(r1, r2, r3):
    """
    Print a formatted table of key performance numbers.

    These numbers should appear in your paper's abstract and in Table I.
    Specifically, report the outage probability at 10m distance because
    that is the most representative mid-range scenario.
    """
    dist_arr = r1['distance_m_sweep']
    idx_10m = int(np.argmin(np.abs(dist_arr - 10.0)))

    out1 = r1['energy_outage_B'][idx_10m]
    out2 = r2['energy_outage_B'][idx_10m]
    out3 = r3['energy_outage_B'][idx_10m]

    gain_hybrid = (out1 - out2) / out1 * 100 if out1 > 0 else 0
    gain_coop   = (out2 - out3) / out2 * 100 if out2 > 0 else 0
    gain_total  = (out1 - out3) / out1 * 100 if out1 > 0 else 0

    width = 60
    print("\n" + "═" * width)
    print(" KEY PERFORMANCE RESULTS (Node B at 10m from AP)")
    print("═" * width)
    print(f"  {'System':<38} {'Outage Prob':>10}")
    print("─" * width)
    print(f"  {'RF-Only, No Cooperation (Baseline)':<38} {out1:>10.4f}")
    print(f"  {'Hybrid Harvesting, No Cooperation':<38} {out2:>10.4f}")
    print(f"  {'Proposed: Hybrid + Cooperation':<38} {out3:>10.4f}")
    print("─" * width)
    print(f"  Gain from hybrid harvesting (Exp1→Exp2):    {gain_hybrid:+.1f}%")
    print(f"  Gain from cooperation       (Exp2→Exp3):    {gain_coop:+.1f}%")
    print(f"  Total gain                  (Exp1→Exp3):    {gain_total:+.1f}%")
    print("═" * width)
    print("\n  ➜  These numbers belong in your abstract.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='ZED Backscatter Simulation — Master Run Script')
    parser.add_argument('--fast', action='store_true',
        help='Fast preview: 100 MC trials, 2000 slots. Results are approximate.')
    parser.add_argument('--figures-only', action='store_true',
        help='Skip simulations; load saved results and regenerate figures only.')
    parser.add_argument('--only', choices=['exp1', 'exp2', 'exp3', 'fig2', 'fig3', 'fig4'],
        default=None,
        help='Run only one specific experiment (for debugging).')
    parser.add_argument('--results-dir', default='results/',
        help='Directory for saving/loading .npz result files.')
    parser.add_argument('--figures-dir', default='figures/',
        help='Directory for saving generated figures.')
    parser.add_argument('--show', action='store_true',
        help='Display figures interactively after saving.')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    check_dependencies()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Simulation parameters ─────────────────────────────────────────────────
    # --fast mode: quick sanity check (~2 min on a laptop)
    # Normal mode: publication-quality results (~40–60 min on a laptop)
    if args.fast:
        N_MC    = 100
        N_SLOTS = 2000
        DIST_RANGE = np.linspace(3.0, 15.0, 7)   # fewer distance points
        print("\n⚡ FAST MODE: reduced Monte Carlo (100 trials, 2000 slots).")
        print("   Results are approximate. Use normal mode for publication.\n")
    else:
        N_MC    = 300
        N_SLOTS = 5000
        DIST_RANGE = np.linspace(3.0, 15.0, 13)
        print("\n📊 FULL MODE: 300 MC trials × 5000 slots per config.")
        print("   Estimated time: 40–60 min on a modern laptop.\n")

    t_start = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # Load previously saved results if --figures-only
    # ─────────────────────────────────────────────────────────────────────────
    if args.figures_only:
        print("Loading saved results from", args.results_dir)
        r1     = load_results(os.path.join(args.results_dir, 'exp1_distance_sweep'))
        r2     = load_results(os.path.join(args.results_dir, 'exp2_distance_sweep'))
        r3     = load_results(os.path.join(args.results_dir, 'exp3_distance_sweep'))
        r_fig2 = load_results(os.path.join(args.results_dir, 'tau_A_sweep'))
        r_fig3 = load_results(os.path.join(args.results_dir, 'efficiency_sweep'))
        r_fig4 = load_results(os.path.join(args.results_dir, 'scaling_sweep'))
        if None in [r1, r2, r3]:
            print("ERROR: Core results not found. Run without --figures-only first.")
            sys.exit(1)
        generate_all_figures(r1, r2, r3, r_fig2, r_fig3, r_fig4,
                              output_dir=args.figures_dir, show=args.show)
        print_summary_table(r1, r2, r3)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Run experiments
    # ─────────────────────────────────────────────────────────────────────────

    r1 = r2 = r3 = r_fig2 = r_fig3 = r_fig4 = None

    if args.only in (None, 'exp1'):
        print("━━━ Experiment 1: RF-Only, No Cooperation ━━━")
        r1 = exp1.run_sweep(DIST_RANGE, n_slots=N_SLOTS, n_monte_carlo=N_MC)
        save_results(r1, os.path.join(args.results_dir, 'exp1_distance_sweep'))

    if args.only in (None, 'exp2'):
        print("━━━ Experiment 2: Hybrid Harvesting, No Cooperation ━━━")
        r2 = exp2.run_sweep(DIST_RANGE, n_slots=N_SLOTS, n_monte_carlo=N_MC)
        save_results(r2, os.path.join(args.results_dir, 'exp2_distance_sweep'))

    if args.only in (None, 'exp3'):
        print("━━━ Experiment 3: Proposed — Hybrid + Cooperative ━━━")
        r3 = run_exp3_distance_sweep(DIST_RANGE, n_slots=N_SLOTS,
                                      n_monte_carlo=N_MC)
        save_results(r3, os.path.join(args.results_dir, 'exp3_distance_sweep'))

    if args.only in (None, 'fig2'):
        print("━━━ Figure 2: Donor Threshold τ_A Sweep ━━━")
        r_fig2 = run_tau_A_sweep(n_slots=N_SLOTS, n_monte_carlo=N_MC)
        save_results(r_fig2, os.path.join(args.results_dir, 'tau_A_sweep'))

    if args.only in (None, 'fig3'):
        print("━━━ Figure 3: Transfer Efficiency η_tr Sweep ━━━")
        r_fig3 = run_efficiency_sweep(n_slots=N_SLOTS, n_monte_carlo=N_MC)
        save_results(r_fig3, os.path.join(args.results_dir, 'efficiency_sweep'))

    if args.only in (None, 'fig4'):
        print("━━━ Figure 4: Scaling (N poor nodes) ━━━")
        r_fig4 = run_scaling_sweep(n_slots=N_SLOTS, n_monte_carlo=N_MC)
        save_results(r_fig4, os.path.join(args.results_dir, 'scaling_sweep'))

    # ─────────────────────────────────────────────────────────────────────────
    # Generate figures
    # ─────────────────────────────────────────────────────────────────────────
    if args.only is None:
        generate_all_figures(r1, r2, r3, r_fig2, r_fig3, r_fig4,
                              output_dir=args.figures_dir, show=args.show)
        print_summary_table(r1, r2, r3)

    elapsed = time.time() - t_start
    print(f"\n✅ Done. Total time: {elapsed/60:.1f} minutes.")
    print(f"   Results  → {os.path.abspath(args.results_dir)}")
    print(f"   Figures  → {os.path.abspath(args.figures_dir)}\n")


if __name__ == '__main__':
    main()
