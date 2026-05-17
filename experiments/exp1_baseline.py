"""
exp1_baseline.py — Experiment 1: RF-Only Harvesting, No Cooperation
====================================================================
This is the strictest baseline. Both ZED nodes harvest energy only from
RF signals (no optical source) and operate in complete isolation (no
energy sharing). This reflects the current state of most deployed IoT
backscatter systems in the literature.

WHY THIS BASELINE MATTERS:
    The gap between this curve and exp3's curve in the final plots directly
    quantifies the *combined* benefit of (a) adding optical harvesting and
    (b) enabling cooperative energy sharing. Without this baseline, you
    cannot make the claim that your proposed system improves on the
    state-of-the-art.

WHAT WE SWEEP:
    The primary sweep is Node B's distance from the hybrid access point
    (3m to 15m). As distance increases, path loss degrades RF harvesting
    and Node B's energy outage worsens. This creates a natural x-axis for
    comparing all three systems on the same figure.

OUTPUTS (dictionary keys):
    distance_m_sweep    : array of distances swept (the x-axis for Figure 1)
    energy_outage_A     : energy outage probability of Node A at each distance
    energy_outage_B     : energy outage probability of Node B at each distance
    comm_outage_B       : communication outage probability of Node B
    system_label        : string label for plot legends
    system_color        : matplotlib color string for consistent styling
"""

import numpy as np
import sys
import os
from tqdm import tqdm

# Allow imports from the core/ directory regardless of where this script
# is called from. os.path manipulation keeps this robust across OSes.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from channel_models import generate_rf_channel_gain
from energy_buffer import SupercapacitorBuffer
from backscatter_comm import compute_backscatter_snr


# ─────────────────────────────────────────────────────────────────────────────
# RF-Only Harvester (defined here rather than importing HybridEnergyHarvester
# because this experiment intentionally excludes the optical component.
# Using a separate class makes the experiment's intent self-documenting.)
# ─────────────────────────────────────────────────────────────────────────────

class RFOnlyHarvester:
    """
    Models a node that harvests energy exclusively from ambient/dedicated RF signals.

    This represents the conventional single-source harvesting architecture
    described in references [1–5] of the research plan — the baseline that
    hybrid harvesting aims to improve upon.

    The harvested energy per slot is:
        E_RF(t) = η_rf * P_rf * h_i * T_slot

    where h_i ~ Exp(d^{-α}) captures Rayleigh fading with distance-dependent
    mean path loss.
    """

    def __init__(self, eta_rf=0.5, P_rf_W=0.1e-3, distance_m=5.0,
                 path_loss_exp=2.5, T_slot_s=1e-3):
        """
        Parameters
        ----------
        eta_rf : float
            RF-to-DC conversion efficiency of the rectenna circuit.
            0.5 is a realistic value for a well-matched rectenna at 915 MHz
            (Powercast P2110B achieves ~40–55% depending on input power level).
        P_rf_W : float
            Mean available RF carrier power at 1m reference distance (Watts).
            0.1 mW is consistent with a typical indoor access point at range.
        distance_m : float
            Distance from the RF source to the harvesting node (meters).
        path_loss_exp : float
            Path loss exponent α. Values: 2.0 (free space), 2.5–3.5 (indoor).
        T_slot_s : float
            Duration of one time slot in seconds. 1ms gives energy in Joules
            consistent with a 1 kHz duty-cycle IoT node.
        """
        self.eta_rf = eta_rf
        self.P_rf = P_rf_W
        self.distance_m = distance_m
        self.path_loss_exp = path_loss_exp
        self.T_slot = T_slot_s

    def harvest(self, n_samples=1):
        """
        Draw n_samples independent realizations of harvested energy.

        Returns
        -------
        E_rf : np.ndarray, shape (n_samples,)
            Harvested energy in Joules. Each element represents one time slot.
        """
        h = generate_rf_channel_gain(
            n_samples,
            path_loss_exponent=self.path_loss_exp,
            distance_m=self.distance_m
        )
        E_rf = self.eta_rf * self.P_rf * h * self.T_slot
        return E_rf


# ─────────────────────────────────────────────────────────────────────────────
# Single-Configuration Trial
# ─────────────────────────────────────────────────────────────────────────────

def run_single_config(distance_B_m, n_slots=5000, rng_seed=None):
    """
    Simulate one configuration of the RF-only, non-cooperative system.

    Node A is fixed at 3m from the AP (energy-rich under RF-only conditions).
    Node B is at distance_B_m (the parameter being swept).
    Neither node shares energy. Both attempt backscatter every non-outage slot.

    Parameters
    ----------
    distance_B_m : float
        Distance of Node B from the hybrid access point (meters).
    n_slots : int
        Number of time slots to simulate (one realization of the Markov chain).
    rng_seed : int or None
        Random seed for reproducibility. Set per Monte Carlo trial in the
        outer loop to get independent but reproducible trials.

    Returns
    -------
    dict with keys:
        energy_outage_A  : fraction of slots where Node A was in energy outage
        energy_outage_B  : fraction of slots where Node B was in energy outage
        comm_outage_B    : fraction of active slots where Node B's SNR < threshold
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # ── Hardware parameters ──────────────────────────────────────────────────
    # Node A is closer to the AP and has better RF coverage.
    distance_A_m = 3.0
    dist_B_to_gateway_m = 6.0      # Node B to IoT gateway distance (backscatter path)
    dist_A_to_gateway_m = 4.0      # Node A to gateway

    # Energy cost per slot — calibrated to ultra-low-power IoT hardware:
    #   Sensing: ~2–5 µJ for a TI HDC1080 humidity sensor at 3.3V
    #   Backscatter modulation: ~0.5–2 µJ (impedance switching only)
    #   Idle/control logic: ~0.5 µJ
    E_sense_J = 4e-6        # 4 µJ sensing event
    E_backscatter_J = 1e-6  # 1 µJ for backscatter impedance modulation
    E_idle_J = 0.5e-6       # 0.5 µJ for control logic even when not transmitting

    # SNR threshold for backscatter decoding: -15 dB is achievable with
    # coherent BPSK backscatter and a sensitive gateway receiver.
    snr_threshold_linear = 10 ** (-15.0 / 10.0)

    # RF carrier power available for backscatter reflection (from the AP).
    # This is the power of the signal *arriving at the node*, not the AP's
    # transmit power. 100 mW EIRP at 5m gives roughly this received level.
    P_carrier_W = 0.05  # 50 mW effective carrier power at node

    # ── Component initialization ─────────────────────────────────────────────
    harvester_A = RFOnlyHarvester(distance_m=distance_A_m)
    harvester_B = RFOnlyHarvester(distance_m=distance_B_m)

    # Both nodes start at half capacity — a neutral, unbiased initial condition.
    # The Markov chain reaches its stationary distribution well within n_slots,
    # so initial conditions only marginally affect long-run outage probabilities.
    buffer_A = SupercapacitorBuffer(E_max_uJ=500.0, E_min_operate_uJ=10.0,
                                     E_initial_uJ=250.0)
    buffer_B = SupercapacitorBuffer(E_max_uJ=500.0, E_min_operate_uJ=10.0,
                                     E_initial_uJ=250.0)

    # ── Counters ─────────────────────────────────────────────────────────────
    outage_count_A = 0
    outage_count_B = 0
    comm_outage_count_B = 0
    active_slots_B = 0  # slots where B had enough energy to attempt backscatter

    # ── Time-slot loop ───────────────────────────────────────────────────────
    for _ in range(n_slots):
        # Step 1: Each node harvests RF energy independently this slot.
        E_harv_A = harvester_A.harvest(n_samples=1)[0]
        E_harv_B = harvester_B.harvest(n_samples=1)[0]

        # Step 2: Both nodes try to sense and backscatter.
        # Energy consumed = sensing + backscatter modulation + idle overhead.
        E_consumed = E_sense_J + E_backscatter_J + E_idle_J

        # Step 3: Update energy buffers.
        # No sharing: E_shared_received = 0, E_shared_sent = 0 for both.
        _, outage_A = buffer_A.update(
            E_harvested=E_harv_A,
            E_consumed=E_consumed,
            E_shared_received=0.0,
            E_shared_sent=0.0
        )
        _, outage_B = buffer_B.update(
            E_harvested=E_harv_B,
            E_consumed=E_consumed,
            E_shared_received=0.0,
            E_shared_sent=0.0
        )

        if outage_A:
            outage_count_A += 1
        if outage_B:
            outage_count_B += 1

        # Step 4: If Node B is not in outage, check whether its backscatter
        # signal reaches the gateway with sufficient SNR.
        # The forward channel is AP→B, the backward channel is B→gateway.
        if not outage_B:
            active_slots_B += 1
            h_fwd = generate_rf_channel_gain(1, distance_m=distance_B_m)[0]
            h_bwd = generate_rf_channel_gain(1, distance_m=dist_B_to_gateway_m)[0]
            snr, _ = compute_backscatter_snr(P_carrier_W, h_fwd, h_bwd)
            if snr < snr_threshold_linear:
                comm_outage_count_B += 1

    # ── Compute outage probabilities ─────────────────────────────────────────
    # Communication outage is conditioned on Node B being awake (not in energy
    # outage). A node that is in energy outage trivially cannot communicate,
    # but we track this separately so the two failure modes are distinguishable
    # in the analysis.
    comm_outage_prob_B = (comm_outage_count_B / active_slots_B
                          if active_slots_B > 0 else 1.0)

    return {
        'energy_outage_A': outage_count_A / n_slots,
        'energy_outage_B': outage_count_B / n_slots,
        'comm_outage_B':   comm_outage_prob_B,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distance Sweep (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(distance_range_m=None, n_slots=5000, n_monte_carlo=300,
              verbose=True):
    """
    Sweep Node B's distance and compute mean outage probabilities via
    Monte Carlo averaging.

    Monte Carlo averaging is necessary because the stochastic energy
    process has high variance for a single n_slots run. By running
    n_monte_carlo independent trials and averaging, we obtain a stable
    estimate of the true (stationary) outage probability.

    Parameters
    ----------
    distance_range_m : array-like or None
        Distances to sweep (meters). Defaults to np.linspace(3, 15, 13).
    n_slots : int
        Slots per trial. 5000 slots = 5 seconds at 1ms slot duration —
        long enough for the Markov chain to reach stationarity.
    n_monte_carlo : int
        Number of independent trials per distance point. 300 gives a
        95% confidence interval half-width of roughly ±0.03 on outage
        probability (by the CLT), which is tight enough for a figure.
    verbose : bool
        Whether to print a progress bar.

    Returns
    -------
    dict — compatible with plot_results.py's expected schema
    """
    if distance_range_m is None:
        distance_range_m = np.linspace(3.0, 15.0, 13)

    distance_range_m = np.asarray(distance_range_m, dtype=float)
    n_dist = len(distance_range_m)

    # Pre-allocate result arrays
    outage_A_mean  = np.zeros(n_dist)
    outage_B_mean  = np.zeros(n_dist)
    comm_out_mean  = np.zeros(n_dist)
    outage_A_std   = np.zeros(n_dist)
    outage_B_std   = np.zeros(n_dist)

    iterator = enumerate(distance_range_m)
    if verbose:
        iterator = enumerate(tqdm(distance_range_m,
                                  desc='[Exp1 RF-Only] Distance sweep'))

    for idx, dist in iterator:
        trial_outA = np.zeros(n_monte_carlo)
        trial_outB = np.zeros(n_monte_carlo)
        trial_comm = np.zeros(n_monte_carlo)

        for trial in range(n_monte_carlo):
            # Use a unique seed per (distance_index, trial) to ensure every
            # trial is statistically independent while remaining reproducible.
            seed = int(idx * 10000 + trial)
            res = run_single_config(dist, n_slots=n_slots, rng_seed=seed)
            trial_outA[trial] = res['energy_outage_A']
            trial_outB[trial] = res['energy_outage_B']
            trial_comm[trial] = res['comm_outage_B']

        outage_A_mean[idx] = trial_outA.mean()
        outage_B_mean[idx] = trial_outB.mean()
        comm_out_mean[idx] = trial_comm.mean()
        outage_A_std[idx]  = trial_outA.std()
        outage_B_std[idx]  = trial_outB.std()

    return {
        # ── Sweep axis (x-axis for Figure 1) ─────────────────────────────────
        'distance_m_sweep':   distance_range_m,

        # ── Primary metrics ───────────────────────────────────────────────────
        'energy_outage_A':    outage_A_mean,
        'energy_outage_B':    outage_B_mean,
        'comm_outage_B':      comm_out_mean,

        # ── Uncertainty estimates (for error bars or shaded bands) ────────────
        'energy_outage_A_std': outage_A_std,
        'energy_outage_B_std': outage_B_std,

        # ── Metadata for plot_results.py ─────────────────────────────────────
        'system_label':  'RF-Only, No Cooperation (Baseline)',
        'system_color':  '#e74c3c',   # red — typically used for worst-case baseline
        'system_ls':     '--',        # dashed line to visually distinguish
        'n_monte_carlo': n_monte_carlo,
        'n_slots':       n_slots,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run this file directly to check the implementation is working)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running self-test: single configuration at 8m distance...")
    result = run_single_config(distance_B_m=8.0, n_slots=5000, rng_seed=42)
    print(f"  Energy outage A : {result['energy_outage_A']:.4f}")
    print(f"  Energy outage B : {result['energy_outage_B']:.4f}")
    print(f"  Comm outage B   : {result['comm_outage_B']:.4f}")
    print("\nSanity checks:")
    print(f"  ✓ B outage > A outage: {result['energy_outage_B'] > result['energy_outage_A']}")
    print("Self-test complete. If B outage is not > A outage, check distance parameters.")
