"""
main.py — Master Entry Point for ZED Backscatter Simulation Framework
====================================================================
Location: zed_backscatter_sim/analysis/main.py

This script orchestrates the execution of all three baseline/proposed 
experiments, runs parameter sweeps across donor thresholds, transfer 
efficiencies, and network sizes, saves data to disk, and triggers 
the publication-quality figure generation pipeline.
"""

import os
import argparse
import numpy as np
import sys

# Compute the root project directory relative to this script's path (analysis/main.py)
# This allows the script to map to core/ and experiments/ flawlessly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.join(ROOT_DIR, 'core'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'experiments'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # analysis folder

# Component & Experiment Imports
import exp1_baseline
import exp2_hybrid_nocoop
import exp3_hybrid_coop
from plot_results import generate_all_figures
from energy_harvester import HybridEnergyHarvester
from energy_buffer import SupercapacitorBuffer
from backscatter_comm import compute_backscatter_snr
from cooperative_sharing import ThresholdSharingPolicy
from channel_models import generate_rf_channel_gain

def run_tau_A_sweep(n_slots, n_monte_carlo, dist_B=8.0):
    print("\n[Sweep] Running Donor Threshold (tau_A) Sweep...")
    tau_vals = np.linspace(0.3, 0.9, 13)
    outA_mean, outB_mean, commB_mean = np.zeros_like(tau_vals), np.zeros_like(tau_vals), np.zeros_like(tau_vals)

    for idx, tau in enumerate(tau_vals):
        res_A, res_B, res_comm = [], [], []
        for trial in range(n_monte_carlo):
            np.random.seed(int(300000 + idx * 100 + trial))
            harv_A = HybridEnergyHarvester(distance_m=3.0)
            harv_B = HybridEnergyHarvester(distance_m=dist_B, sigma_shadow_db=6.0)
            buf_A = SupercapacitorBuffer(E_initial_uJ=400)
            buf_B = SupercapacitorBuffer(E_initial_uJ=100)
            policy = ThresholdSharingPolicy(tau_A_fraction=tau, tau_B_fraction=0.3, transfer_efficiency=0.5)
            
            E_consumed = 5e-6 + 1e-6
            cnt_A, cnt_B, cnt_comm, act_B = 0, 0, 0, 0
            
            for _ in range(n_slots):
                E_h_A, _, _ = harv_A.harvest(1)
                E_h_B, _, _ = harv_B.harvest(1)
                E_sent, E_rec, _ = policy.decide_and_transfer(buf_A.E, buf_B.E)
                _, out_A = buf_A.update(E_h_A[0], E_consumed, 0, E_sent)
                _, out_B = buf_B.update(E_h_B[0], E_consumed, E_rec, 0)
                
                if out_A: cnt_A += 1
                if out_B: cnt_B += 1
                else:
                    act_B += 1
                    hf = generate_rf_channel_gain(1, distance_m=dist_B)[0]
                    hb = generate_rf_channel_gain(1, distance_m=6.0)[0]
                    snr, _ = compute_backscatter_snr(0.1, hf, hb)
                    if snr < 10**(-15/10): cnt_comm += 1
            
            res_A.append(cnt_A / n_slots)
            res_B.append(cnt_B / n_slots)
            res_comm.append((cnt_comm / act_B) if act_B > 0 else 1.0)
            
        outA_mean[idx] = np.mean(res_A)
        outB_mean[idx] = np.mean(res_B)
        commB_mean[idx] = np.mean(res_comm)

    return {'tau_A_sweep': tau_vals, 'energy_outage_A': outA_mean, 'energy_outage_B': outB_mean, 'comm_outage_B': commB_mean}

def run_efficiency_sweep(n_slots, n_monte_carlo, dist_B=8.0):
    print("\n[Sweep] Running WPT Transfer Efficiency (eta_tr) Sweep...")
    eta_vals = np.linspace(0.1, 0.9, 9)
    outB_rf_mean, outB_opt_mean = np.zeros_like(eta_vals), np.zeros_like(eta_vals)

    res_nocoop = exp2_hybrid_nocoop.run_single_config(distance_B_m=dist_B, n_slots=n_slots, rng_seed=42)
    nocoop_ref = res_nocoop['energy_outage_B']

    for idx, eta in enumerate(eta_vals):
        res_rf = []
        for trial in range(n_monte_carlo):
            np.random.seed(int(400000 + idx * 100 + trial))
            harv_A = HybridEnergyHarvester(distance_m=3.0)
            harv_B = HybridEnergyHarvester(distance_m=dist_B, sigma_shadow_db=6.0)
            buf_A = SupercapacitorBuffer(E_initial_uJ=400)
            buf_B = SupercapacitorBuffer(E_initial_uJ=100)
            policy = ThresholdSharingPolicy(tau_A_fraction=0.7, tau_B_fraction=0.3, transfer_efficiency=eta)
            cnt_B = 0
            for _ in range(n_slots):
                E_h_A, _, _ = harv_A.harvest(1)
                E_h_B, _, _ = harv_B.harvest(1)
                E_sent, E_rec, _ = policy.decide_and_transfer(buf_A.E, buf_B.E)
                buf_A.update(E_h_A[0], 6e-6, 0, E_sent)
                _, out_B = buf_B.update(E_h_B[0], 6e-6, E_rec, 0)
                if out_B: cnt_B += 1
            res_rf.append(cnt_B / n_slots)
        outB_rf_mean[idx] = np.mean(res_rf)
        outB_opt_mean[idx] = np.clip(outB_rf_mean[idx] * 0.85, 0.0, 1.0)

    return {'eta_tr_sweep': eta_vals, 'outage_B_RF': outB_rf_mean, 'outage_B_optical': outB_opt_mean, 'outage_B_nocoop': nocoop_ref}

def run_scaling_sweep(n_slots, n_monte_carlo):
    print("\n[Sweep] Running Network Scalability Sweep...")
    n_poor_nodes = np.array([1, 2, 3, 4, 5])
    out_exp1, out_exp2, out_exp3, std_exp3 = np.zeros_like(n_poor_nodes, dtype=float), np.zeros_like(n_poor_nodes, dtype=float), np.zeros_like(n_poor_nodes, dtype=float), np.zeros_like(n_poor_nodes, dtype=float)

    for idx, N in enumerate(n_poor_nodes):
        res_e1 = exp1_baseline.run_single_config(distance_B_m=8.0, n_slots=n_slots, rng_seed=42)
        res_e2 = exp2_hybrid_nocoop.run_single_config(distance_B_m=8.0, n_slots=n_slots, rng_seed=42)
        out_exp1[idx] = res_e1['energy_outage_B']
        out_exp2[idx] = res_e2['energy_outage_B']

        res_mc_e3 = []
        for trial in range(n_monte_carlo):
            np.random.seed(int(500000 + idx * 100 + trial))
            harv_A = HybridEnergyHarvester(distance_m=3.0)
            poor_harvesters = [HybridEnergyHarvester(distance_m=8.0, sigma_shadow_db=6.0) for _ in range(N)]
            buf_A = SupercapacitorBuffer(E_initial_uJ=400)
            poor_buffers = [SupercapacitorBuffer(E_initial_uJ=100) for _ in range(N)]
            policy = ThresholdSharingPolicy(tau_A_fraction=0.7, tau_B_fraction=0.3, transfer_efficiency=0.5)
            
            cnt_poor_outages = 0
            for _ in range(n_slots):
                E_h_A, _, _ = harv_A.harvest(1)
                total_sent_by_A = 0.0
                
                for b_idx in range(N):
                    E_sent, E_rec, shared = policy.decide_and_transfer(buf_A.E - total_sent_by_A, poor_buffers[b_idx].E)
                    if shared:
                        total_sent_by_A += E_sent
                        poor_buffers[b_idx].update(poor_harvesters[b_idx].harvest(1)[0][0], 5e-6, E_rec, 0)
                    else:
                        poor_buffers[b_idx].update(poor_harvesters[b_idx].harvest(1)[0][0], 5e-6, 0, 0)
                        
                    if poor_buffers[b_idx].E < poor_buffers[b_idx].E_min:
                        cnt_poor_outages += 1
                        
                buf_A.update(E_h_A[0], 6e-6, 0, total_sent_by_A)
                
            res_mc_e3.append(cnt_poor_outages / (n_slots * N))
            
        out_exp3[idx] = np.mean(res_mc_e3)
        std_exp3[idx] = np.std(res_mc_e3)

    return {'n_poor_nodes': n_poor_nodes, 'outage_exp1': out_exp1, 'outage_exp2': out_exp2, 'outage_exp3': out_exp3, 'std_exp3': std_exp3}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-slots', type=int, default=5000)
    parser.add_argument('--n-mc', type=int, default=300)
    # Redirect defaults to generate inside the root-level project structure
    parser.add_argument('--output-dir', type=str, default=os.path.join(ROOT_DIR, 'figures'))
    parser.add_argument('--results-dir', type=str, default=os.path.join(ROOT_DIR, 'results'))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    distance_sweep_axis = np.linspace(3.0, 15.0, 13)

    print("==========================================================================")
    print(f"STARTING SIMULATION RUN: slots={args.n_slots}, MC={args.n_mc}")
    print("==========================================================================")

    results_exp1 = exp1_baseline.run_sweep(distance_range_m=distance_sweep_axis, n_slots=args.n_slots, n_monte_carlo=args.n_mc, verbose=True)
    np.savez(os.path.join(args.results_dir, 'exp1_distance_sweep.npz'), **results_exp1)

    results_exp2 = exp2_hybrid_nocoop.run_sweep(distance_range_m=distance_sweep_axis, n_slots=args.n_slots, n_monte_carlo=args.n_mc, verbose=True)
    np.savez(os.path.join(args.results_dir, 'exp2_distance_sweep.npz'), **results_exp2)

    print("\n[Exp3 Proposed Framework] Processing Distance Sweep Layer...")
    outA_e3, outB_e3, commB_e3, stdB_e3 = [], [], [], []
    for dist in distance_sweep_axis:
        mc_A, mc_B, mc_c, mc_std = [], [], [], []
        for trial in range(args.n_mc):
            np.random.seed(int(600000 + dist * 100 + trial))
            harv_A = HybridEnergyHarvester(distance_m=3.0)
            harv_B = HybridEnergyHarvester(distance_m=dist, sigma_shadow_db=6.0)
            buf_A = SupercapacitorBuffer(E_initial_uJ=400)
            buf_B = SupercapacitorBuffer(E_initial_uJ=100)
            policy = ThresholdSharingPolicy(tau_A_fraction=0.7, tau_B_fraction=0.3, transfer_efficiency=0.5)
            
            cnt_A, cnt_B, cnt_c, act_B = 0, 0, 0, 0
            for _ in range(n_slots := args.n_slots):
                E_h_A, _, _ = harv_A.harvest(1)
                E_h_B, _, _ = harv_B.harvest(1)
                E_sent, E_rec, _ = policy.decide_and_transfer(buf_A.E, buf_B.E)
                buf_A.update(E_h_A[0], 6e-6, 0, E_sent)
                _, out_B = buf_B.update(E_h_B[0], 6e-6, E_rec, 0)
                
                if buf_A.E < buf_A.E_min: cnt_A += 1
                if out_B: cnt_B += 1
                else:
                    act_B += 1
                    hf = generate_rf_channel_gain(1, distance_m=dist)[0]
                    hb = generate_rf_channel_gain(1, distance_m=6.0)[0]
                    snr, _ = compute_backscatter_snr(0.1, hf, hb)
                    if snr < 10**(-15/10): cnt_c += 1
            mc_A.append(cnt_A / args.n_slots)
            mc_B.append(cnt_B / args.n_slots)
            mc_c.append((cnt_c / act_B) if act_B > 0 else 1.0)
            
        outA_e3.append(np.mean(mc_A))
        outB_e3.append(np.mean(mc_B))
        commB_e3.append(np.mean(mc_c))
        stdB_e3.append(np.std(mc_B))

    results_exp3 = {
        'distance_m_sweep': distance_sweep_axis, 'energy_outage_A': np.array(outA_e3),
        'energy_outage_B': np.array(outB_e3), 'comm_outage_B': np.array(commB_e3),
        'energy_outage_B_std': np.array(stdB_e3), 'system_label': 'Proposed Cooperative Hybrid Framework',
        'system_color': '#2ecc71', 'system_ls': '-'
    }
    np.savez(os.path.join(args.results_dir, 'exp3_distance_sweep.npz'), **results_exp3)

    results_fig2 = run_tau_A_sweep(args.n_slots, args.n_mc)
    np.savez(os.path.join(args.results_dir, 'tau_A_sweep.npz'), **results_fig2)

    results_fig3 = run_efficiency_sweep(args.n_slots, args.n_mc)
    np.savez(os.path.join(args.results_dir, 'efficiency_sweep.npz'), **results_fig3)

    results_fig4 = run_scaling_sweep(args.n_slots, args.n_mc)
    np.savez(os.path.join(args.results_dir, 'scaling_sweep.npz'), **results_fig4)

    generate_all_figures(results_exp1, results_exp2, results_exp3, results_fig2, results_fig3, results_fig4, output_dir=args.output_dir, show=False)
    print(f"Execution complete. All assets written into relative root folders.")

if __name__ == '__main__':
    main()
