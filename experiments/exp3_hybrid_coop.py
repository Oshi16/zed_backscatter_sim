# experiments/exp3_hybrid_coop.py  — The Proposed System
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../core')

from energy_harvester import HybridEnergyHarvester
from energy_buffer import SupercapacitorBuffer
from backscatter_comm import compute_backscatter_snr
from cooperative_sharing import ThresholdSharingPolicy
from channel_models import generate_rf_channel_gain

def run_simulation(n_slots=10000, n_monte_carlo=1000, 
                   tau_A_frac=0.7, tau_B_frac=0.3,
                   transfer_efficiency=0.5):
    """
    Run one Monte Carlo trial of the proposed cooperative hybrid harvesting system.
    
    Returns: (energy_outage_A, energy_outage_B, comm_outage_B, sharing_fraction)
    """
    outages_A = []
    outages_B = []
    comm_outages = []
    sharing_events = []
    
    for trial in tqdm(range(n_monte_carlo), desc="Monte Carlo trials"):
        # Initialize components for this trial
        harvester_A = HybridEnergyHarvester(distance_m=3.0)   # Node A: closer to AP
        harvester_B = HybridEnergyHarvester(distance_m=8.0,   # Node B: farther away
                                             sigma_shadow_db=6.0)  # more shadowing
        buffer_A = SupercapacitorBuffer(E_initial_uJ=400)
        buffer_B = SupercapacitorBuffer(E_initial_uJ=100)  # B starts energy-poor
        policy = ThresholdSharingPolicy(tau_A_fraction=tau_A_frac, 
                                        tau_B_fraction=tau_B_frac,
                                        transfer_efficiency=transfer_efficiency)
        
        # Energy consumption per slot (sensing + control logic)
        E_sense_J = 5e-6     # 5 µJ per sensing event (ultra-low-power sensor)
        E_backscatter_J = 1e-6  # 1 µJ for backscatter modulation (very low)
        
        trial_outages_A = 0
        trial_outages_B = 0
        trial_comm_outages = 0
        trial_shares = 0
        
        for t in range(n_slots):
            # Step 1: Harvest energy this slot
            E_harv_A, _, _ = harvester_A.harvest(n_samples=1)
            E_harv_B, _, _ = harvester_B.harvest(n_samples=1)
            
            # Step 2: Make sharing decision
            E_sent, E_received, shared = policy.decide_and_transfer(buffer_A.E, buffer_B.E)
            if shared:
                trial_shares += 1
            
            # Step 3: Update buffers (harvest + share + consume)
            _, outage_A = buffer_A.update(E_harv_A[0], E_sense_J + E_backscatter_J, 
                                           E_shared_received=0, E_shared_sent=E_sent)
            _, outage_B = buffer_B.update(E_harv_B[0], E_sense_J, 
                                           E_shared_received=E_received, E_shared_sent=0)
            
            if outage_A:
                trial_outages_A += 1
            if outage_B:
                trial_outages_B += 1
            
            # Step 4: Node B attempts backscatter if it has energy
            if not outage_B:
                h_fwd = generate_rf_channel_gain(1, distance_m=8.0)[0]
                h_bwd = generate_rf_channel_gain(1, distance_m=6.0)[0]  # B-to-gateway distance
                snr, _ = compute_backscatter_snr(0.1, h_fwd, h_bwd)   # 100 mW carrier
                snr_threshold = 10 ** (-15 / 10)  # -15 dB threshold
                if snr < snr_threshold:
                    trial_comm_outages += 1
        
        outages_A.append(trial_outages_A / n_slots)
        outages_B.append(trial_outages_B / n_slots)
        comm_outages.append(trial_comm_outages / n_slots)
        sharing_events.append(trial_shares / n_slots)
    
    return (np.mean(outages_A), np.mean(outages_B), 
            np.mean(comm_outages), np.mean(sharing_events))
