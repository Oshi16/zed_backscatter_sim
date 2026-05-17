import numpy as np

def compute_backscatter_snr(P_carrier_W, h_forward, h_backward, N0_W_per_Hz=1e-14, 
                              bandwidth_Hz=200e3):
    """
    Compute the received SNR at the gateway for a backscatter transmission.
    
    Implements: SNR = (P_t * |h_f|^2 * |h_b|^2) / N0
    (from the research plan, consistent with bistatic backscatter literature)
    
    Parameters:
        P_carrier_W: power of the carrier signal from the access point (Watts)
        h_forward: channel gain from AP to ZED node (linear, not dB)
        h_backward: channel gain from ZED node to gateway (linear, not dB)
        N0_W_per_Hz: noise power spectral density (thermal noise: kT ≈ 4e-21 W/Hz at 290K)
        bandwidth_Hz: signal bandwidth (200 kHz is typical for LoRa-like backscatter)
    
    Returns:
        snr_linear: SNR in linear scale
        snr_db: SNR in dB
    """
    noise_power = N0_W_per_Hz * bandwidth_Hz
    snr_linear = (P_carrier_W * h_forward * h_backward) / noise_power
    snr_db = 10 * np.log10(snr_linear + 1e-30)  # avoid log(0)
    
    return snr_linear, snr_db


def compute_comm_outage_prob(P_carrier_W, n_samples, h_fwd_gains, h_bwd_gains, 
                              snr_threshold_db=-15.0):
    """
    Compute empirical communication outage probability over n_samples realizations.
    
    snr_threshold_db: minimum SNR for successful decoding (-15 dB is achievable 
                      with BPSK and modern backscatter decoders)
    """
    snr_threshold_linear = 10 ** (snr_threshold_db / 10)
    
    snr_vals, _ = compute_backscatter_snr(P_carrier_W, h_fwd_gains, h_bwd_gains)
    
    outage_events = (snr_vals < snr_threshold_linear)
    outage_prob = np.mean(outage_events)
    
    return outage_prob, snr_vals
