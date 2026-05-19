import numpy as np

def generate_rf_channel_gain(n_samples, path_loss_exponent=2.5, distance_m=5.0):
    """
    Generate n_samples of RF channel power gains.
    The gain combines distance-dependent path loss (deterministic) and Rayleigh small-scale fading (stochastic). 
    In Rayleigh fading, the power gain h^2 is exponentially distributed with mean 1.
    Parameters:
        n_samples: number of Monte Carlo realizations
        path_loss_exponent: α in the path loss formula (2.5 typical indoors)
        distance_m: distance from AP to node in meters
    """
    
    # Distance-dependent mean path loss (deterministic component)
    # Using the simplified path loss model: PL = (d/d0)^α, with d0 = 1m reference
    mean_path_loss = distance_m ** (-path_loss_exponent)
    
    # Rayleigh fading: power gain is exponentially distributed
    # np.random.exponential(scale=mean) gives Exp(mean) samples
    h_squared = np.random.exponential(scale=mean_path_loss, size=n_samples)
    
    return h_squared  # shape: (n_samples,)

def generate_optical_channel_gain(n_samples, sigma_shadow_db=3.0, mean_gain_linear=0.8):
    """
    Generate n_samples of optical channel power gains using log-normal shadowing.
    sigma_shadow_db: standard deviation of shadowing in dB (3 dB is mild shadowing)
    mean_gain_linear: mean optical efficiency (accounts for photodiode efficiency, incidence angle, and nominal optical path)
    """
    
    # Log-normal: if X ~ Normal(mu, sigma^2), then 10^(X/10) is log-normal
    sigma_linear = sigma_shadow_db / (10 * np.log10(np.e))  # convert dB std to natural log std
    
    # Draw log-normal samples with the given mean and variance
    log_mean = np.log(mean_gain_linear) - 0.5 * sigma_linear**2
    g = np.random.lognormal(mean=log_mean, sigma=sigma_linear, size=n_samples)
    
    return g  # shape: (n_samples,)
