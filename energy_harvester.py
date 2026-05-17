import numpy as np
from channel_models import generate_rf_channel_gain, generate_optical_channel_gain

class HybridEnergyHarvester:
    """
    Models hybrid RF + optical energy harvesting at a ZED node.
    
    This implements:
        E_RF(t) = η_rf * P_rf(t) * h_i  (from research plan eq. 1)
        E_opt(t) = η_opt * P_opt(t) * g_i (from research plan eq. 2)
        E_total(t) = E_RF(t) + E_opt(t)  (from research plan eq. 3)
    """
    
    def __init__(self, eta_rf=0.5, eta_opt=0.6, P_rf_mean_mW=0.1, P_opt_mean_mW=1.0,
                 distance_m=5.0, path_loss_exp=2.5, sigma_shadow_db=3.0):
        """
        Parameters:
            eta_rf: RF energy conversion efficiency (0.3–0.6 is realistic for rectennas)
            eta_opt: optical energy conversion efficiency (0.5–0.7 for silicon PV cells)
            P_rf_mean_mW: mean transmitted RF power available for harvesting (in mW)
            P_opt_mean_mW: mean optical illumination power at the node (in mW)
            distance_m: node distance from the hybrid access point
            path_loss_exp: RF path loss exponent
            sigma_shadow_db: optical shadowing standard deviation
        """
        self.eta_rf = eta_rf
        self.eta_opt = eta_opt
        self.P_rf = P_rf_mean_mW * 1e-3   # convert to Watts
        self.P_opt = P_opt_mean_mW * 1e-3
        self.distance_m = distance_m
        self.path_loss_exp = path_loss_exp
        self.sigma_shadow_db = sigma_shadow_db
    
    def harvest(self, n_samples):
        """
        Returns n_samples realizations of total harvested energy per time slot (in Joules).
        
        Assumes each time slot is T=1ms (adjust T_slot_s for different durations).
        """
        T_slot_s = 1e-3  # 1 millisecond time slot (typical for IoT duty cycling)
        
        h = generate_rf_channel_gain(n_samples, self.path_loss_exp, self.distance_m)
        g = generate_optical_channel_gain(n_samples, self.sigma_shadow_db)
        
        E_rf = self.eta_rf * self.P_rf * h * T_slot_s   # Joules
        E_opt = self.eta_opt * self.P_opt * g * T_slot_s  # Joules
        
        E_total = E_rf + E_opt
        
        return E_total, E_rf, E_opt  # return components separately for analysis
