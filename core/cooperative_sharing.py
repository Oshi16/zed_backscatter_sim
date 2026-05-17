import numpy as np

class ThresholdSharingPolicy:
    """
    Cooperative energy sharing between a donor ZED (Node A) and 
    a receiver ZED (Node B) using a simple threshold-based policy.
    
    Logic:
        - Node A shares IF: E_A > tau_A (A has surplus beyond its own safety margin)
        - Node B receives IF: E_B < tau_B (B is below its sustainability threshold)
        - Transfer amount: min(E_transfer_max, E_A - tau_A) — A never depletes below tau_A
    
    This implements a conservative energy cooperation strategy that guarantees
    donor node survival, which is the key design constraint for ZED cooperation.
    """
    
    def __init__(self, tau_A_fraction=0.7, tau_B_fraction=0.3, 
                 transfer_efficiency=0.5, E_max_J=500e-6):
        """
        Parameters:
            tau_A_fraction: fraction of E_max above which Node A will donate
            tau_B_fraction: fraction of E_max below which Node B will request
            transfer_efficiency: η_tr in the research plan (WPT path loss 
                                 reduces how much Node B actually receives)
            E_max_J: maximum buffer capacity in Joules
        """
        self.tau_A = tau_A_fraction * E_max_J
        self.tau_B = tau_B_fraction * E_max_J
        self.eta_tr = transfer_efficiency
        self.E_max = E_max_J
    
    def decide_and_transfer(self, E_A, E_B):
        """
        Make sharing decision and compute energy exchange for one time slot.
        
        Returns:
            E_sent: energy removed from Node A's buffer (Joules)
            E_received: energy added to Node B's buffer (accounting for transfer losses)
            sharing_occurred: boolean flag
        """
        sharing_occurred = False
        E_sent = 0.0
        E_received = 0.0
        
        if E_A > self.tau_A and E_B < self.tau_B:
            # Determine how much A can safely give
            E_transferable = E_A - self.tau_A
            E_sent = E_transferable  # A gives everything above its safety margin
            
            # Transfer loss: B receives eta_tr fraction due to WPT path loss
            E_received = self.eta_tr * E_sent
            
            sharing_occurred = True
        
        return E_sent, E_received, sharing_occurred
