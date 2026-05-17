import numpy as np

class SupercapacitorBuffer:
    """
    Finite-capacity energy buffer (supercapacitor) for a ZED node.
    
    Implements the energy evolution equation from the research plan:
        E_i(t+1) = min{ E_i(t) - E_use(t) + E_harv(t) + E_share(t), E_max }
    
    The 'energy outage' event occurs when E_i(t) < E_min_operate,
    meaning the node cannot perform even its most basic sensing function.
    """
    
    def __init__(self, E_max_uJ=500.0, E_min_operate_uJ=10.0, E_initial_uJ=None):
        """
        Parameters:
            E_max_uJ: maximum storage capacity in microjoules (500 µJ is typical 
                      for a 100µF supercap charged to 3.3V: E = 0.5 * C * V^2)
            E_min_operate_uJ: minimum energy needed to wake up and sense
            E_initial_uJ: starting energy (defaults to half capacity if None)
        """
        self.E_max = E_max_uJ * 1e-6   # convert to Joules
        self.E_min = E_min_operate_uJ * 1e-6
        self.E = (E_max_uJ / 2) * 1e-6 if E_initial_uJ is None else E_initial_uJ * 1e-6
    
    def update(self, E_harvested, E_consumed, E_shared_received=0.0, E_shared_sent=0.0):
        """
        Update buffer state for one time slot.
        
        Returns:
            new_energy: updated buffer level (Joules)
            outage: True if node cannot operate (energy below minimum)
        """
        # Apply energy causality: cannot use more than available
        E_available = self.E + E_harvested + E_shared_received
        E_net = E_available - E_consumed - E_shared_sent
        
        # Enforce storage bounds
        self.E = np.clip(E_net, 0.0, self.E_max)
        
        # Check for energy outage
        outage = (self.E < self.E_min)
        
        return self.E, outage
