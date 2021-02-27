import torch
import torch.nn as nn    
    

class PassiveRates(torch.nn.Module):
    """ Scalar multiple of the passive rates with learnable proportionality factor. """
    
    def __init__(self, hparams, passive_rates_fn):
        super(PassiveRates, self).__init__()
        self.passive_rates_fn = passive_rates_fn
        Z = getattr(hparams, 'Z', 1.0)
        self.Z = nn.Parameter(torch.tensor(Z))
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch size, lattice_size, lattice_size]
    
        Returns:
            out: Tensor of shape [batch size, out_channels, lattice_size, lattice_size]
                 proportional to the passive rates
        """
        
        return self.Z * self.passive_rates_fn(x)
