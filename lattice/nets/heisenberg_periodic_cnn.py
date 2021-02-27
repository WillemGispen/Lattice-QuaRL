import math
import torch
import torch.nn as nn    
    

class HeisenbergPeriodicCNN(torch.nn.Module):
    """
    Fully-correlated CNN for square lattice with periodic boundary conditions.
    
    Properties:
        - Only 1 in-going channel
        - Each lattice site is correlated with every other
        - Conserves shape of input by using 'periodic half-padding'
        
    Interpretation:
        The array (b, :, h, w) of the output is to be interpreted as a set of
        transition rates pertaining to site (h, w) in sample b of the batch.
        - Ising model: set out_channels = 1, then element (b, 0, h, w) is the
            transition rate to flip the spin at site (h, w)
        - XY/BH model: out_channels = 4 (number of neighboring sites), then element
            (b, out, h, w) is the transition rate for a particle to hop from site
            (h, w) to its neighbor numbered 'out'. We could e.g. count with the
            clock where (0, 1, 2, 3) corresponds to ('up', 'right', 'down', 'left').
    """

    def __init__(self, hparams):
        """
        hparams:
            hidden_channels (int): Number of hidden channels to use in the inner
              layer of the CNN.
            out_channels (int): Number of channels going out of the CNN.
            lattice_size (int): Height and width of the square lattice.
            kernel_size (int): Height and width of the kernel of the convolution.
            
        Layers:
            A number of layers, each a 2d convolution followed by a ReLU.
            Padding:
                Each 2d convolution preserves the shape of its input by using
                half-padding. The padding values are taken periodically, e.g. the
                padding values just right of the right-most column are the values of
                the left-most column.
            Number of layers:
                num_layers is chosen such that with the chosen kernel_size,
                each lattice site is correlated with every other.
                
        Raises:
            ValueError:
                if lattice_size is not such that an appropriate num_layers can be
                chosen, or if kernel_size is not odd.
        """
      
        super(HeisenbergPeriodicCNN, self).__init__()
        hidden_channels = hparams.hidden_channels
        self.out_channels = 1 if hparams.potential.startswith('ising') else 4
        lattice_size = hparams.lattice_size
        kernel_size = hparams.kernel_size
        
        # Test if lattice_size and kernel_size are appropriate
        if (lattice_size - 1) % (kernel_size - 1) == 0 and kernel_size % 2 != 0:
            num_inner_layers = (lattice_size - 1) // (kernel_size - 1)
            padding = int((kernel_size - 1) / 2)  # half-padding
        else:
            num_inner_layers = 1 + (lattice_size - 1) // (kernel_size - 1)
            padding = int((kernel_size - 1) / 2)  # half-padding
        
        # Define initial layer
        self.initial_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_channels,
                            kernel_size, padding=padding, padding_mode='circular'),
            torch.nn.ReLU()
        )
        
        # Define inner layers
        self.inner_layers = nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                                padding=padding, padding_mode='circular'),
                torch.nn.ReLU()
            ) for i in range(num_inner_layers)
        ])
        
        # Define final layer
        self.final_layer = torch.nn.Conv2d(
            hidden_channels, 2 * self.out_channels, kernel_size,
            padding=padding, padding_mode='circular')

        self.final_layer.weight.data *= 0.0
        

    def forward(self, state):
        """
        Args:
            state: Tensor of shape [batch size, lattice_size, lattice_size]
    
        Returns:
            out: Tensor of shape [batch size, out_channels, lattice_size, lattice_size]
                filled with non-negative floats.
        """
        x = state.unsqueeze(dim=-3)
        hidden = self.initial_layer(x)
        
        for layer in self.inner_layers:
            hidden = layer(hidden)
        
        out = self.final_layer(hidden)
        re_out, im_out = out.chunk(2, dim=1)

        sublattice = state[:, 0::2, 0::2] + state[:, 1::2, 1::2] # one sublattice, summed for convenience
        marshall_phase = sublattice.sum(dim=(-1, -2)) # number of up spins on one sublattice
        marshall_phase += 1 # because we want phase of state after one spin flip, not of current state
        marshall_phase = math.pi * torch.fmod(marshall_phase, 2) # convert to radians

        im_out += marshall_phase.view(state.shape[0], 1, 1, 1)
        
        return re_out, im_out
