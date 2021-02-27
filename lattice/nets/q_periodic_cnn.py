import torch
import torch.nn as nn    
    

class QPeriodicCNN(torch.nn.Module):
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
      
        super(QPeriodicCNN, self).__init__()
        hidden_channels = hparams.hidden_channels
        out_channels = 1 if hparams.potential.startswith('ising') else 2
        lattice_size = hparams.lattice_size
        kernel_size = hparams.kernel_size
        
        # Test if lattice_size and kernel_size are appropriate
        num_inner_layers = getattr(hparams, 'num_inner_layers', None)
        padding = int((kernel_size - 1) / 2)  # half-padding
        if num_inner_layers is None:
            if (lattice_size - 1) % (kernel_size - 1) == 0 and kernel_size % 2 != 0:
                num_inner_layers = (lattice_size - 1) // (kernel_size - 1)
            else:
                num_inner_layers = 1 + (lattice_size - 1) // (kernel_size - 1)
        
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
            hidden_channels, out_channels, kernel_size,
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
        
        return out
