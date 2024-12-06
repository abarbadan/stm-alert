import torch
import torch.nn as nn

NUM_INPUT_FEATURES = 4
BASE_HIDDEN_NODES = 4
ENCODING_HIDDEN_NODES = 3

class Predictor(nn.Module):
    def __init__(self, width: int):
        """
        Initialize feedforward network with configurable hidden layer sizes
        
        Args:
            hidden_sizes (tuple): Tuple of two integers specifying the size of hidden layers
                                in encoder and decoder paths. Default (32, 16)
        """
        super().__init__()
        
        h1 = 2 * BASE_HIDDEN_NODES * width
        h2 = BASE_HIDDEN_NODES * width
        
        # Encoder path: 4 inputs -> h1 -> h2 -> 3 latent
        self.encoder = nn.Sequential(
            nn.Linear(NUM_INPUT_FEATURES, h1),
            nn.ReLU(),
            nn.Linear(h1, h2), 
            nn.ReLU(),
            nn.Linear(h2, ENCODING_HIDDEN_NODES)
        )
        
        # Decoder path: 3 latent -> h2 -> h1 -> 5 outputs
        self.decoder = nn.Sequential(
            nn.Linear(ENCODING_HIDDEN_NODES, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, 6)
        )

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 5)
        """
        latent = self.encoder(x)
        output = self.decoder(latent)
        
        return output
