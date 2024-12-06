import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        reconstruction_weight: float = 0.1
    ):
        """
        Initialize the trainer
        
        Args:
            model: The predictor model
            learning_rate: Learning rate for optimization
            reconstruction_weight: Weight for reconstruction loss term
        """
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.reconstruction_weight = reconstruction_weight
        
    def gaussian_nll_loss(
        self,
        pred_mean: torch.Tensor,
        pred_sigma: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log likelihood loss for Gaussian distribution
        
        Args:
            pred_mean: Predicted mean values
            pred_sigma: Predicted standard deviation values
            target: True target values
            
        Returns:
            Negative log likelihood loss
        """
        # Ensure sigma is positive
        pred_sigma = torch.exp(pred_sigma)
        
        # Compute gaussian NLL: -log(P(x|μ,σ))
        nll = 0.5 * torch.log(2 * np.pi * pred_sigma**2) + \
              0.5 * ((target - pred_mean)**2) / (pred_sigma**2)
        
        return nll.mean()
    
    def reconstruction_loss(
        self,
        pred_reconstruction: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE reconstruction loss
        
        Args:
            pred_reconstruction: Predicted reconstruction values
            inputs: Original input values
            
        Returns:
            Reconstruction loss
        """
        return nn.MSELoss()(pred_reconstruction, inputs)
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one training step
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            Tuple of (total loss, nll loss, reconstruction loss)
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Split outputs into gaussian parameters and reconstruction
        pred_mean = outputs[:, 0]
        pred_sigma = outputs[:, 1]
        pred_reconstruction = outputs[:, 2:]
        
        # Compute losses
        nll_loss = self.gaussian_nll_loss(pred_mean, pred_sigma, targets)
        recon_loss = self.reconstruction_loss(pred_reconstruction, inputs)
        
        # Combine losses
        total_loss = nll_loss + self.reconstruction_weight * recon_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss, nll_loss, recon_loss
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of average (total loss, nll loss, reconstruction loss)
        """
        self.model.train()
        total_losses = []
        nll_losses = []
        recon_losses = []
        
        for inputs, targets in train_loader:
            total_loss, nll_loss, recon_loss = self.train_step(inputs, targets)
            total_losses.append(total_loss.item())
            nll_losses.append(nll_loss.item())
            recon_losses.append(recon_loss.item())
            
        return (
            np.mean(total_losses),
            np.mean(nll_losses),
            np.mean(recon_losses)
        )
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of average (total loss, nll loss, reconstruction loss)
        """
        self.model.eval()
        total_losses = []
        nll_losses = []
        recon_losses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Forward pass
                outputs = self.model(inputs)
                
                # Split outputs
                pred_mean = outputs[:, 0]
                pred_sigma = outputs[:, 1]
                pred_reconstruction = outputs[:, 2:]
                
                # Compute losses
                nll_loss = self.gaussian_nll_loss(pred_mean, pred_sigma, targets)
                recon_loss = self.reconstruction_loss(pred_reconstruction, inputs)
                total_loss = nll_loss + self.reconstruction_weight * recon_loss
                
                total_losses.append(total_loss.item())
                nll_losses.append(nll_loss.item())
                recon_losses.append(recon_loss.item())
        
        return (
            np.mean(total_losses),
            np.mean(nll_losses),
            np.mean(recon_losses)
        )
