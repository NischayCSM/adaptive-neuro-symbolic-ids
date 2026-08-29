"""
Neural IDS detector as described in the design document.

Architecture:
Dense -> ReLU -> Dropout -> Dense -> ReLU -> Output

Includes `get_embeddings` for extracting penultimate-layer activations.
"""

import torch
import torch.nn as nn
import numpy as np

class NeuralIDS(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout_rate: float = 0.3, output_dim: int = 1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        
        # Architecture: Dense -> ReLU -> Dropout -> Dense -> ReLU -> Output
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        # Dense -> ReLU -> Dropout
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        # Dense -> ReLU (Penultimate)
        x = self.layer2(x)
        x = self.relu2(x)
        
        # Output
        x = self.output_layer(x)
        return x
        
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the penultimate-layer activations (after second ReLU) 
        for use in rule induction or other tasks.
        """
        with torch.no_grad():
            x = self.layer1(x)
            x = self.relu1(x)
            x = self.dropout(x)
            x = self.layer2(x)
            x = self.relu2(x)
            return x

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Returns predicted probabilities as numpy array."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_dim == 1:
                logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                return np.column_stack([1 - probs, probs])
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return probs
