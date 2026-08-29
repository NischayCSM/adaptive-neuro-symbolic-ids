import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size: int = 2000, fraction_hard: float = 0.3):
        """
        Initializes the ReplayBuffer.
        
        Args:
            max_size: Maximum number of samples to store in the buffer.
            fraction_hard: Fraction of the per-class allocation dedicated to hard examples
                           (lowest-confidence correct predictions).
        """
        self.max_size = max_size
        self.fraction_hard = fraction_hard
        self.buffer_X = []
        self.buffer_y = []

    def update(self, model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device):
        """
        Selects samples from the current experience (X, y) and adds them to the buffer.
        Evaluates the model on X to find hard examples (lowest confidence correct predictions).
        """
        model.eval()
        with torch.no_grad():
            # Batch inference to get confidences
            # If X is very large, this should be batched, but for this milestone X is manageable.
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(X_tensor)
            
            # Multi-class softmax
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            confidences = confidences.cpu().numpy()
            preds = preds.cpu().numpy()
            
        correct_mask = (preds == y)
        
        unique_classes = np.unique(y)
        # Allocate capacity roughly evenly among all classes seen so far.
        # Since we don't know total classes yet, we just allocate based on current classes
        # and trim later.
        samples_per_class = max(1, (self.max_size // 2) // len(unique_classes))
        
        selected_indices = []
        
        for cls in unique_classes:
            cls_mask = (y == cls)
            cls_correct_mask = cls_mask & correct_mask
            
            # 1. Hard Examples (Lowest confidence correct predictions)
            hard_indices = np.where(cls_correct_mask)[0]
            num_hard = int(samples_per_class * self.fraction_hard)
            
            if len(hard_indices) > 0:
                hard_confs = confidences[hard_indices]
                sorted_idx = np.argsort(hard_confs) # Ascending order (lowest confidence first)
                hard_indices = hard_indices[sorted_idx]
                selected_indices.extend(hard_indices[:num_hard].tolist())
                
            # 2. Random Examples (Standard representative sampling)
            all_cls_indices = np.where(cls_mask)[0]
            num_random = samples_per_class - num_hard
            
            if len(all_cls_indices) > 0:
                rand_indices = np.random.choice(all_cls_indices, min(num_random, len(all_cls_indices)), replace=False)
                selected_indices.extend(rand_indices.tolist())
                
        # Append selected samples to buffer
        for idx in selected_indices:
            self.buffer_X.append(X[idx])
            self.buffer_y.append(y[idx])
            
        # Trim buffer if it exceeds max_size (Random removal to maintain distribution)
        if len(self.buffer_X) > self.max_size:
            keep_indices = np.random.choice(len(self.buffer_X), self.max_size, replace=False)
            self.buffer_X = [self.buffer_X[i] for i in keep_indices]
            self.buffer_y = [self.buffer_y[i] for i in keep_indices]

    def get_data(self) -> tuple:
        """
        Returns the entire buffer as a tuple (X_replay, y_replay).
        """
        if len(self.buffer_X) == 0:
            return np.array([]), np.array([])
        return np.array(self.buffer_X), np.array(self.buffer_y)
