import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    """
    Drift detector based on the Kolmogorov-Smirnov (KS) test.
    
    Justification: 
    The KS-test is a non-parametric test that compares the cumulative distributions 
    of two 1D empirical samples. By calculating the KS statistic for each feature 
    independently and averaging them, we obtain a global drift score that is 
    sensitive to changes in both the mean and the shape of the feature distributions 
    (which is exactly what happens when new attack payloads introduce new structural 
    characteristics in the network flows). It is computationally lightweight compared 
    to training a discriminator model on the fly, making it suitable for real-time IDS.
    """
    
    def __init__(self, threshold: float = 0.15, max_ref_size: int = 5000):
        """
        Args:
            threshold: The average KS statistic threshold above which drift is flagged.
            max_ref_size: Maximum number of reference samples to retain for speed.
        """
        self.threshold = threshold
        self.max_ref_size = max_ref_size
        self.X_ref = None

    def update_reference(self, X: np.ndarray):
        """
        Updates the reference distribution. In a real stream, you might blend this.
        Here we just replace or append up to max_ref_size.
        """
        if len(X) > self.max_ref_size:
            idx = np.random.choice(len(X), self.max_ref_size, replace=False)
            self.X_ref = X[idx].copy()
        else:
            self.X_ref = X.copy()

    def detect(self, X_new: np.ndarray) -> dict:
        """
        Compares X_new against X_ref.
        Returns:
            dict containing:
            - drift_score (float): Average KS statistic across all features.
            - drift_detected (bool): True if drift_score > threshold.
            - p_value_avg (float): Average p-value.
        """
        if self.X_ref is None:
            raise ValueError("Reference distribution not set. Call update_reference() first.")
            
        num_features = self.X_ref.shape[1]
        
        # Subsample X_new if too large to keep KS fast
        if len(X_new) > self.max_ref_size:
            idx = np.random.choice(len(X_new), self.max_ref_size, replace=False)
            X_new = X_new[idx]
            
        ks_stats = []
        p_values = []
        
        for i in range(num_features):
            ref_feat = self.X_ref[:, i]
            new_feat = X_new[:, i]
            
            # Avoid calculating if both are completely constant and identical (can happen with categorical mappings)
            if np.all(ref_feat == ref_feat[0]) and np.all(new_feat == new_feat[0]) and ref_feat[0] == new_feat[0]:
                ks_stats.append(0.0)
                p_values.append(1.0)
            else:
                stat, p_val = ks_2samp(ref_feat, new_feat)
                ks_stats.append(stat)
                p_values.append(p_val)
                
        avg_ks = float(np.mean(ks_stats))
        avg_p = float(np.mean(p_values))
        
        return {
            "drift_score": avg_ks,
            "drift_detected": avg_ks > self.threshold,
            "p_value_avg": avg_p
        }
