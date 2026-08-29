import yaml
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

class ExperienceBuilder:
    def __init__(self, config_path: str = "configs/continual.yaml"):
        """
        Initializes the ExperienceBuilder which parses the data split configuration.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.experiences = self.config.get("experiences", [])

    def get_num_experiences(self) -> int:
        return len(self.experiences)

    def get_classes_for_experience(self, exp_id: int) -> List[str]:
        for exp in self.experiences:
            if exp["id"] == exp_id:
                return exp["classes"]
        return []

    def get_cumulative_classes(self, exp_id: int) -> List[str]:
        """
        Returns all classes introduced up to and including the given experience ID.
        """
        classes = []
        for exp in self.experiences:
            classes.extend(exp["classes"])
            if exp["id"] == exp_id:
                break
        return classes

    def build_experience(self, X: np.ndarray, y_cat: np.ndarray, exp_id: int, cumulative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters the dataset to only include classes present in the requested experience.
        
        Args:
            X: Feature matrix
            y_cat: Label array (strings)
            exp_id: The experience ID (1-indexed)
            cumulative: If True, returns all classes seen up to exp_id. If False, only returns classes new to exp_id.
            
        Returns:
            X_exp, y_cat_exp
        """
        if cumulative:
            target_classes = self.get_cumulative_classes(exp_id)
        else:
            target_classes = self.get_classes_for_experience(exp_id)
            
        mask = np.isin(y_cat, target_classes)
        return X[mask], y_cat[mask]

    def split_dataset(self, X: np.ndarray, y_cat: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Splits the entire dataset into a dictionary of isolated experiences.
        Returns: {1: (X_exp1, y_exp1), 2: (X_exp2, y_exp2), ...}
        """
        splits = {}
        for exp in self.experiences:
            eid = exp["id"]
            splits[eid] = self.build_experience(X, y_cat, eid, cumulative=False)
        return splits
