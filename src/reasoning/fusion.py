import numpy as np
from typing import Dict, Any

class NeuroSymbolicFuser:
    def __init__(self, alpha: float = 0.5):
        """
        Initializes the Fuser.
        
        Args:
            alpha: Weight given to the Neural Score (0.0 to 1.0). 
                   (1 - alpha) is given to the Symbolic Score.
        """
        self.alpha = alpha

    def fuse(self, neural_probs: Dict[str, float], symbolic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuses neural probabilities with symbolic confidence.
        
        Args:
            neural_probs: Dictionary mapping class name to predicted probability.
            symbolic_result: The output dict from SymbolicReasoner.predict().
            
        Returns:
            Dict containing:
                - final_predicted_class
                - final_confidence
                - neural_score
                - symbolic_score
                - symbolic_explanation
        """
        # Determine all possible classes, in a fixed order: neural_probs' key order (every caller
        # builds it in class-index order), then classes named only by a fired rule.
        all_classes = list(neural_probs.keys())

        # Calculate symbolic scores per class
        symbolic_scores = {cls: 0.0 for cls in all_classes}

        # We can extract per-class symbolic scores from the all_matches if present
        if "all_matches" in symbolic_result and symbolic_result["all_matches"]:
            for rule in symbolic_result["all_matches"]:
                cls = rule["predicted_class"]
                prec = rule.get("precision", 0.0)
                if cls not in symbolic_scores:
                    symbolic_scores[cls] = 0.0
                    all_classes.append(cls)
                # Keep the max precision for that class
                symbolic_scores[cls] = max(symbolic_scores[cls], prec)

        final_scores = {}
        for cls in all_classes:
            n_score = neural_probs.get(cls, 0.0)
            s_score = symbolic_scores.get(cls, 0.0)
            final_scores[cls] = (self.alpha * n_score) + ((1 - self.alpha) * s_score)

        # Exact ties in the top score go to the first tied class in all_classes, i.e. the lowest
        # class index, as np.argmax does in fusion_variants.v0_linear (max() keeps the first maximum).
        # all_classes used to be a set, whose iteration order over class-name strings depends on
        # PYTHONHASHSEED, so a tied row could get a different class from one run to the next.
        final_predicted_class = max(final_scores.items(), key=lambda x: x[1])[0]
        final_confidence = final_scores[final_predicted_class]
        
        # Determine explanation: if the chosen class matches the best symbolic class, return its evidence
        explanation = []
        if final_predicted_class == symbolic_result.get("predicted_class"):
            explanation = symbolic_result.get("evidence", [])
        elif symbolic_scores.get(final_predicted_class, 0.0) > 0:
            explanation = [f"A lower-confidence symbolic rule for {final_predicted_class} fired."]
        else:
            explanation = ["No symbolic rule fired for this class. Prediction is purely neural."]
            
        return {
            "final_predicted_class": final_predicted_class,
            "final_confidence": final_confidence,
            "neural_score": neural_probs.get(final_predicted_class, 0.0),
            "symbolic_score": symbolic_scores.get(final_predicted_class, 0.0),
            "symbolic_explanation": explanation
        }
