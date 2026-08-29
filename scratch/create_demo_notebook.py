import json
from pathlib import Path
import os

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['# Neuro-Symbolic Fusion Demo\n', 'This notebook demonstrates the end-to-end pipeline on 5 example network flows.']
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import numpy as np\n',
                'import torch\n',
                'from pathlib import Path\n',
                'import os\n\n',
                'from src.models.neural_ids import NeuralIDS\n',
                'from src.rules.knowledge_base import KnowledgeBase\n',
                'from src.reasoning.reasoner import SymbolicReasoner\n',
                'from src.reasoning.fusion import NeuroSymbolicFuser\n\n',
                'PROJECT_ROOT = Path(os.getcwd()).parent\n',
                'PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"\n',
                'RESULTS_DIR = PROJECT_ROOT / "results"\n',
                'NEURAL_DIR = RESULTS_DIR / "neural_model"\n',
                'DB_PATH = PROCESSED_DIR / "knowledge_base.db"\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Load data\n',
                'try:\n',
                '    X_test = np.load(PROCESSED_DIR / "X_test.npy")\n',
                '    y_cat_test = np.load(PROCESSED_DIR / "y_cat_test.npy")\n',
                '    \n',
                '    # Pick 5 examples\n',
                '    sample_indices = np.random.choice(len(X_test), 5, replace=False)\n',
                '    X_demo = X_test[sample_indices]\n',
                '    y_demo = y_cat_test[sample_indices]\n',
                '    \n',
                '    # Mock feature names if pipeline not available\n',
                '    try:\n',
                '        from src.preprocessing.pipeline import UNSW_NB15_Pipeline\n',
                '        pipe = UNSW_NB15_Pipeline.load(PROCESSED_DIR / "pipeline.joblib")\n',
                '        feature_names = pipe.feature_names\n',
                '    except:\n',
                '        feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]\n',
                '        \n',
                '    print("Data loaded successfully.")\n',
                'except FileNotFoundError:\n',
                '    print("Error: Data not found.")\n',
                '    X_demo = None\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'if X_demo is not None:\n',
                '    # Initialize Neural Model\n',
                '    device = torch.device("cpu")\n',
                '    neural_model = NeuralIDS(input_dim=X_demo.shape[1]).to(device)\n',
                '    try:\n',
                '        neural_model.load_state_dict(torch.load(NEURAL_DIR / "neural_ids_weights.pth", map_location=device))\n',
                '    except:\n',
                '        print("Warning: Neural weights not found, using untrained model for demo.")\n',
                '    neural_model.eval()\n',
                '    \n',
                '    # Initialize KnowledgeBase and Reasoner\n',
                '    kb = KnowledgeBase(db_path=DB_PATH)\n',
                '    reasoner = SymbolicReasoner(kb, feature_names)\n',
                '    \n',
                '    # Initialize Fuser\n',
                '    fuser = NeuroSymbolicFuser(alpha=0.4)\n',
                '    \n',
                '    # Assuming binary model for now: class 0 is Normal, class 1 is Attack\n',
                '    # Note: If y_cat_test has specific attack types, the neural model might just output binary.\n',
                '    # Let\'s map index 0 to "Normal" and 1 to the actual attack class if it\'s an attack, \n',
                '    # or just use "Attack" as the class name.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'if X_demo is not None:\n',
                '    print("\\n" + "="*80)\n',
                '    print("NEURO-SYMBOLIC FUSION DEMO")\n',
                '    print("="*80 + "\\n")\n',
                '    \n',
                '    with torch.no_grad():\n',
                '        logits = neural_model(torch.tensor(X_demo, dtype=torch.float32)).squeeze(-1)\n',
                '        probs = torch.sigmoid(logits).numpy()\n',
                '        \n',
                '    for i in range(5):\n',
                '        print(f"--- Flow {i+1} ---")\n',
                '        print(f"True Label: {y_demo[i]}")\n',
                '        \n',
                '        # Neural Prediction\n',
                '        # Assuming 1 is Attack, 0 is Normal\n',
                '        prob_attack = float(probs[i])\n',
                '        neural_probs = {"Normal": 1 - prob_attack, "Attack": prob_attack}\n',
                '        print(f"\\n[Neural Model]")\n',
                '        print(f"  P(Normal): {neural_probs[\'Normal\']:.4f}")\n',
                '        print(f"  P(Attack): {neural_probs[\'Attack\']:.4f}")\n',
                '        \n',
                '        # Symbolic Prediction\n',
                '        sym_result = reasoner.predict(X_demo[i])\n',
                '        print(f"\\n[Symbolic Reasoner]")\n',
                '        print(f"  Predicted Class: {sym_result[\'predicted_class\']}")\n',
                '        print(f"  Confidence: {sym_result[\'confidence\']:.4f}")\n',
                '        if sym_result[\'evidence\']:\n',
                '            print(f"  Evidence:")\n',
                '            for ev in sym_result[\'evidence\']:\n',
                '                print(f"    - {ev}")\n',
                '        \n',
                '        # Fusion\n',
                '        fusion_result = fuser.fuse(neural_probs, sym_result)\n',
                '        print(f"\\n[Neuro-Symbolic Fusion]")\n',
                '        print(f"  Final Predicted Class: {fusion_result[\'final_predicted_class\']}")\n',
                '        print(f"  Final Confidence: {fusion_result[\'final_confidence\']:.4f}")\n',
                '        if fusion_result[\'symbolic_explanation\']:\n',
                '            print(f"  Explanation:")\n',
                '            for ex in fusion_result[\'symbolic_explanation\']:\n',
                '                print(f"    - {ex}")\n',
                '                \n',
                '        print("\\n" + "-"*80 + "\\n")\n'
            ]
        }
    ],
    'metadata': {},
    'nbformat': 4,
    'nbformat_minor': 5
}

with open(Path("c:/Users/nisch/OneDrive/Desktop/major/adaptive-neuro-symbolic-ids/notebooks/05_neuro_symbolic_demo.ipynb"), 'w') as f:
    json.dump(notebook, f, indent=1)
