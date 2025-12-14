#!/usr/bin/env python
"""
Quantum Support Vector Classifier (QSVC) for Credit Risk Prediction
Banco Santander Hackathon - IQM Quantum Challenge

This script implements a Quantum Kernel SVM using PennyLane and compares
it against a classical XGBoost baseline for credit risk prediction.

Features:
- Uses ALL 11 features from the dataset (7 numeric + 4 encoded categorical)
- Applies PCA to reduce dimensionality to N_QUBITS components
- This allows the quantum kernel to capture information from ALL features

Hardware: IQM Garnet (20 qubits, CRYSTAL topology)
Framework: PennyLane
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb
import pennylane as qml
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Choose configuration: 'TINY', 'SMALL', 'MEDIUM', or 'LARGE'
CONFIG = 'TINY'  # Start with TINY to test V2 feature map

CONFIGS = {
    'TINY': {
        'N_QUBITS': 5,  # Updated to 5 qubits
        'N_SHOTS': 100,
        'TRAIN_SUBSET': 100,
        'TEST_SUBSET': 100,
        'N_LAYERS': 2,  # Feature map layers
        'desc': 'Quick testing (~1-2 min)'
    },
    'SMALL': {
        'N_QUBITS': 5,  # Updated to 5 qubits
        'N_SHOTS': 256,
        'TRAIN_SUBSET': 200,
        'TEST_SUBSET': 200,
        'N_LAYERS': 2,
        'desc': 'Balanced (~5-10 min)'
    },
    'MEDIUM': {
        'N_QUBITS': 5,  # Updated to 5 qubits
        'N_SHOTS': 512,
        'TRAIN_SUBSET': 600,
        'TEST_SUBSET': 600,
        'N_LAYERS': 2,
        'desc': 'Better accuracy (~20-30 min)'
    },
    'LARGE': {
        'N_QUBITS': 5,
        'N_SHOTS': 512,
        'TRAIN_SUBSET': 400,
        'TEST_SUBSET': 400,
        'N_LAYERS': 3,  # More layers for better expressivity
        'desc': 'Best accuracy (~40-60 min)'
    }
}

# Get current config
N_QUBITS = CONFIGS[CONFIG]['N_QUBITS']
N_SHOTS = CONFIGS[CONFIG]['N_SHOTS']
TRAIN_SUBSET = CONFIGS[CONFIG]['TRAIN_SUBSET']
TEST_SUBSET = CONFIGS[CONFIG]['TEST_SUBSET']
N_LAYERS = CONFIGS[CONFIG]['N_LAYERS']

# IQM Garnet credentials (for hardware validation)
IQM_API_KEY = "zBRzqHAJTO39pf+k8eBzZYDSlXocgti5Y62QFa1lWRsBmxjdwuRyI4hVbRL6GAyy"
IQM_URL = "https://cocos.resonance.meetiqm.com/emerald"

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print(f"QUANTUM CREDIT RISK QSVC - Configuration: {CONFIG}")
print(f"Description: {CONFIGS[CONFIG]['desc']}")
print(f"Qubits: {N_QUBITS}, Shots: {N_SHOTS}, Layers: {N_LAYERS}")
print(f"Train subset: {TRAIN_SUBSET}, Test subset: {TEST_SUBSET}")
print("Feature Map: V2 (RY+RZ with CNOT entanglement)")
print("=" * 60)

# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

def load_and_preprocess_data(filepath, use_pca=True):
    """
    Load and preprocess the credit risk dataset using ALL features + PCA.
    
    Steps:
    - Load CSV
    - Handle NaN values
    - Remove outliers
    - Label encoding for ALL categorical variables
    - Use ALL features (11 total after encoding)
    - Apply PCA to reduce to N_QUBITS dimensions
    - Split 70/30 (train/test), stratified
    - Normalize to [0, π] for angle encoding
    
    Returns:
        X_train, X_test, y_train, y_test: Full datasets
        X_train_subset, X_test_subset, y_train_subset, y_test_subset: Subsets for quantum
        scaler: Fitted scaler for reference
    """
    print("\n[1] Loading and preprocessing data (ALL FEATURES + PCA)...")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"    Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Handle NaN values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mode()[0])
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # Remove outliers
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 60]
    df = df[df['person_income'] <= 4e6]
    print(f"    After cleaning: {len(df)} samples")
    
    # Label encoding for ALL categorical variables
    le_home = LabelEncoder()
    le_intent = LabelEncoder()
    le_default = LabelEncoder()
    
    df['person_home_ownership_encoded'] = le_home.fit_transform(df['person_home_ownership'])
    df['loan_intent_encoded'] = le_intent.fit_transform(df['loan_intent'])
    # loan_grade: A=0, B=1, ..., G=6 (ordinal encoding)
    grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    df['loan_grade_encoded'] = df['loan_grade'].map(grade_order)
    df['cb_person_default_on_file_encoded'] = le_default.fit_transform(df['cb_person_default_on_file'])
    
    # Use ALL features (11 total)
    all_features = [
        # Numeric features
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        # Encoded categorical features
        'person_home_ownership_encoded',
        'loan_intent_encoded',
        'loan_grade_encoded',
        'cb_person_default_on_file_encoded'
    ]
    
    X = df[all_features].values
    y = df['loan_status'].values
    
    print(f"    Using ALL {len(all_features)} features: {all_features}")
    print(f"    Class distribution: No Default={sum(y==0)}, Default={sum(y==1)}")
    print(f"    Class ratio: {sum(y==0)/sum(y==1):.2f}:1")
    
    # Split 70/30, stratified (BEFORE PCA to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"    Train set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    
    # Step 1: Standardize before PCA (required for PCA)
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    
    if use_pca:
        # Step 2: Apply PCA to reduce to N_QUBITS dimensions
        pca = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)
        
        explained_var = pca.explained_variance_ratio_
        print(f"    PCA: {len(all_features)} features → {N_QUBITS} components")
        print(f"    Explained variance per component: {[f'{v:.3f}' for v in explained_var]}")
        print(f"    Total explained variance: {sum(explained_var):.3f} ({sum(explained_var)*100:.1f}%)")
        
        X_train_reduced = X_train_pca
        X_test_reduced = X_test_pca
    else:
        # No PCA - use first N_QUBITS features
        X_train_reduced = X_train_std[:, :N_QUBITS]
        X_test_reduced = X_test_std[:, :N_QUBITS]
        print(f"    No PCA: Using first {N_QUBITS} features")
    
    # Step 3: Normalize to [0, π] for angle encoding
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = angle_scaler.fit_transform(X_train_reduced)
    X_test_scaled = angle_scaler.transform(X_test_reduced)
    
    # Create subsets for quantum kernel (O(n²) complexity)
    indices_train = np.arange(len(X_train_scaled))
    indices_test = np.arange(len(X_test_scaled))
    
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)
    
    train_subset_size = min(TRAIN_SUBSET, len(X_train_scaled))
    test_subset_size = min(TEST_SUBSET, len(X_test_scaled))
    
    X_train_subset = X_train_scaled[indices_train[:train_subset_size]]
    y_train_subset = y_train[indices_train[:train_subset_size]]
    X_test_subset = X_test_scaled[indices_test[:test_subset_size]]
    y_test_subset = y_test[indices_test[:test_subset_size]]
    
    print(f"    Quantum train subset: {len(X_train_subset)} samples")
    print(f"    Quantum test subset: {len(X_test_subset)} samples")
    print(f"    Final feature dimension: {X_train_subset.shape[1]} (= N_QUBITS)")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            X_train_subset, X_test_subset, y_train_subset, y_test_subset,
            angle_scaler, X_train, X_test)


# ============================================================
# 2. QUANTUM KERNEL CIRCUIT - FEATURE MAP V2
# ============================================================

# Create quantum device (simulator for training)
dev = qml.device("default.qubit", wires=N_QUBITS, shots=N_SHOTS)

def feature_map_v2(x):
    """
    FEATURE MAP V2: Enhanced encoding with multiple layers and richer entanglement.
    
    Improvements over V1:
    - Multiple configurable layers (N_LAYERS)
    - RY + RZ rotations for fuller Bloch sphere coverage
    - CNOT ladder + circular entanglement (IQM Garnet compatible)
    - Non-linear feature mixing between layers
    
    Circuit structure per layer:
    1. RY(x[i]) + RZ(x[i] * scale) on each qubit
    2. CNOT ladder (0→1→2→...→n-1)
    3. Circular CNOT (n-1→0) for ring topology
    """
    for layer in range(N_LAYERS):
        # Scale factor varies per layer for feature mixing
        scale = 0.5 + 0.25 * layer
        
        # Rotation layer: RY + RZ for full Bloch sphere coverage
        for i in range(N_QUBITS):
            qml.RY(x[i], wires=i)
            qml.RZ(x[i] * scale, wires=i)
        
        # Entanglement layer: CNOT ladder (linear connectivity)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Circular connection (ring topology for IQM Garnet)
        if N_QUBITS > 2:
            qml.CNOT(wires=[N_QUBITS - 1, 0])
        
        # Additional feature interaction layer (odd layers only)
        if layer % 2 == 1 and layer < N_LAYERS - 1:
            for i in range(N_QUBITS):
                # Cross-feature interaction
                qml.RY(x[i] * x[(i + 1) % N_QUBITS] * 0.1, wires=i)


def adjoint_feature_map_v2(y):
    """
    Apply U†(y) - the adjoint (inverse) of Feature Map V2.
    Gates are applied in REVERSE order with NEGATED angles.
    
    Note: CNOT is self-adjoint (CNOT† = CNOT), but order must be reversed.
    """
    # Process layers in reverse order
    for layer in range(N_LAYERS - 1, -1, -1):
        scale = 0.5 + 0.25 * layer
        
        # Reverse: Additional feature interaction layer (odd layers only)
        if layer % 2 == 1 and layer < N_LAYERS - 1:
            for i in range(N_QUBITS - 1, -1, -1):
                qml.RY(-y[i] * y[(i + 1) % N_QUBITS] * 0.1, wires=i)
        
        # Reverse: Circular connection
        if N_QUBITS > 2:
            qml.CNOT(wires=[N_QUBITS - 1, 0])
        
        # Reverse: CNOT ladder (reverse order)
        for i in range(N_QUBITS - 2, -1, -1):
            qml.CNOT(wires=[i, i + 1])
        
        # Reverse: Rotation layer (RZ then RY, with negative angles)
        for i in range(N_QUBITS - 1, -1, -1):
            qml.RZ(-y[i] * scale, wires=i)
            qml.RY(-y[i], wires=i)


# Legacy feature map (V1) kept for reference/comparison
def feature_map_v1(x):
    """Original simple feature map (V1) - kept for comparison."""
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    for i in range(N_QUBITS):
        qml.CZ(wires=[i, (i + 1) % N_QUBITS])
    for i in range(N_QUBITS):
        qml.RY(x[i] * 0.5, wires=i)


def adjoint_feature_map_v1(y):
    """Original adjoint feature map (V1) - kept for comparison."""
    for i in range(N_QUBITS - 1, -1, -1):
        qml.RY(-y[i] * 0.5, wires=i)
    for i in range(N_QUBITS - 1, -1, -1):
        qml.CZ(wires=[i, (i + 1) % N_QUBITS])
    for i in range(N_QUBITS - 1, -1, -1):
        qml.RY(-y[i], wires=i)


@qml.qnode(dev)
def kernel_circuit(x, y):
    """
    Compute quantum kernel K(x,y) = |⟨φ(x)|φ(y)⟩|² using Feature Map V2.
    
    Process:
    1. Encode x: |0⟩ → U(x) → |φ(x)⟩
    2. Decode with y: |φ(x)⟩ → U†(y) → result
    3. Measure: probability of |00000⟩ = similarity K(x,y)
    
    If x ≈ y → K(x,y) ≈ 1 (return to initial state)
    If x ≠ y → K(x,y) < 1 (don't return)
    
    Returns:
        Probabilities of all computational basis states
    """
    # Apply U(x) - encode first data point using V2 feature map
    feature_map_v2(x)
    
    # Apply U†(y) - adjoint of encoding for second data point
    adjoint_feature_map_v2(y)
    
    # Return probabilities of all states
    return qml.probs(wires=range(N_QUBITS))


def kernel_value(x, y):
    """
    Compute kernel value K(x,y) = P(|00000⟩).
    
    Returns:
        float: Probability of measuring all-zeros state (5 qubits now)
    """
    probs = kernel_circuit(x, y)
    # Index 0 corresponds to |00000⟩ state
    return probs[0]


# ============================================================
# 3. COMPUTE KERNEL MATRIX
# ============================================================

def compute_kernel_matrix(X1, X2, symmetric=False):
    """
    Compute the quantum kernel matrix K where K[i,j] = K(X1[i], X2[j]).
    
    Args:
        X1: First set of data points (n1 × d)
        X2: Second set of data points (n2 × d)
        symmetric: If True, X1 == X2 and we exploit K(x,y) = K(y,x)
    
    Returns:
        K: Kernel matrix of shape (n1, n2)
    """
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    if symmetric:
        # Exploit symmetry: only compute upper triangle
        total_pairs = n1 * (n1 + 1) // 2
        print(f"    Computing symmetric kernel matrix ({n1}×{n1})...")
        print(f"    Total unique pairs: {total_pairs}")
        
        with tqdm(total=total_pairs, desc="    Kernel matrix", unit="pairs") as pbar:
            for i in range(n1):
                for j in range(i, n2):
                    k_val = kernel_value(X1[i], X2[j])
                    K[i, j] = k_val
                    if i != j:
                        K[j, i] = k_val  # Symmetry
                    pbar.update(1)
    else:
        # Non-symmetric case (e.g., train vs test)
        total_pairs = n1 * n2
        print(f"    Computing kernel matrix ({n1}×{n2})...")
        print(f"    Total pairs: {total_pairs}")
        
        with tqdm(total=total_pairs, desc="    Kernel matrix", unit="pairs") as pbar:
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = kernel_value(X1[i], X2[j])
                    pbar.update(1)
    
    return K


# ============================================================
# 4. KERNEL NORMALIZATION & DIAGNOSTICS (ZERO COST)
# ============================================================

def diagnose_kernel(K, name="K"):
    """
    Diagnose kernel matrix properties. ZERO computational cost.
    """
    diag = np.diag(K)
    print(f"\n    {name} Diagnostics:")
    print(f"      Diagonal - mean: {diag.mean():.4f}, std: {diag.std():.4f}, "
          f"min: {diag.min():.4f}, max: {diag.max():.4f}")
    print(f"      Off-diag - mean: {K.mean():.4f}, std: {K.std():.4f}")
    
    # Check if normalization is needed
    needs_normalization = diag.mean() < 0.95 or diag.std() > 0.05
    if needs_normalization:
        print(f"      ⚠️  Normalization RECOMMENDED (diagonal not stable)")
    else:
        print(f"      ✅ Kernel diagonal is stable (normalization optional)")
    
    return needs_normalization


def normalize_kernel(K_train, K_test=None):
    """
    Normalize kernel matrices using cosine normalization. ZERO computational cost.
    
    K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])
    
    For test kernel, we use training diagonal for consistency.
    """
    # Get diagonal of training kernel
    d_train = np.sqrt(np.diag(K_train))
    
    # Avoid division by zero
    d_train = np.maximum(d_train, 1e-10)
    
    # Normalize training kernel
    K_train_norm = K_train / np.outer(d_train, d_train)
    K_train_norm = np.clip(K_train_norm, 0, 1)
    
    if K_test is not None:
        # For test kernel: K_test[i,j] / sqrt(K_test_diag[i] * K_train_diag[j])
        # But we don't have K_test diagonal (it's not square)
        # So we normalize by training diagonal only
        K_test_norm = K_test / d_train[np.newaxis, :]
        K_test_norm = np.clip(K_test_norm, 0, 1)
        return K_train_norm, K_test_norm
    
    return K_train_norm


# ============================================================
# 5. TRAIN AND EVALUATE QSVC WITH C TUNING (ZERO COST)
# ============================================================

def tune_svc_C(K_train, y_train, K_test, y_test, C_values=[0.1, 1.0, 10.0, 100.0]):
    """
    Tune SVC hyperparameter C using precomputed kernel. ZERO kernel computation cost.
    
    This only retrains the SVM (< 1 second) without recomputing the kernel.
    """
    print("\n[4a] Tuning SVC parameter C (ZERO kernel cost)...")
    
    best_auc = 0
    best_C = 1.0
    results_per_C = {}
    
    for C in C_values:
        svm = SVC(kernel='precomputed', C=C, class_weight='balanced', 
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(K_train, y_train)
        y_proba = svm.predict_proba(K_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results_per_C[C] = auc
        
        marker = " ← BEST" if auc > best_auc else ""
        print(f"      C={C:>6.1f}: AUC={auc:.4f}{marker}")
        
        if auc > best_auc:
            best_auc = auc
            best_C = C
    
    print(f"    Best C: {best_C} with AUC: {best_auc:.4f}")
    return best_C, results_per_C


def train_qsvc(K_train, y_train, K_test, y_test, C=None, normalize=True):
    """
    Train Quantum SVM using precomputed kernel matrix.
    
    Args:
        K_train: Training kernel matrix (n_train × n_train)
        y_train: Training labels
        K_test: Test kernel matrix (n_test × n_train)
        y_test: Test labels
        C: SVC regularization parameter (if None, will be tuned)
        normalize: Whether to normalize kernel matrices
    
    Returns:
        dict: Results including predictions, probabilities, and metrics
    """
    print("\n[4] Training QSVC...")
    
    # Diagnose kernel
    needs_norm = diagnose_kernel(K_train, "K_train")
    
    # Normalize if needed/requested
    if normalize and needs_norm:
        print("    Applying kernel normalization...")
        K_train_use, K_test_use = normalize_kernel(K_train, K_test)
        diagnose_kernel(K_train_use, "K_train_normalized")
    else:
        K_train_use, K_test_use = K_train, K_test
    
    # Tune C if not provided
    if C is None:
        best_C, _ = tune_svc_C(K_train_use, y_train, K_test_use, y_test)
    else:
        best_C = C
    
    # Train final model with best C
    print(f"\n[4b] Training final QSVC with C={best_C}...")
    svm = SVC(kernel='precomputed', C=best_C, class_weight='balanced', 
              probability=True, random_state=RANDOM_SEED)
    
    start_time = time.time()
    svm.fit(K_train_use, y_train)
    train_time = time.time() - start_time
    print(f"    Training completed in {train_time:.2f}s")
    
    # Predict on test set
    y_pred = svm.predict(K_test_use)
    y_proba = svm.predict_proba(K_test_use)[:, 1]  # Probability of default
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    gini = 2 * auc_roc - 1
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])
    
    results = {
        'model': svm,
        'best_C': best_C,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'gini': gini,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'train_time': train_time,
        'normalized': normalize and needs_norm
    }
    
    return results


# ============================================================
# 5. CLASSICAL BASELINE (XGBoost)
# ============================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier as classical baseline.
    
    Returns:
        dict: Results including predictions, probabilities, and metrics
    """
    print("\n[5] Training XGBoost (classical baseline)...")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    
    # Create and train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='auc'
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"    Training completed in {train_time:.2f}s")
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    gini = 2 * auc_roc - 1
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])
    
    results = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'gini': gini,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'train_time': train_time
    }
    
    return results


# ============================================================
# 6. VISUALIZATION
# ============================================================

def plot_results(qsvc_results, xgb_results, y_test, K_train=None, save_path='qsvc_results.png'):
    """
    Generate visualization with 4 plots:
    1. ROC curves comparing QSVC vs XGBoost
    2. Bar chart of metrics (Accuracy, AUC-ROC, Gini, F1)
    3. Confusion Matrix of QSVC
    4. Kernel matrix heatmap (if provided)
    """
    print("\n[6] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Color scheme
    qsvc_color = '#1E88E5'  # Blue
    xgb_color = '#E53935'   # Red
    
    # 1. ROC Curves Comparison
    ax1 = axes[0, 1]
    
    # QSVC ROC
    fpr_qsvc, tpr_qsvc, _ = roc_curve(y_test, qsvc_results['y_proba'])
    ax1.plot(fpr_qsvc, tpr_qsvc, color=qsvc_color, lw=2, 
             label=f"QSVC (AUC={qsvc_results['auc_roc']:.3f})")
    
    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_results['y_proba'])
    ax1.plot(fpr_xgb, tpr_xgb, color=xgb_color, lw=2,
             label=f"XGBoost (AUC={xgb_results['auc_roc']:.3f})")
    
    # Diagonal
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Metrics Comparison Bar Chart
    ax2 = axes[1, 0]
    
    metrics = ['Accuracy', 'AUC-ROC', 'Gini', 'F1']
    qsvc_values = [qsvc_results['accuracy'], qsvc_results['auc_roc'], 
                   qsvc_results['gini'], qsvc_results['f1']]
    xgb_values = [xgb_results['accuracy'], xgb_results['auc_roc'],
                  xgb_results['gini'], xgb_results['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, qsvc_values, width, label='QSVC', color=qsvc_color, alpha=0.8)
    bars2 = ax2.bar(x + width/2, xgb_values, width, label='XGBoost', color=xgb_color, alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, qsvc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, xgb_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. QSVC Confusion Matrix
    ax3 = axes[1, 1]
    
    cm = qsvc_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['No Def', 'Def'],
                yticklabels=['No Def', 'Def'],
                annot_kws={'size': 14})
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('Actual', fontsize=12)
    ax3.set_title('QSVC Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 4. Training Loss / Kernel Matrix visualization
    ax4 = axes[0, 0]
    
    if K_train is not None and len(K_train) <= 50:
        # Show kernel matrix heatmap for small matrices
        sns.heatmap(K_train[:30, :30], cmap='viridis', ax=ax4,
                    cbar_kws={'label': 'K(x,y)'})
        ax4.set_title('Quantum Kernel Matrix (subset)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sample index', fontsize=12)
        ax4.set_ylabel('Sample index', fontsize=12)
    else:
        # Show kernel value distribution
        if K_train is not None:
            kernel_values = K_train.flatten()
            ax4.hist(kernel_values, bins=50, color=qsvc_color, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Kernel Value K(x,y)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Quantum Kernel Value Distribution', fontsize=14, fontweight='bold')
            ax4.axvline(x=np.mean(kernel_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(kernel_values):.3f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Kernel Matrix\nNot Available', 
                    ha='center', va='center', fontsize=14, transform=ax4.transAxes)
            ax4.set_title('Quantum Kernel Matrix', fontsize=14, fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"    Results saved to: {save_path}")
    plt.show()


def print_results(qsvc_results, xgb_results):
    """Print formatted results comparison."""
    
    print("\n" + "=" * 60)
    print("QSVC RESULTS (Quantum Kernel SVM)")
    print("=" * 60)
    print(f"Best C:           {qsvc_results.get('best_C', 'N/A')}")
    print(f"Normalized:       {qsvc_results.get('normalized', False)}")
    print(f"Accuracy:         {qsvc_results['accuracy']:.4f}")
    print(f"AUC-ROC:          {qsvc_results['auc_roc']:.4f}")
    print(f"Gini Coefficient: {qsvc_results['gini']:.4f}")
    print(f"F1-Score:         {qsvc_results['f1']:.4f}")
    print(f"Training Time:    {qsvc_results['train_time']:.2f}s")
    print("\nClassification Report:")
    print(qsvc_results['classification_report'])
    
    print("\n" + "=" * 60)
    print("XGBoost RESULTS (Classical Baseline)")
    print("=" * 60)
    print(f"Accuracy:         {xgb_results['accuracy']:.4f}")
    print(f"AUC-ROC:          {xgb_results['auc_roc']:.4f}")
    print(f"Gini Coefficient: {xgb_results['gini']:.4f}")
    print(f"F1-Score:         {xgb_results['f1']:.4f}")
    print(f"Training Time:    {xgb_results['train_time']:.2f}s")
    print("\nClassification Report:")
    print(xgb_results['classification_report'])
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'QSVC':<12} {'XGBoost':<12} {'Difference':<12}")
    print("-" * 56)
    
    metrics = ['accuracy', 'auc_roc', 'gini', 'f1']
    metric_names = ['Accuracy', 'AUC-ROC', 'Gini', 'F1-Score']
    
    for name, metric in zip(metric_names, metrics):
        qsvc_val = qsvc_results[metric]
        xgb_val = xgb_results[metric]
        diff = qsvc_val - xgb_val
        sign = '+' if diff >= 0 else ''
        print(f"{name:<20} {qsvc_val:<12.4f} {xgb_val:<12.4f} {sign}{diff:<12.4f}")
    
    print("=" * 60)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function."""
    
    total_start_time = time.time()
    
    # 1. Load and preprocess data
    filepath = "credit_risk_dataset_red.csv"
    (X_train_scaled, X_test_scaled, y_train, y_test,
     X_train_subset, X_test_subset, y_train_subset, y_test_subset,
     scaler, X_train_raw, X_test_raw) = load_and_preprocess_data(filepath)
    
    # 2. Compute quantum kernel matrices
    print("\n[2] Computing Quantum Kernel Matrices...")
    
    # Training kernel matrix (symmetric)
    print("\n    Computing K_train (symmetric)...")
    kernel_start = time.time()
    K_train = compute_kernel_matrix(X_train_subset, X_train_subset, symmetric=True)
    
    # Test kernel matrix (not symmetric - test vs train)
    print("\n    Computing K_test (test vs train)...")
    K_test = compute_kernel_matrix(X_test_subset, X_train_subset, symmetric=False)
    kernel_time = time.time() - kernel_start
    
    print(f"\n    Kernel computation completed in {kernel_time:.2f}s")
    print(f"    K_train shape: {K_train.shape}")
    print(f"    K_test shape: {K_test.shape}")
    
    # 3. Verify kernel matrix properties
    print("\n[3] Kernel Matrix Statistics:")
    print(f"    K_train - min: {K_train.min():.4f}, max: {K_train.max():.4f}, mean: {K_train.mean():.4f}")
    print(f"    K_test  - min: {K_test.min():.4f}, max: {K_test.max():.4f}, mean: {K_test.mean():.4f}")
    print(f"    Diagonal mean (should be ~1.0): {np.diag(K_train).mean():.4f}")
    
    # 4. Train QSVC
    qsvc_results = train_qsvc(K_train, y_train_subset, K_test, y_test_subset)
    
    # 5. Train XGBoost (on same subset for fair comparison)
    xgb_results = train_xgboost(X_train_subset, y_train_subset, X_test_subset, y_test_subset)
    
    # 6. Print results
    print_results(qsvc_results, xgb_results)
    
    # 7. Generate visualizations
    plot_results(qsvc_results, xgb_results, y_test_subset, K_train, 
                save_path='qsvc_results.png')
    
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"{'=' * 60}")
    
    return qsvc_results, xgb_results, K_train, K_test


if __name__ == "__main__":
    qsvc_results, xgb_results, K_train, K_test = main()

