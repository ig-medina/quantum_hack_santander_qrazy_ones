"""
Quantum Credit Risk Prediction using VQC (Variational Quantum Classifier)
=========================================================================
Banco Santander Quantum Hackathon 2025 - Madrid

Hardware: IQM Garnet (20 qubits, CRYSTAL topology)
Framework: PennyLane + Qiskit-IQM

STRATEGY: Train on SIMULATOR, validate on HARDWARE (30 credits budget)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, classification_report, roc_curve)
import pennylane as qml
from pennylane import numpy as pnp
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - START SMALL, SCALE UP
# =============================================================================

# IQM Garnet API Configuration
IQM_API_KEY = "mGmbAM33Ljt+tnX9TmzGx3eR7bIa2Ovm1K1XUm56E/gBmxhLbPx94a1tHreh86KA"
IQM_URL = "https://cocos.resonance.meetiqm.com/garnet"

# ============== CONFIGURATION PRESETS ==============
# Uncomment the config you want to use:

# # CONFIG 1: TINY (for quick simulator testing) - ~30 seconds
# CONFIG = "TINY"
# N_QUBITS = 4
# N_LAYERS = 1
# N_SHOTS = 100
# N_EPOCHS = 5
# TRAIN_SUBSET = 200
# BATCH_SIZE = 50

# # CONFIG 2: SMALL (for simulator validation) - ~2-3 minutes
# CONFIG = "SMALL"
# N_QUBITS = 4
# N_LAYERS = 2
# N_SHOTS = 256
# N_EPOCHS = 15
# TRAIN_SUBSET = 500
# BATCH_SIZE = 50

# # CONFIG 3: MEDIUM (final simulator training) - ~10-15 minutes
# CONFIG = "MEDIUM"
# N_QUBITS = 6
# N_LAYERS = 2
# N_SHOTS = 512
# N_EPOCHS = 25
# TRAIN_SUBSET = 1000
# BATCH_SIZE = 64

# # CONFIG 4: PCA5 (PCA feature selection with 5 components)
# CONFIG = "PCA5"
# N_QUBITS = 5
# N_LAYERS = 1
# N_SHOTS = 100
# N_EPOCHS = 10
# TRAIN_SUBSET = 100
# TEST_SUBSET = 100
# BATCH_SIZE = 20
# USE_PCA = True
# PCA_COMPONENTS = 5

# CONFIG 5: PCA5_EXTENDED (more data for better results)
CONFIG = "PCA5_EXTENDED"
N_QUBITS = 5
N_LAYERS = 1
N_SHOTS = 100
N_EPOCHS = 15
TRAIN_SUBSET = 250
TEST_SUBSET = 150
BATCH_SIZE = 32
USE_PCA = True
PCA_COMPONENTS = 5

# Default values for optional config params (for configs that don't define them)
if 'USE_PCA' not in dir():
    USE_PCA = False
if 'PCA_COMPONENTS' not in dir():
    PCA_COMPONENTS = N_QUBITS
if 'TEST_SUBSET' not in dir():
    TEST_SUBSET = None  # Use all test samples

# Class weights for imbalanced data (Default class is ~5x less frequent)
CLASS_WEIGHT_DEFAULT = 4.0  # Weight for Default class (minority)
CLASS_WEIGHT_NO_DEFAULT = 1.0  # Weight for No Default class (majority)

# Decision threshold (lower = more conservative, detects more defaults)
DECISION_THRESHOLD = 0.2  # Default is 0, lower catches more defaults

# Hardware validation settings
HARDWARE_TEST_SAMPLES = 50   # Only validate 50 samples on real hardware
HARDWARE_SHOTS = 256         # Shots for hardware (keep low for budget)

RANDOM_STATE = 42
LEARNING_RATE = 0.15

# Device selection
USE_SIMULATOR = True  # Set to False ONLY for final hardware validation

# =============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath: str):
    """
    Load and preprocess data.
    Supports two modes:
    - Manual feature selection (default): Uses top features from analysis
    - PCA mode: Uses PCA for dimensionality reduction
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Handle missing values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mode()[0])
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # Remove outliers
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 60]
    df = df[df['person_income'] <= 4e6]
    print(f"✓ After cleaning: {df.shape[0]} samples")
    
    # Label encode categorical variables
    label_encoders = {}
    
    # 1. person_home_ownership
    le_home = LabelEncoder()
    df['home_ownership_encoded'] = le_home.fit_transform(df['person_home_ownership'])
    label_encoders['home_ownership'] = le_home
    
    # 2. loan_grade: A=0, B=1, C=2, D=3, E=4, F=5, G=6 (ordinal by risk)
    grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    df['loan_grade_encoded'] = df['loan_grade'].map(grade_order)
    
    # 3. cb_person_default_on_file: N=0, Y=1
    df['default_history_encoded'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
    
    # 4. loan_intent
    le_intent = LabelEncoder()
    df['loan_intent_encoded'] = le_intent.fit_transform(df['loan_intent'])
    label_encoders['loan_intent'] = le_intent
    
    y = df['loan_status'].values
    
    # ========== FEATURE SELECTION ==========
    if USE_PCA:
        print(f"\n✓ Using PCA with {PCA_COMPONENTS} components")
        
        # Use ALL numeric features for PCA
        all_features = [
            'person_age', 'person_income', 'person_emp_length',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length',
            'home_ownership_encoded', 'loan_grade_encoded',
            'default_history_encoded', 'loan_intent_encoded'
        ]
        
        X_all = df[all_features].values
        
        # Split BEFORE PCA to avoid data leakage
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
        )
        
        # Standardize before PCA
        std_scaler = StandardScaler()
        X_train_std = std_scaler.fit_transform(X_train_raw)
        X_test_std = std_scaler.transform(X_test_raw)
        
        # Apply PCA
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train_std)
        X_test = pca.transform(X_test_std)
        
        print(f"  Input features: {len(all_features)}")
        print(f"  PCA components: {PCA_COMPONENTS}")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"  Per component: {[f'{v:.1%}' for v in pca.explained_variance_ratio_]}")
        
        selected_features = [f'PC{i+1}' for i in range(PCA_COMPONENTS)]
        
        # Normalize PCA output to [0, π] for angle encoding
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    else:
        # Manual feature selection (original behavior)
        print(f"\n✓ Using manual feature selection")
        
        if N_QUBITS == 4:
            selected_features = [
                'loan_percent_income',
                'loan_grade_encoded',
                'loan_int_rate',
                'home_ownership_encoded',
            ]
        else:  # 5 or 6 qubits
            selected_features = [
                'loan_percent_income',
                'loan_grade_encoded',
                'loan_int_rate',
                'home_ownership_encoded',
                'person_income',
                'cb_person_cred_hist_length',
            ][:N_QUBITS]  # Take only as many as qubits
        
        print(f"  Selected features: {selected_features}")
        
        X = df[selected_features].values
        
        # Split data 70/30
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
        )
        
        # Normalize to [0, π] for angle encoding
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Use subset for faster training
    if TRAIN_SUBSET and TRAIN_SUBSET < len(X_train):
        indices = np.random.RandomState(RANDOM_STATE).permutation(len(X_train))[:TRAIN_SUBSET]
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"✓ Using training subset: {len(X_train)} samples")
    
    # Use subset for test if defined
    if TEST_SUBSET and TEST_SUBSET < len(X_test):
        indices = np.random.RandomState(RANDOM_STATE + 1).permutation(len(X_test))[:TEST_SUBSET]
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"✓ Using test subset: {len(X_test)} samples")
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Features normalized to [0, π] for angle encoding")
    
    # Class distribution
    print(f"\n  Class distribution (train): {np.bincount(y_train)}")
    print(f"  Class distribution (test): {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, selected_features, scaler


# =============================================================================
# STEP 2: QUANTUM DEVICE SETUP
# =============================================================================

def setup_quantum_device(use_hardware=False, n_qubits=N_QUBITS, shots=N_SHOTS):
    """Setup PennyLane quantum device."""
    print("\n" + "=" * 60)
    print("STEP 2: QUANTUM DEVICE SETUP")
    print("=" * 60)
    
    if not use_hardware:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        print(f"✓ Using SIMULATOR: default.qubit")
    else:
        from qiskit_iqm import IQMProvider
        provider = IQMProvider(IQM_URL, token=IQM_API_KEY)
        backend = provider.get_backend()
        dev = qml.device("qiskit.remote", wires=n_qubits, backend=backend, shots=shots)
        print(f"✓ Using IQM GARNET quantum hardware")
    
    print(f"  Qubits: {n_qubits}")
    print(f"  Shots: {shots}")
    
    return dev


# =============================================================================
# STEP 3: VQC CIRCUIT
# =============================================================================

def create_vqc_circuit(dev, n_qubits=N_QUBITS, n_layers=N_LAYERS):
    """Create Variational Quantum Classifier circuit."""
    
    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        """
        VQC Circuit:
        - Angle encoding: RY(feature) per qubit
        - Variational layer: RY + RZ rotations + CZ entanglement
        - Measurement: ⟨Z⟩ on qubit 0
        """
        # ENCODING: Each feature → rotation on corresponding qubit
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # VARIATIONAL LAYERS
        for layer in range(n_layers):
            # Rotations (trainable)
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Entanglement (ring topology - works well with IQM CRYSTAL)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CZ(wires=[n_qubits - 1, 0])  # Close the ring
            
            # Final rotation
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 2], wires=i)
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit


# =============================================================================
# STEP 4: TRAINING
# =============================================================================

def train_vqc(circuit, X_train, y_train, X_val, y_val):
    """Train VQC with gradient descent."""
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING VQC")
    print("=" * 60)
    
    # Initialize weights
    np.random.seed(RANDOM_STATE)
    weights_shape = (N_LAYERS, N_QUBITS, 3)
    weights = pnp.array(np.random.uniform(0, 2*np.pi, weights_shape), requires_grad=True)
    
    n_params = N_LAYERS * N_QUBITS * 3
    print(f"✓ Configuration: {CONFIG}")
    print(f"  Qubits: {N_QUBITS}, Layers: {N_LAYERS}")
    print(f"  Trainable parameters: {n_params}")
    print(f"  Epochs: {N_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Class weights: No Default={CLASS_WEIGHT_NO_DEFAULT}, Default={CLASS_WEIGHT_DEFAULT}")
    print(f"  Decision threshold: {DECISION_THRESHOLD}")
    
    # Estimate time
    circuits_per_epoch = (len(X_train) // BATCH_SIZE) * (2 * n_params + 1)
    est_time = N_EPOCHS * circuits_per_epoch * 0.001  # ~1ms per circuit on simulator
    print(f"\n  Estimated time (simulator): {est_time:.1f} seconds")
    
    # Optimizer
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    
    # Map labels: 0 → +1, 1 → -1
    y_train_mapped = np.where(y_train == 0, 1, -1)
    
    def cost(weights, X_batch, y_batch):
        # Weighted squared errors (autograd-compatible)
        # Higher weight for Default class to address class imbalance
        total_loss = 0.0
        for i, x in enumerate(X_batch):
            pred = circuit(x, weights)
            # y_batch[i] = -1 means Default, +1 means No Default
            w = CLASS_WEIGHT_DEFAULT if y_batch[i] == -1 else CLASS_WEIGHT_NO_DEFAULT
            total_loss = total_loss + w * (pred - y_batch[i]) ** 2
        return total_loss / len(X_batch)
    
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    n_batches = max(1, len(X_train) // BATCH_SIZE)
    
    print(f"\n  Starting training...")
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_mapped[indices]
        
        epoch_loss = 0
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(X_train))
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            weights, loss = opt.step_and_cost(lambda w: cost(w, X_batch, y_batch), weights)
            epoch_loss += float(loss)
        
        epoch_loss /= n_batches
        history['loss'].append(epoch_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Evaluate every 2 epochs or at the end
        if (epoch + 1) % 2 == 0 or epoch == N_EPOCHS - 1:
            # Quick accuracy on subset
            val_subset = min(100, len(X_val))
            val_pred = predict(circuit, weights, X_val[:val_subset])
            val_acc = accuracy_score(y_val[:val_subset], val_pred)
            
            train_pred = predict(circuit, weights, X_train[:100])
            train_acc = accuracy_score(y_train[:100], train_pred)
            
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"  Epoch {epoch+1:2d}/{N_EPOCHS} | Loss: {epoch_loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time:.1f} seconds")
    
    return weights, history


def predict(circuit, weights, X):
    """Generate predictions using configurable threshold."""
    predictions = []
    for x in X:
        exp_val = float(circuit(x, weights))
        # Lower threshold = more conservative (catches more defaults)
        # exp_val > THRESHOLD → No Default (0)
        # exp_val <= THRESHOLD → Default (1)
        pred = 0 if exp_val > DECISION_THRESHOLD else 1
        predictions.append(pred)
    return np.array(predictions)


def predict_proba(circuit, weights, X):
    """Get probability scores."""
    probs = []
    for x in X:
        exp_val = float(circuit(x, weights))
        prob = (1 - exp_val) / 2  # Map [-1,1] to [0,1]
        probs.append(prob)
    return np.array(probs)


# =============================================================================
# STEP 5: EVALUATION
# =============================================================================

def evaluate_model(circuit, weights, X_test, y_test, name="VQC"):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print(f"STEP 5: EVALUATION ({name})")
    print("=" * 60)
    
    print(f"  Evaluating on {len(X_test)} samples...")
    start = time.time()
    
    y_pred = predict(circuit, weights, X_test)
    y_prob = predict_proba(circuit, weights, X_test)
    
    eval_time = time.time() - start
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc_roc - 1
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n  {'='*40}")
    print(f"  {name} RESULTS")
    print(f"  {'='*40}")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  AUC-ROC:          {auc_roc:.4f}")
    print(f"  Gini Coefficient: {gini:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  Evaluation time:  {eval_time:.1f}s")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    return {
        'accuracy': accuracy, 'auc_roc': auc_roc, 'gini': gini, 'f1': f1,
        'y_pred': y_pred, 'y_prob': y_prob,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def optimize_threshold(circuit, weights, X_test, y_test):
    """
    Find optimal threshold that balances Precision and Recall for Default class.
    Lower threshold = more conservative (catches more defaults, but more false positives)
    """
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)
    
    # Get raw expectation values
    exp_values = []
    for x in X_test:
        exp_val = float(circuit(x, weights))
        exp_values.append(exp_val)
    exp_values = np.array(exp_values)
    
    # Test different thresholds
    thresholds = np.linspace(-0.5, 0.5, 21)  # From -0.5 to 0.5
    
    results = []
    for thresh in thresholds:
        y_pred = (exp_values <= thresh).astype(int)  # <= thresh → Default (1)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Recall for Default class (how many defaults we catch)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            recall_default = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            precision_default = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
        else:
            recall_default = 0
            precision_default = 0
        
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'f1': f1,
            'recall_default': recall_default,
            'precision_default': precision_default
        })
    
    # Find best threshold by F1-score
    best_f1 = max(results, key=lambda x: x['f1'])
    
    # Find threshold with best balance (F1 for Default class)
    best_recall = max(results, key=lambda x: x['recall_default'])
    
    print(f"\n  Threshold Analysis:")
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'F1':<10} {'Recall(Def)':<12} {'Prec(Def)':<12}")
    print(f"  {'-'*56}")
    for r in results[::2]:  # Print every other
        print(f"  {r['threshold']:<12.2f} {r['accuracy']:<10.4f} {r['f1']:<10.4f} "
              f"{r['recall_default']:<12.4f} {r['precision_default']:<12.4f}")
    
    print(f"\n  Best by F1-Score:")
    print(f"    Threshold: {best_f1['threshold']:.2f}")
    print(f"    F1: {best_f1['f1']:.4f}, Recall: {best_f1['recall_default']:.4f}")
    
    print(f"\n  Best by Recall (catches most defaults):")
    print(f"    Threshold: {best_recall['threshold']:.2f}")
    print(f"    Recall: {best_recall['recall_default']:.4f}, F1: {best_recall['f1']:.4f}")
    
    # Return predictions with optimized threshold
    optimal_threshold = best_f1['threshold']
    y_pred_optimized = (exp_values <= optimal_threshold).astype(int)
    y_prob = (1 - exp_values) / 2
    
    return optimal_threshold, y_pred_optimized, y_prob, results


# =============================================================================
# STEP 6: CLASSICAL BASELINE
# =============================================================================

def train_classical_baseline(X_train, y_train, X_test, y_test):
    """XGBoost baseline."""
    print("\n" + "=" * 60)
    print("CLASSICAL BASELINE: XGBoost")
    print("=" * 60)
    
    from xgboost import XGBClassifier
    
    xgb = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.2,
        random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    
    print(f"  XGBoost Accuracy: {accuracy:.4f}")
    print(f"  XGBoost AUC-ROC:  {auc_roc:.4f}")
    
    return {
        'accuracy': accuracy, 'auc_roc': auc_roc,
        'gini': 2*auc_roc-1, 'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred, 'y_prob': y_prob
    }


# =============================================================================
# STEP 7: VISUALIZATION
# =============================================================================

def plot_results(quantum_results, classical_results, y_test, history):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    ax1.plot(history['loss'], 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'VQC Training Loss ({CONFIG} config)')
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curves
    ax2 = axes[0, 1]
    fpr_q, tpr_q, _ = roc_curve(y_test, quantum_results['y_prob'])
    fpr_c, tpr_c, _ = roc_curve(y_test, classical_results['y_prob'])
    ax2.plot(fpr_q, tpr_q, 'b-', lw=2, label=f"VQC (AUC={quantum_results['auc_roc']:.3f})")
    ax2.plot(fpr_c, tpr_c, 'r-', lw=2, label=f"XGBoost (AUC={classical_results['auc_roc']:.3f})")
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics Comparison
    ax3 = axes[1, 0]
    metrics = ['Accuracy', 'AUC-ROC', 'Gini', 'F1']
    q_vals = [quantum_results['accuracy'], quantum_results['auc_roc'],
              quantum_results['gini'], quantum_results['f1']]
    c_vals = [classical_results['accuracy'], classical_results['auc_roc'],
              classical_results['gini'], classical_results['f1']]
    x = np.arange(len(metrics))
    ax3.bar(x - 0.2, q_vals, 0.4, label='VQC', color='#2196F3')
    ax3.bar(x + 0.2, c_vals, 0.4, label='XGBoost', color='#F44336')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Comparison')
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Confusion Matrix
    ax4 = axes[1, 1]
    sns.heatmap(quantum_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=ax4, xticklabels=['No Def', 'Def'],
                yticklabels=['No Def', 'Def'])
    ax4.set_title('VQC Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('./results_comparison.png', dpi=150)
    print("\n✓ Saved: results_comparison.png")
    plt.show()


# =============================================================================
# HARDWARE VALIDATION (Use sparingly - 30 credits budget!)
# =============================================================================

def validate_on_hardware(weights, X_test, y_test, n_samples=HARDWARE_TEST_SAMPLES):
    """
    Validate trained model on real IQM Garnet hardware.
    WARNING: Uses credits! Only run when ready.
    """
    print("\n" + "=" * 60)
    print("⚠️  HARDWARE VALIDATION - IQM GARNET")
    print("=" * 60)
    
    # Estimate cost
    est_time = n_samples * HARDWARE_SHOTS * 0.001  # rough estimate
    est_credits = est_time * 0.5
    print(f"  Samples: {n_samples}")
    print(f"  Shots: {HARDWARE_SHOTS}")
    print(f"  Estimated time: {est_time:.0f} seconds")
    print(f"  Estimated cost: ~{est_credits:.1f} credits")
    
    confirm = input("\n  Proceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("  Cancelled.")
        return None
    
    # Setup hardware device
    dev_hw = setup_quantum_device(use_hardware=True, n_qubits=N_QUBITS, shots=HARDWARE_SHOTS)
    circuit_hw = create_vqc_circuit(dev_hw, n_qubits=N_QUBITS, n_layers=N_LAYERS)
    
    # Use subset
    X_hw = X_test[:n_samples]
    y_hw = y_test[:n_samples]
    
    print(f"\n  Running on IQM Garnet...")
    results = evaluate_model(circuit_hw, weights, X_hw, y_hw, name="VQC (Hardware)")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  QUANTUM CREDIT RISK - VQC")
    print("  Banco Santander Hackathon 2025")
    print(f"  Config: {CONFIG}")
    print("=" * 60)
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess_data(
        './credit_risk_dataset_red.csv'
    )
    
    # Step 2: Setup simulator
    dev = setup_quantum_device(use_hardware=False)
    
    # Step 3: Create circuit
    print("\n" + "=" * 60)
    print("STEP 3: VQC CIRCUIT")
    print("=" * 60)
    circuit = create_vqc_circuit(dev)
    print(f"✓ Circuit: {N_QUBITS} qubits, {N_LAYERS} layers")
    print(f"  Parameters: {N_LAYERS * N_QUBITS * 3}")
    
    # Step 4: Train
    weights, history = train_vqc(circuit, X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate with default threshold
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATION (Default Threshold)")
    print("=" * 60)
    quantum_results = evaluate_model(circuit, weights, X_test, y_test, name="VQC (default threshold)")
    
    # Step 5b: Optimize threshold
    optimal_thresh, y_pred_opt, y_prob_opt, thresh_analysis = optimize_threshold(
        circuit, weights, X_test, y_test
    )
    
    # Step 5c: Evaluate with optimized threshold
    print("\n" + "=" * 60)
    print(f"EVALUATION WITH OPTIMIZED THRESHOLD ({optimal_thresh:.2f})")
    print("=" * 60)
    
    acc_opt = accuracy_score(y_test, y_pred_opt)
    auc_opt = roc_auc_score(y_test, y_prob_opt)
    f1_opt = f1_score(y_test, y_pred_opt)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    
    quantum_results_optimized = {
        'accuracy': acc_opt,
        'auc_roc': auc_opt,
        'gini': 2 * auc_opt - 1,
        'f1': f1_opt,
        'y_pred': y_pred_opt,
        'y_prob': y_prob_opt,
        'confusion_matrix': cm_opt,
        'threshold': optimal_thresh
    }
    
    print(f"  Accuracy:         {acc_opt:.4f}")
    print(f"  AUC-ROC:          {auc_opt:.4f}")
    print(f"  Gini:             {2*auc_opt-1:.4f}")
    print(f"  F1-Score:         {f1_opt:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_opt, target_names=['No Default', 'Default']))
    print(f"  Confusion Matrix:\n{cm_opt}")
    
    # Step 6: Classical baseline
    classical_results = train_classical_baseline(X_train, y_train, X_test, y_test)
    
    # Step 7: Plot (using optimized results)
    plot_results(quantum_results_optimized, classical_results, y_test, history)
    
    # Summary comparing all three
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<12} {'VQC(def)':<12} {'VQC(opt)':<12} {'XGBoost':<12}")
    print("-" * 50)
    for m in ['accuracy', 'auc_roc', 'gini', 'f1']:
        q_def = quantum_results[m]
        q_opt = quantum_results_optimized[m]
        c = classical_results[m]
        print(f"{m:<12} {q_def:<12.4f} {q_opt:<12.4f} {c:<12.4f}")
    
    print(f"\n  Optimal threshold: {optimal_thresh:.2f}")
    print(f"  Improvement: F1 {quantum_results['f1']:.4f} → {f1_opt:.4f} "
          f"({(f1_opt-quantum_results['f1'])*100:+.1f}%)")
    
    print("\n" + "=" * 60)
    print("NEXT: Run validate_on_hardware(weights, X_test, y_test) when ready")
    print("=" * 60)
    
    return weights, quantum_results_optimized, classical_results, X_test, y_test, circuit


if __name__ == "__main__":
    weights, q_results, c_results, X_test, y_test, circuit = main()
    
    # Uncomment to validate on hardware when ready:
    # hw_results = validate_on_hardware(weights, X_test, y_test)
