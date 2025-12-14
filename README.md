# 🏆 Quantum Credit Risk Prediction

**Banco Santander Quantum Hackathon 2025 - Madrid**

---

## 📋 Table of Contents

1. [The Challenge](#-the-challenge)
2. [Data Pipeline](#-data-pipeline)
3. [Approach 1: VQC](#-approach-1-vqc-variational-quantum-classifier)
4. [Approach 2: QSVC](#-approach-2-qsvc-quantum-support-vector-classifier)
5. [Why QSVC Excels in Small-Data Regime](#-why-qsvc-excels-in-small-data-regime)
6. [Quick Start](#-quick-start)
7. [Project Files](#-project-files)

---

## 🎯 The Challenge

| Aspect | Description |
|--------|-------------|
| **Task** | Binary classification - predict loan default (Yes/No) |
| **Dataset** | Credit risk data with financial and personal features |
| **Goal** | Leverage quantum computing for credit risk assessment |

---

## 📈 Data Pipeline

Both quantum approaches share the same preprocessing pipeline:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│   Encode     │────▶│     PCA      │────▶│  Normalize   │
│  12 features │     │ Categoricals │     │   11 → 5     │     │   [0, π]     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Step 1: Handle Missing Values & Outliers

```python
df['person_emp_length'].fillna(df['person_emp_length'].mode()[0])
df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df = df[df['person_age'] <= 100]
```

### Step 2: Encode Categorical Variables

| Variable | Encoding |
|----------|----------|
| `person_home_ownership` | LabelEncoder |
| `loan_grade` | Ordinal (A=0...G=6) |
| `loan_intent` | LabelEncoder |
| `cb_person_default_on_file` | Binary (N=0, Y=1) |

### Step 3: PCA Reduction

```python
# 11 features → 5 principal components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
```

**Why PCA?**
- Reduces dimensionality to match qubit count
- Captures maximum variance in fewer components
- Eliminates feature correlations

### Step 4: Normalize to [0, π]

```python
X_quantum = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_pca)
```

**Why [0, π]?** Quantum rotation gates RY(θ) have period 2π. Scaling to [0, π] ensures full rotation range without redundancy.

---

## ⚛️ Approach 1: VQC (Variational Quantum Classifier)

### Circuit Architecture

```
       ENCODING              VARIATIONAL LAYER           MEASUREMENT
          │                         │                         │
q0: ──[RY(x₀)]────[RY(θ₀)]──[RZ(θ₁)]──●────[RY(θ₂)]─────────< Z >
                                      │
q1: ──[RY(x₁)]────[RY(θ₃)]──[RZ(θ₄)]──Z──●──[RY(θ₅)]──────────
                                         │
q2: ──[RY(x₂)]────[RY(θ₆)]──[RZ(θ₇)]─────Z──●──[RY(θ₈)]───────
                                            │
q3: ──[RY(x₃)]────[RY(θ₉)]──[RZ(θ₁₀)]───────Z──●──[RY(θ₁₁)]───
                                               │
q4: ──[RY(x₄)]────[RY(θ₁₂)]──[RZ(θ₁₃)]─────────Z──[RY(θ₁₄)]───
```

### How It Works

1. **Encoding**: Each PCA component → RY rotation on corresponding qubit
2. **Variational Layer**: Trainable RY/RZ rotations + CZ entanglement
3. **Measurement**: ⟨Z⟩ expectation value on qubit 0
4. **Training**: Optimize θ parameters via gradient descent (Parameter Shift Rule)
5. **Decision**: ⟨Z⟩ > threshold → No Default, else → Default

---

## 🔬 Approach 2: QSVC (Quantum Support Vector Classifier)

### The Core Idea: Quantum Kernel

QSVC uses the quantum circuit not to classify directly, but to **measure similarity** between data points. This similarity function (kernel) is then used by a classical SVM.

### Mathematical Foundation

The quantum kernel is based on **state fidelity**:

$$K(x, y) = |\langle\phi(x)|\phi(y)\rangle|^2$$

Where:
- $|φ(x)⟩ = U(x)|0⟩$ is the quantum state encoding data point x
- $U(x)$ is the feature map circuit
- $K(x,y)$ measures how similar two quantum states are

**Key Property**: If $x ≈ y$, then $U(x) ≈ U(y)$, so $U†(y)U(x) ≈ I$, thus $K(x,y) ≈ 1$

### Computing the Kernel: Circuit Design

To compute $K(x,y)$, we apply $U(x)$ followed by $U†(y)$ and measure:

$$K(x, y) = |\langle 0|U^\dagger(y) \cdot U(x)|0\rangle|^2 = P(|00000\rangle)$$

```
         U(x) - Feature Map              U†(y) - Adjoint              MEASURE
              │                               │                          │
              │                               │                          │
q0: ──[RY(x₀)]──[RZ(x₀)]──●──────────[RZ(-y₀)]──[RY(-y₀)]────────────── P(|0⟩)
                          │
q1: ──[RY(x₁)]──[RZ(x₁)]──X──●───────[RZ(-y₁)]──[RY(-y₁)]────────────── P(|0⟩)
                             │
q2: ──[RY(x₂)]──[RZ(x₂)]─────X──●────[RZ(-y₂)]──[RY(-y₂)]────────────── P(|0⟩)
                                │
q3: ──[RY(x₃)]──[RZ(x₃)]────────X──●─[RZ(-y₃)]──[RY(-y₃)]────────────── P(|0⟩)
                                   │
q4: ──[RY(x₄)]──[RZ(x₄)]───────────X─[RZ(-y₄)]──[RY(-y₄)]────────────── P(|0⟩)
```

### Step-by-Step: Encoding to Kernel Value

#### Step 1: Feature Map U(x)

For each data point x = [x₀, x₁, x₂, x₃, x₄]:

```python
def feature_map(x):
    for layer in range(N_LAYERS):
        # Rotation layer - encode features
        for i in range(N_QUBITS):
            qml.RY(x[i], wires=i)           # Amplitude encoding
            qml.RZ(x[i] * scale, wires=i)   # Phase encoding
        
        # Entanglement layer - create correlations
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])   # Circular connection
```

This transforms $|00000⟩$ → $|φ(x)⟩$

#### Step 2: Adjoint Feature Map U†(y)

Apply the **inverse** operations in **reverse order**:

```python
def adjoint_feature_map(y):
    for layer in reversed(range(N_LAYERS)):
        # Reverse entanglement
        qml.CNOT(wires=[N_QUBITS - 1, 0])
        for i in reversed(range(N_QUBITS - 1)):
            qml.CNOT(wires=[i, i + 1])
        
        # Reverse rotations (negative angles)
        for i in range(N_QUBITS):
            qml.RZ(-y[i] * scale, wires=i)
            qml.RY(-y[i], wires=i)
```

#### Step 3: Measure Kernel Value

```python
@qml.qnode(dev)
def kernel_circuit(x, y):
    feature_map(x)           # Apply U(x)
    adjoint_feature_map(y)   # Apply U†(y)
    return qml.probs(wires=range(N_QUBITS))

def kernel_value(x, y):
    probs = kernel_circuit(x, y)
    return probs[0]  # P(|00000⟩) = K(x, y)
```

### Building the Kernel Matrix

For training, we need the similarity between **all pairs** of training points:

```
              x₁    x₂    x₃   ...   xₙ
           ┌─────┬─────┬─────┬─────┬─────┐
      x₁   │ 1.0 │ 0.8 │ 0.3 │ ... │ 0.5 │
           ├─────┼─────┼─────┼─────┼─────┤
      x₂   │ 0.8 │ 1.0 │ 0.4 │ ... │ 0.6 │
           ├─────┼─────┼─────┼─────┼─────┤
K_train =  x₃   │ 0.3 │ 0.4 │ 1.0 │ ... │ 0.2 │
           ├─────┼─────┼─────┼─────┼─────┤
      ...  │ ... │ ... │ ... │ ... │ ... │
           ├─────┼─────┼─────┼─────┼─────┤
      xₙ   │ 0.5 │ 0.6 │ 0.2 │ ... │ 1.0 │
           └─────┴─────┴─────┴─────┴─────┘
```

**Properties:**
- Diagonal ≈ 1.0 (each point is identical to itself)
- Symmetric: K(x,y) = K(y,x)
- Off-diagonal: similarity between different points

```python
def compute_kernel_matrix(X1, X2, symmetric=False):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(i if symmetric else 0, n2):
            K[i, j] = kernel_value(X1[i], X2[j])
            if symmetric:
                K[j, i] = K[i, j]
    return K
```

### Training the QSVC

Once we have the kernel matrix, training is **classical**:

```python
from sklearn.svm import SVC

# 1. Compute quantum kernel matrices
K_train = compute_kernel_matrix(X_train, X_train, symmetric=True)
K_test = compute_kernel_matrix(X_test, X_train, symmetric=False)

# 2. Normalize kernel (optional but recommended)
d = np.sqrt(np.diag(K_train))
K_train_norm = K_train / np.outer(d, d)
K_test_norm = K_test / d

# 3. Train SVM with precomputed kernel
svm = SVC(kernel='precomputed', C=best_C, class_weight='balanced')
svm.fit(K_train_norm, y_train)

# 4. Predict
y_pred = svm.predict(K_test_norm)
```

### Hyperparameter Tuning

The SVM parameter **C** controls regularization. We tune it **without recomputing the kernel**:

```python
def tune_C(K_train, y_train):
    best_auc, best_C = 0, 1.0
    for C in [0.1, 1.0, 10.0, 100.0]:
        svm = SVC(kernel='precomputed', C=C, probability=True)
        svm.fit(K_train, y_train)
        # Cross-validate or use validation set
        if auc > best_auc:
            best_auc, best_C = auc, C
    return best_C
```

### Inference: New Clients

When a new client applies for a loan:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NEW CLIENT INFERENCE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

  NEW CLIENT DATA
  ───────────────
  │ age: 28, income: 45000, home: "RENT", ...
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PREPROCESS (same pipeline as training)                                   │
│                                                                                  │
│   Raw → Encode → StandardScaler → PCA → MinMaxScaler[0,π]                       │
│                                                                                  │
│   x_new = [0.82, 1.45, 2.31, 0.67, 1.89]  (5 values in [0, π])                  │
└─────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: COMPUTE KERNEL VECTOR                                                    │
│                                                                                  │
│   For each training point xᵢ:                                                    │
│     k_new[i] = kernel_circuit(x_new, xᵢ)  →  P(|00000⟩)                         │
│                                                                                  │
│   k_new = [0.45, 0.72, 0.31, 0.88, ...]  (n_train values)                       │
└─────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SVM PREDICTION                                                           │
│                                                                                  │
│   prediction = svm.predict(k_new)                                                │
│   probability = svm.predict_proba(k_new)                                         │
│                                                                                  │
│   Output: class=0 (No Default), P(Default)=0.23                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: DECISION                                                                 │
│                                                                                  │
│   if P(Default) < threshold:                                                     │
│       APPROVE LOAN                                                               │
│   else:                                                                          │
│       REJECT LOAN (or require further review)                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Why QSVC Excels in Small-Data Regime

### 1. Kernel Methods Are Data-Efficient

Unlike neural networks that need massive datasets, **kernel methods** (including SVM) are designed to work well with limited data. They find decision boundaries using only the most informative points (support vectors).

### 2. Exponentially Large Feature Space

With 5 qubits, the quantum kernel implicitly operates in a **2⁵ = 32 dimensional** Hilbert space. This rich representation can capture complex patterns that would require many more features classically.

```
Original space:    5 dimensions (PCA components)
Quantum space:    32 dimensions (quantum amplitudes)
```

### 3. Entanglement Captures Feature Interactions

The CNOT gates create **quantum correlations** between qubits. This means the kernel naturally captures interactions between features (e.g., income × loan amount) that classical kernels like RBF cannot efficiently represent.

### 4. No Parameters to Overfit

Unlike VQC which has trainable parameters (θ), **QSVC has no quantum parameters to train**. The feature map is fixed. This eliminates the risk of overfitting the quantum circuit to the training data.

### 5. Implicit Regularization

The quantum kernel provides a form of **implicit regularization**. The structure of the quantum circuit constrains which similarity functions are possible, preventing the model from fitting noise.

### 6. Theoretical Foundations

Research has shown that quantum kernels can provide **provable advantages** for certain data distributions. Credit risk data, with its complex feature interactions and class imbalance, appears to benefit from the quantum kernel's properties.

---

## 🚀 Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate iqm
```

### Run VQC

```bash
python quantum_credit_risk_vqc.py
```

### Run QSVC

```bash
python quantum_credit_risk_qsvc.py
```

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `quantum_credit_risk_vqc.py` | VQC implementation |
| `quantum_credit_risk_qsvc.py` | QSVC implementation |
| `environment.yml` | Conda environment |

---

*Developed for Banco Santander Quantum Hackathon 2025 - Madrid*
