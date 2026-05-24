import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

from qiskit.primitives import StatevectorSampler

# ----------------
# Load
# ----------------

df = pd.read_csv(
    "datasets/final_dataset.csv"
)

# Reduce for quantum
df = df.sample(
    1500,
    random_state=42
)

if "attack_cat" in df.columns:
    df = df.drop(
        columns=["attack_cat"]
    )

X = df.drop(
    columns=["label"]
)

y = df["label"]

# Quantum input reduction
X = X.iloc[:, :4]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Convert to numpy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# ----------------
# Quantum Model
# ----------------

num_features = 4

feature_map = ZZFeatureMap(
    feature_dimension=num_features,
    reps=3
)

ansatz = RealAmplitudes(
    num_features,
    reps=3
)

sampler = StatevectorSampler()

model = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(
        maxiter=40
    )
)

print("Training Quantum Model...")

model.fit(
    X_train,
    y_train
)

score = model.score(
    X_test,
    y_test
)

print(
    "\nQuantum Accuracy:",
    score
)