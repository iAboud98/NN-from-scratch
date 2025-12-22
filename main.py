import os
import pandas as pd
import numpy as np

from data.dataset_loader import Dataset_Loader
from nn.layer import Layer
from nn.loss import BackPropagation

np.random.seed(0)

if not os.path.exists("data/digits_dataset.csv"):
    Dataset_Loader("data/digits")

df = pd.read_csv("data/digits_dataset.csv")

y = df.iloc[:, 0].to_numpy().astype(int)
X = df.iloc[:, 1:].to_numpy().astype(np.float32)


# ---------------------------
# Train/Val split (80/20)
# ---------------------------
N = X.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

split = int(0.8 * N)
train_idx = indices[:split]
val_idx   = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]


Dense_layer1 = Layer(neurons_num=128, features_num=784)
Dense_layer2 = Layer(neurons_num=10,  features_num=128)

# ---------------------------
# Training hyperparameters
# ---------------------------
epochs = 40
batch_size = 64
learning_rate = 0.10

# ---------------------------
# Helper: accuracy
# ---------------------------
def accuracy(probs, y_true):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(1, epochs + 1):

    # shuffle training data each epoch
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    # mini-batches
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    epoch_loss = 0.0
    epoch_acc  = 0.0

    for b in range(num_batches):
        start = b * batch_size
        end   = min((b + 1) * batch_size, X_train.shape[0])

        Xb = X_train[start:end]
        yb = y_train[start:end]

        # Forward
        Dense_layer1.forward(Xb)
        Dense_layer1.run_activation()

        Dense_layer2.forward(Dense_layer1.output)
        Dense_layer2.run_softmax()

        # Loss
        bp = BackPropagation(Dense_layer2.output, yb)
        encoded = bp.one_hot(10)
        loss = bp.cross_entropy(encoded)

        # Acc
        acc = accuracy(Dense_layer2.output, yb)

        epoch_loss += loss
        epoch_acc  += acc

        # Backward (start from dZ2)
        dZ2 = bp.d_softmax_cross_entropy()
        Dense_layer2.backward(dZ2)

        dA1 = Dense_layer2.dinputs
        dZ1 = Dense_layer1.backward_activation(dA1)
        Dense_layer1.backward(dZ1)

        Dense_layer1.apply_gradients(learning_rate)
        Dense_layer2.apply_gradients(learning_rate)

    # epoch averages
    epoch_loss /= num_batches
    epoch_acc  /= num_batches

    # ---------------------------
    # Validation
    # ---------------------------
    Dense_layer1.forward(X_val)
    Dense_layer1.run_activation()

    Dense_layer2.forward(Dense_layer1.output)
    Dense_layer2.run_softmax()

    val_loss_bp = BackPropagation(Dense_layer2.output, y_val)
    val_encoded = val_loss_bp.one_hot(10)
    val_loss = val_loss_bp.cross_entropy(val_encoded)
    val_acc  = accuracy(Dense_layer2.output, y_val)

    print(f"Epoch {epoch:02d} | train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")