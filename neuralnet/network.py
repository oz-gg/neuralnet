import numpy as np
import time
from typing import Literal

from .activations import build_activation_registry


class NeuralNetwork:
    def __init__(
        self,
        layers: list[int],
        cuda: bool = True,
        activation: Literal["relu", "leaky_relu", "elu", "tanh", "softsign", "swish"] = "relu"
    ):
        if cuda:
            try:
                import cupy as xp
            except ImportError:
                print("CuPy not found, falling back to NumPy")
                import numpy as xp
        else:
            import numpy as xp

        self.xp = xp
        self.layers = layers
        self.weights = []
        self.biases = []

        activs = build_activation_registry(self.xp)
        if activation not in activs:
            raise ValueError(f"Invalid activation: {activation}")

        self.activation, self.deriv = activs[activation]

        self.xp.random.seed(1)
        for i in range(len(layers) - 1):
            self.weights.append(
                self.xp.random.randn(layers[i], layers[i + 1]) * self.xp.sqrt(2 / layers[i])
            )
            self.biases.append(self.xp.zeros((1, layers[i + 1])))

    def forward(self, X):
        if not isinstance(X, self.xp.ndarray):
            X = self.xp.array(X)

        self.activations = [X]
        self.zs = []
        a = X

        for i in range(len(self.weights) - 1):
            z = a.dot(self.weights[i]) + self.biases[i]
            a = self.activation(z)
            self.zs.append(z)
            self.activations.append(a)

        z = a.dot(self.weights[-1]) + self.biases[-1]
        a = z
        self.zs.append(z)
        self.activations.append(a)

        return a

    def backward(self, y, norm_threshold):
        dws = [None] * len(self.weights)
        dbs = [None] * len(self.biases)

        d = self.activations[-1] - y
        largest_norm = -1

        for i in reversed(range(len(self.weights))):
            dws[i] = self.activations[i].T.dot(d)
            dbs[i] = self.xp.sum(d, axis=0, keepdims=True)

            if i > 0:
                d = d.dot(self.weights[i].T) * self.deriv(self.zs[i - 1])
                norm = self.xp.sqrt(self.xp.sum(d * d))
                if norm > largest_norm:
                    largest_norm = norm
                if norm > norm_threshold:
                    d = d * (norm_threshold / norm)

        return dws, dbs, largest_norm

    def train(self, X, y, epochs: int = 10000, lr: float = 0.01, norm_threshold: float = 5.0, save_filename: str = "best"):
        if not isinstance(X, self.xp.ndarray):
            X = self.xp.array(X)
        if not isinstance(y, self.xp.ndarray):
            y = self.xp.array(y)

        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]
        best_loss = None

        print("")
        start_time = time.time()

        for epoch in range(epochs):
            out = self.forward(X)
            loss = self.xp.mean((y - out) ** 2)

            if best_loss is None:
                best_loss = loss

            if loss < best_loss:
                best_loss = loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]

            dws, dbs, norm = self.backward(y, norm_threshold)

            for i in range(len(self.weights)):
                self.weights[i] -= lr * dws[i]
                self.biases[i] -= lr * dbs[i]

            progress = epoch / epochs
            filled = int(progress * 50)
            bar = "#" * filled + " " * (50 - filled)
            percent = int(100 * progress)

            elapsed = time.time() - start_time
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            eta_h = int(eta // 3600)
            eta_m = int((eta % 3600) // 60)
            eta_s = int(eta % 60)

            print(
                f"\033[F\033[K[{bar}] {percent}% ({epoch}/{epochs})\t"
                f"Loss: {float(loss):.6f}\tBest: {float(best_loss):.6f}\t"
                f"Norm: {float(norm):.4f}\tETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}",
                flush=True
            )

            if loss <= 1e-5:
                break

        self.save_weights(save_filename, best_weights, best_biases)
        print(f"Completed training with a best loss of {float(best_loss):.6f}")

    def save_weights(self, filename, weights, biases):
        w_obj = np.empty(len(weights), dtype=object)
        b_obj = np.empty(len(biases), dtype=object)

        is_cupy = self.xp.__name__ == "cupy"
        for i in range(len(weights)):
            w_obj[i] = self.xp.asnumpy(weights[i]) if is_cupy else weights[i]
            b_obj[i] = self.xp.asnumpy(biases[i]) if is_cupy else biases[i]

        np.savez(filename, weights=w_obj, biases=b_obj)

    def load_weights(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.weights = [self.xp.array(w) for w in data["weights"]]
        self.biases = [self.xp.array(b) for b in data["biases"]]