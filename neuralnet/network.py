import numpy as np
import time
from typing import Literal, List, Tuple

from .activations import build_activation_registry
from .losses import build_loss_registry

class NeuralNetwork:
	def __init__(
		self,
		layers: List[Tuple[int, Literal["relu", "leaky_relu", "elu", "tanh", "softsign", "swish", None]]],
		loss: Literal["mse", "mae", "cross_entropy"] = "mse",
		cuda: bool = True,
		lambda_l2: float = 1e-4
	):
		self.is_cupy = False
		if cuda:
			try:
				import cupy as xp
				self.is_cupy = True
			except ImportError:
				print("CuPy not found, falling back to NumPy")
				import numpy as xp
		else:
			import numpy as xp

		self.xp = xp
		self.layers = layers
		self.weights = []
		self.biases = []
		self.lambda_l2 = lambda_l2

		activs = build_activation_registry(self.xp)
		losses = build_loss_registry(self.xp)
		self.loss_fn, self.loss_grad = losses[loss]

		self.activation, self.deriv = [], []
		for layer in layers:
			_a, _d = activs[layer[1]]
			self.activation.append(_a)
			self.deriv.append(_d)

		self.xp.random.seed(1)
		for i in range(len(layers) - 1):
			self.weights.append(
				self.xp.random.randn(layers[i][0], layers[i + 1][0]) * self.xp.sqrt(2 / layers[i][0])
			)
			self.biases.append(self.xp.zeros((1, layers[i + 1][0])))

	def forward(self, X):
		if not isinstance(X, self.xp.ndarray):
			X = self.xp.array(X)

		self.activations = [X]
		self.zs = []
		a = X

		for i in range(len(self.weights) - 1):
			z = a @ self.weights[i] + self.biases[i]
			a = self.activation[i](z)
			self.zs.append(z)
			self.activations.append(a)

		z = a @ self.weights[-1] + self.biases[-1]
		a = self.activation[-1](z)
		self.zs.append(z)
		self.activations.append(a)

		return a

	def backward(self, y, norm_threshold):
		dws = [None] * len(self.weights)
		dbs = [None] * len(self.biases)

		d = self.loss_grad(self.activations[-1], y)
		largest_norm = -1

		for i in reversed(range(len(self.weights))):
			dws[i] = self.activations[i].T.dot(d)
			dws[i] += 2 * self.lambda_l2 * self.weights[i] # l2
			dbs[i] = self.xp.sum(d, axis=0, keepdims=True)

			if i > 0:
				d = d.dot(self.weights[i].T) * self.deriv[i](self.zs[i - 1])
				norm = self.xp.sqrt(self.xp.sum(d * d))
				if norm > largest_norm:
					largest_norm = norm
				if norm > norm_threshold:
					d = d * (norm_threshold / norm)

		return dws, dbs, largest_norm

	def l2_loss(self):
		l2_sum = 0
		for w in self.weights:
			l2_sum += self.xp.sum(w ** 2)
		return l2_sum

	def train(self, X, y, epochs: int = 10000, lr: float = 0.01, train_split: float = 0.8, batch_size: int = 32, norm_threshold: float = 5.0, save_filename: str = "best"):
		if not isinstance(X, self.xp.ndarray):
			X = self.xp.array(X)
		if not isinstance(y, self.xp.ndarray):
			y = self.xp.array(y)

		best_weights = [w.copy() for w in self.weights]
		best_biases = [b.copy() for b in self.biases]
		best_loss = None

		split = int(train_split * len(X))
		X_train, X_val = X[:split], X[split:]
		y_train, y_val = y[:split], y[split:]

		n = X_train.shape[0]

		print("")
		start_time = time.time()

		for epoch in range(epochs):
			indices = self.xp.random.permutation(n)
			X_shuffled = X_train[indices]
			y_shuffled = y_train[indices]

			epoch_loss = 0
			max_norm = 0.0
			total_samples = 0

			for i in range(0, n, batch_size):
				X_batch = X_shuffled[i:i+batch_size]
				y_batch = y_shuffled[i:i+batch_size]
				current_batch_size = X_batch.shape[0]

				out = self.forward(X_batch)
				epoch_loss += self.loss_fn(y_batch, out) * current_batch_size
				
				dws, dbs, norm = self.backward(y_batch, norm_threshold)
				max_norm = max(max_norm, float(norm))
				
				total_samples += current_batch_size

				for j in range(len(self.weights)):
					self.weights[j] -= lr * dws[j]
					self.biases[j] -= lr * dbs[j]
			
			epoch_loss = (epoch_loss + self.l2_loss() * self.lambda_l2) / total_samples

			if best_loss is None or epoch_loss < best_loss:
				best_loss = epoch_loss
				best_weights = [w.copy() for w in self.weights]
				best_biases = [b.copy() for b in self.biases]

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
				f"Loss: {float(epoch_loss):.6f}\tBest: {float(best_loss):.6f}\t"
				f"Norm: {float(max_norm):.4f}\tETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}",
				flush=True
			)

			if epoch_loss <= 1e-6:
				break

		self.save_weights(save_filename, best_weights, best_biases)
		vals = self.forward(X_val)
		val_loss = self.loss_fn(vals, y_val)
		print(f"Completed training with a training loss of {float(best_loss):.6f} and validation loss of {float(val_loss):.6f}")

	def save_weights(self, filename, weights, biases):
		w_obj = np.empty(len(weights), dtype=object)
		b_obj = np.empty(len(biases), dtype=object)

		for i in range(len(weights)):
			w_obj[i] = getattr(weights[i], "get", lambda: weights[i])()
			b_obj[i] = getattr(biases[i], "get", lambda: biases[i])()

		np.savez(filename, weights=w_obj, biases=b_obj)

	def load_weights(self, filename):
		data = np.load(filename, allow_pickle=True)
		self.weights = [self.xp.array(w) for w in data["weights"]]
		self.biases = [self.xp.array(b) for b in data["biases"]]