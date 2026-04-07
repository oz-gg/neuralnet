def mse(xp):
	def loss(y_pred, y_true):
		return xp.mean((y_pred - y_true) ** 2)
	
	def grad(y_pred, y_true):
		return 2 * (y_pred - y_true) / y_true.shape[0]
	
	return loss, grad

def mae(xp):
	def loss(y_pred, y_true):
		return xp.mean(xp.abs(y_pred - y_true))
	
	def grad(y_pred, y_true):
		return xp.sign(y_pred - y_true) / y_true.shape[0]
	
	return loss, grad

def cross_entropy(xp):
	def loss(y_pred, y_true):
		eps = 1e-12
		y_pred = xp.clip(y_pred, eps, 1 - eps)
		return -xp.mean(xp.sum(y_true * xp.log(y_pred), axis=1))
	
	def grad(y_pred, y_true):
		eps = 1e-12
		y_pred = xp.clip(y_pred, eps, eps - 1)
		return (y_pred - y_true) / y_true.shape[0]
	
	return loss, grad

def build_loss_registry(xp):
	return {
		"mse": mse(xp),
		"mae": mae(xp),
		"cross_entropy": cross_entropy(xp),
		None: mse(xp)
	}