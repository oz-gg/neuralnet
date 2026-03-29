def relu(xp):
    return lambda x: xp.maximum(0, x)

def leaky_relu(xp):
    return lambda x: xp.where(x > 0, x, 0.01 * x)

def elu(xp):
    return lambda x: xp.where(x > 0, x, xp.exp(x) - 1)

def tanh(xp):
    return lambda x: xp.tanh(x)

def softsign(xp):
    return lambda x: x / (1 + xp.abs(x))

def swish(xp):
    return lambda x: x / (1 + xp.exp(-x))


def relu_deriv(xp):
    return lambda x: (x > 0).astype(xp.float32)

def leaky_relu_deriv(xp):
    return lambda x: xp.where(x > 0, 1, 0.01)

def elu_deriv(xp):
    return lambda x: xp.where(x > 0, 1, xp.exp(x))

def tanh_deriv(xp):
    return lambda x: 1 - xp.tanh(x) ** 2

def softsign_deriv(xp):
    return lambda x: 1 / (1 + xp.abs(x)) ** 2

def swish_deriv(xp):
    return lambda x: (lambda sig: sig + x * sig * (1 - sig))(1 / (1 + xp.exp(-x)))


def build_activation_registry(xp):
    return {
        "relu": (relu(xp), relu_deriv(xp)),
        "leaky_relu": (leaky_relu(xp), leaky_relu_deriv(xp)),
        "elu": (elu(xp), elu_deriv(xp)),
        "tanh": (tanh(xp), tanh_deriv(xp)),
        "softsign": (softsign(xp), softsign_deriv(xp)),
        "swish": (swish(xp), swish_deriv(xp)),
    }