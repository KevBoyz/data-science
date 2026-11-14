import numpy as np

np.random.seed(123)


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def tanh(u):
    return np.tanh(u)


def relu(u):
    return np.maximum(0, u)


def leaky_relu(u, alpha=0.01):
    return np.where(u > 0, u, alpha * u)


def mean_squared_error(output: np.array, desirable: np.array):
    return np.mean((output - desirable) ** 2)


def hard_derivate(x, func, appr=4, eps=1e-8):
    right = lambda eps: (func(x + eps) - func(x)) / eps
    left = lambda eps: (func(x) - func(x - eps)) / eps
    aproximations = []
    for i in range(appr):
        alpha = (i % 3) + 1  # 1, 2, ..., appr
        aproximations.append(right(alpha * eps))
        aproximations.append(left(alpha * eps))
    return sum(aproximations) / len(aproximations)
    

class MLP:
    def __init__(self, layers: list, loss: callable):
        self.layers = layers
        self.loss = loss

    def foward(self, input: np.array):
        foward_results = []
        for layer in self.layers:
            result = layer.foward(input)
            foward_results.append(result)
            input = result['activation']
        return foward_results

    def calulate_loss(self, output, desireble):
        return self.loss(output, desireble)

    def calculate_deltas(self):
        deltas = []
        exit_delta = hard_derivate(
            self.layers[-1].weights,
            self.layers[-1].activation
            ) * hard_derivate(
            self.layers
            )
        for layer in self.layers[0:]:
            ...


class Layer:
    def __init__(self, activation:callable, features_in:int=None, features_out:int=None,
                 weights:np.array=None, bias:np.array=None):
        self.features_in = features_in
        self.features_out = features_out
        self.weights = weights
        self.bias = bias
        if not self.weights.size:
            if not self.features_in or not self.features_out:
                raise Exception('you may pass features_in and features_in')
            self.weights = np.random.randn(features_out, features_in)
        if not self.bias.size:
            if not self.features_in or not self.features_out:
                raise Exception('you may pass features_in and features_in')
            self.bias = np.random.randn(features_out, 1)
        
        self.activation = activation

    def foward(self, input):
        potencial = self.weights @ input + self.bias
        activation = self.activation(potencial)
        return {'potencial':potencial, 'activation':activation}


# features x amostras
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

W1 = np.array([[0.2, -0.4],
               [0.7,  0.1]])
b1 = np.array([[ 0.1],
               [-0.3]])

W2 = np.array([[0.5, -0.6]])
b2 = np.array([[0.2]])

mlp = MLP(
    [
        Layer(activation=sigmoid, features_in=2, features_out=2, weights=W1, bias=b1),
        Layer(activation=sigmoid, features_in=2, features_out=1, weights=W2, bias=b2),
    ],
    loss=mean_squared_error
)

print(mlp.calulate_loss(mlp.foward(X)[-1]['activation'], Y))