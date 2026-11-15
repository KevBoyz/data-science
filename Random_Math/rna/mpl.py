import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)


# Funções de ativação

def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def tanh(u):
    return np.tanh(u)


def relu(u):
    return np.maximum(0, u)


def leaky_relu(u, alpha=0.01):
    return np.where(u > 0, u, alpha * u)


# Funções de perda

def quadratic_error(output: np.array, desirable: np.array, n:int=1, report:bool=False):
    if report:
        loss = 0
        for o, d in zip(output.flatten(), desirable.flatten()):
            loss += ((o - d) ** 2) / n
        return loss / output.shape[1]
    else:
        return ((output - desirable) ** 2) / n
 

def hard_derivate(func, x, *args, appr=4, eps=1e-8, **kwargs):
    f = lambda out: func(out, *args, **kwargs)
    right = lambda eps: (f(x + eps) - f(x)) / eps
    left = lambda eps: (f(x) - f(x - eps)) / eps
    aproximations = []
    alphas = list(range(1, appr + 1))
    for a in alphas:
        aproximations.append(right(a * eps))
        aproximations.append(left(a * eps))
    return sum(aproximations) / len(aproximations)


class MLP:  # BGD
    def __init__(self, layers: list, loss: callable, X: np.array, Y: np.array, eta=0.1):
        self.layers = layers
        self.loss = loss
        self.X = X  # features x amostras
        self.Y = Y  # saídas x amostras
        self.eta = eta

    def predict(self, input: np.array = None):
        # sem input usa self.X; passar input só para inferir em dados novos
        input = self.X if input is None else input
        foward_results = []
        for layer in self.layers:
            result = layer.foward(input)
            foward_results.append(result)
            input = result['activations']
        return foward_results

    def get_out(self, input: np.array = None):
        pred = self.predict(input)
        return pred[-1]['activations']

    def report(self):
        return self.loss(self.get_out(), self.Y, report=True)

    def __calculate_deltas(self, foward_results):
        # dimensões da matriz δi(output,input)
        # ou seja δi(neuronio, amostra)
        deltas = []
        exit_delta = hard_derivate(
            self.loss,
            foward_results[-1]['activations'],
            desirable=self.Y
        ) * hard_derivate(
                self.layers[-1].activation,
                foward_results[-1]['potencials'],
            )
        deltas.append(exit_delta)
        for i, layer in enumerate(reversed(self.layers[:-1])):
            back_layer_i = -(i+1)
            actual_layer_i = back_layer_i - 1
            deltas.append(
                (self.layers[back_layer_i].weights.T @ deltas[-1]) *
                hard_derivate(layer.activation, foward_results[actual_layer_i]['potencials'])
            )
        return list(reversed(deltas))

    def __calculate_gradients(self, foward_results, deltas):
        weight_gradients = []
        weight_gradients.append(deltas[0] @ self.X.T)

        for i in range(1, len(self.layers)):
            weight_gradients.append( # δ @ yᵀ = Σₙ δₙ · yₙᵀ
                (deltas[i] @ foward_results[i-1]['activations'].T)
                / self.X.shape[1]
            )
        bias_gradients = [np.sum(d, axis=1, keepdims=True) for d in deltas]
        return (weight_gradients, bias_gradients)

    def __get_gradients(self):
        foward_results = self.predict()
        deltas = self.__calculate_deltas(foward_results)
        gradients = self.__calculate_gradients(foward_results, deltas)
        return gradients

    def update_layers(self):
        gradients = self.__get_gradients()
        wg = gradients[0]
        bg = gradients[1]
        for i, layer in enumerate(self.layers):
            layer.weights -= self.eta * wg[i]
            layer.bias -= self.eta * bg[i]


class Layer:
    def __init__(self, activation: callable, shape:list=None,
                 weights: np.array = None, bias: np.array = None):
        if shape:
            self.features_in = shape[0]
            self.features_out = shape[1]
        self.weights = weights
        self.bias = bias
        if not self.weights.size:
            if not self.features_in or not self.features_out:
                raise Exception('you may pass features_in and features_in')
            self.weights = np.random.randn(self.features_out, self.features_in)
        if not self.bias.size:
            if not self.features_in or not self.features_out:
                raise Exception('you may pass features_in and features_in')
            self.bias = np.random.randn(self.features_out, 1)

        self.activation = activation

    def foward(self, input):
        potencial = self.weights @ input + self.bias
        activation = self.activation(potencial)
        return {'potencials': potencial, 'activations': activation}


X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

Y = np.array([[0, 1, 1, 0]])

W1 = np.array([[0.9, -0.9],
               [0.7,  0.8]])
b1 = np.array([[0.1],
               [-0.4]])

W2 = np.array([[0.5, 0.9]])
b2 = np.array([[0.2]])

model = MLP(
    [
        Layer(activation=sigmoid,
                weights=W1, bias=b1),
        Layer(activation=sigmoid,
                weights=W2, bias=b2),
    ],
    loss=quadratic_error,
    X=X,
    Y=Y,
    eta=0.1
)

losses = []
dif = float("inf")

while dif >= 1e-5:
    loss = model.report()
    losses.append(loss)

    model.update_layers()

    if len(losses) > 1:
        dif = abs(losses[-2] - losses[-1])

plt.plot(range(len(losses)), losses)
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.show()