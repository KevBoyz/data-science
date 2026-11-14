import numpy as np
import pandas as pd

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

def quadratic_error(output: np.array, desirable: np.array):
    return (output - desirable) ** 2


# Otimizadores

def batch_gd():
    ...




def hard_derivate(func, x, appr=4, eps=1e-8):
    def right(eps): return (func(x + eps) - func(x)) / eps
    def left(eps): return (func(x) - func(x - eps)) / eps
    aproximations = []
    alphas = list(range(1, appr + 1))
    for a in alphas:
        aproximations.append(right(a * eps))
        aproximations.append(left(a * eps))
    return sum(aproximations) / len(aproximations)


class MLP:
    def __init__(self, layers: list, loss: callable, eta=0.1):
        self.layers = layers
        self.loss = loss

    def foward(self, input: np.array):
        foward_results = []
        for layer in self.layers:
            result = layer.foward(input)
            foward_results.append(result)
            input = result['activations']
        return foward_results

    def calulate_loss(self, output, desireble):
        return self.loss(output, desireble)


    def calculate_deltas(self, foward_results, desirable):
        deltas = []
        # fixar desirable e chamar passando somente x
        loss_fixed = lambda out: self.loss(out, desirable)
        exit_delta = hard_derivate(
            loss_fixed,
            foward_results[-1]['activations']
        ) * hard_derivate(
                self.layers[-1].activation,
                foward_results[-1]['potencials']
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

    def calculate_gradients(self, input, foward_results, deltas):
        gradients = []
        gradients.append(
            deltas[0] @ input.T
        )
        for i, layer in enumerate(self.layers[0:]):
            try:  # Pular o primeiro delta já usado
                gradients.append(
                    deltas[i+1] @ foward_results[i]['activations'].T
                )
            except IndexError:
                pass
        return gradients

    def backprogapation(self, input, desirable):
        foward_results = self.foward(input)
        deltas = self.calculate_deltas(foward_results, desirable)
        gradients = self.calculate_gradients(input, foward_results, deltas)

        df = pd.DataFrame([foward_results, deltas, gradients])
        print(df.head(10))
        

class Layer:
    def __init__(self, activation: callable, features_in: int = None, features_out: int = None,
                 weights: np.array = None, bias: np.array = None):
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
        return {'potencials': potencial, 'activations': activation}


# features x amostras
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

W1 = np.array([[0.2, -0.4],
               [0.7,  0.1]])
b1 = np.array([[0.1],
               [-0.3]])

W2 = np.array([[0.5, -0.6]])
b2 = np.array([[0.2]])

mlp = MLP(
    [
        Layer(activation=sigmoid, features_in=2,
              features_out=2, weights=W1, bias=b1),
        Layer(activation=sigmoid, features_in=2,
              features_out=1, weights=W2, bias=b2),
    ],
    loss=quadratic_error
)

mlp.backprogapation(X, Y)
