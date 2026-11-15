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


class MLP:  # SGD
    def __init__(self, layers: list, loss: callable, eta=0.1):
        self.layers = layers
        self.loss = loss
        self.eta = eta

    def predict(self, input: np.array):
        foward_results = []
        for layer in self.layers:
            result = layer.foward(input)
            foward_results.append(result)
            input = result['activations']
        return foward_results

    def get_out(self, input):
        pred = self.predict(input)
        return pred[-1]['activations']

    def report(self, input, desireble):
        return self.loss(self.get_out(input), desireble, report=True)

    def __calculate_deltas(self, foward_results, desirable):
        # dimensões da matriz δi(output,input)
        # ou seja δi(neuronio, amostra)
        deltas = []
        # fixar desirable e chamar passando somente x
        exit_delta = hard_derivate(
            self.loss,
            foward_results[-1]['activations'],
            desirable=desirable
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

    def __calculate_gradients(self, input, foward_results, deltas):
        weight_gradients = []
        weight_gradients.append(deltas[0] @ input.T)
        
        for i in range(1, len(self.layers)):
            weight_gradients.append( # δ @ yᵀ = Σₙ δₙ · yₙᵀ
                (deltas[i] @ foward_results[i-1]['activations'].T)
                / input.shape[1]
            ) 
        bias_gradients = [np.sum(d, axis=1, keepdims=True) for d in deltas]
        return (weight_gradients, bias_gradients)

    def get_gradients(self, input, desirable):
        foward_results = self.predict(input)
        deltas = self.__calculate_deltas(foward_results, desirable)
        gradients = self.__calculate_gradients(input, foward_results, deltas)
        return gradients

    def update_layers(self, input, desirable):
        gradients = self.get_gradients(input, desirable)
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


# features x amostras
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

W1 = np.array([[0., -0.],
               [0.,  0.]])
b1 = np.array([[0.],
               [-0.]])

W2 = np.array([[0., 0.]])
b2 = np.array([[0.2]])

model = MLP(
    [
        Layer(activation=sigmoid,
                weights=W1, bias=b1),
        Layer(activation=sigmoid,
                weights=W2, bias=b2),
    ],
    loss=quadratic_error,
    eta=0.1
)

losses = []
for i in range(100):
    losses.append(model.report(X, Y))
    model.update_layers(X,Y)
plt.plot(list(range(len(losses))), losses)
plt.show()