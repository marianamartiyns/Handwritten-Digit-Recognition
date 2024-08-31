import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Função de construção da lista de pontos classificados incorretamente
def constroiListaPCI(x_bias, y, w):
    predictions = np.sign(np.dot(x_bias, w))  # produto escalar
    l = [i for i in range(len(y)) if predictions[i] != y[i]]  # índices dos pontos incorretos
    return l

# Algoritmo de Aprendizagem do Perceptron
def PLA(X, y, max_it):
    x_bias = np.c_[np.ones(X.shape[0]), X]  # Adiciona o bias
    w = np.zeros(x_bias.shape[1])  # Inicializa pesos
    it = 0
    
    while it < max_it:
        l = constroiListaPCI(x_bias, y, w)
        if len(l) == 0:
            break
        
        idx = np.random.choice(l)
        x_i = x_bias[idx]
        y_i = y[idx]
        w += y_i * x_i
        it += 1
    
    return it, w

def classify(X, w):
    x_bias = np.c_[np.ones(X.shape[0]), X]  # Adiciona o bias
    return np.sign(np.dot(x_bias, w))
