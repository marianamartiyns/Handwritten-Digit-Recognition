import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from matplotlib.lines import Line2D

# Definição da classe LogisticRegression_ conforme fornecido anteriormente
class RegressaoLogistica_:
    def __init__(self, eta=0.001, tmax=1500, epsilon=1e-5):
        self.eta = eta
        self.tmax = tmax
        self.epsilon = epsilon
        self.w = None

    def fit(self, _X, _y):
        X = np.array(_X)
        y = np.array(_y)
        N, d = X.shape

        # Adiciona o termo de bias (intercepto)
        X = np.c_[np.ones(N), X]  # Adiciona uma coluna de 1s
        self.w = np.zeros(X.shape[1])
        
        for t in range(self.tmax):
            # Calcula as probabilidades usando a função sigmoide
            z = np.dot(X, self.w)
            probs = 1 / (1 + np.exp(-z))

            # Calcula o gradiente
            errors = y - probs
            gradient = -np.dot(X.T, errors) / N

            # Atualiza os pesos
            self.w -= self.eta * gradient

            # Verifica a convergência
            if LA.norm(gradient) < self.epsilon:
                break

    def predict_prob(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]  # Adiciona uma coluna de 1s
        z = np.dot(X, self.w)
        probs = 1 / (1 + np.exp(-z))
        return probs

    def predict(self, X):
        probs = self.predict_prob(X)
        return (probs >= 0.5).astype(int) * 2 - 1

    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        if self.w is None or len(self.w) != 3:
            raise ValueError("O modelo não está ajustado ou não tem 3 coeficientes.")
        return (-self.w[0] + shift - self.w[1] * regressionX) / self.w[2]