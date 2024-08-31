import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class RegressaoLinear:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # bias
        y = np.array(y)
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # bias
        return X_b.dot(self.w)
    
    def getW(self):
        return self.w

# Função para plot dos modelos no classificador de dígitos completos (1xTodos)
def plot_rl(X, y_true, y_pred, modelos, labels):
    plt.figure(figsize=(12, 8))
        
    # Definir cores para cada classe
    class_colors = {0: 'blue', 1: 'green', 4: 'red', 5: 'orange'}
        
    # Plotar pontos de dados com base na classe prevista
    scatter = plt.scatter(X[:, 0], X[:, 1], c=[class_colors[yi] for yi in y_pred], edgecolor='k', s=40, alpha=0.8)
        
    # Adicionar as retas de decisão para cada modelo
    for modelo, label in zip(modelos, labels):
        w = modelo.getW()
            
        # Adicionando um termo de bias para a plotagem
        bias = w[0]
        coef = w[1:]
            
        # Determinar a cor da linha de decisão
        class_label = int(label.split()[0][1])  # Extrair a classe do rótulo
        line_color = class_colors[class_label]
            
        # Calculando os valores para plotar a reta de decisão
        x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_values = - (coef[0] * x_values + bias) / coef[1]
            
        # Plotando a reta de decisão
        plt.plot(x_values, y_values, label=f'{label} Decision Boundary', linestyle='--', color=line_color)
            
        # Imprimindo a função hipótese
        print(f"Função hipótese para {label}: h(x) = {bias:.3f} + {coef[0]:.3f}*intensidade + {coef[1]:.3f}*simetria")
        
    plt.xlim(40, 160)
    plt.ylim(50, 170)
        
    # Configurando o gráfico
    plt.title('Resultados da Classificação por Regressão Linear com Fronteiras de Decisão')
    plt.xlabel('Intensidade')
    plt.ylabel('Simetria')
        
    # Adicionar uma barra de cores para indicar a classe predita
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
                for color in class_colors.values()]
    plt.legend(handles=handles, labels=list(class_colors.keys()), title='Classe Predita')
    plt.show()