import numpy as np
import random
import time

from trab4.dados import Dados
from trab4.treinador import Treinador
from trab4.lib import dict_string, format_tempo
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor





# DEFINIÇÕES
ARQUIVO_ENTRADA = "./dados4.csv"
PORCENT_MEDIDA = 10
PORCENT_TREINAMENTO = 100 - PORCENT_MEDIDA


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    # DEFININDO OS MODELOS QUE SERÃO TESTADOS
    MODELOS = [
        {
            'nome': "Regressão Linear",
            'classe': LinearRegression,
            'parametros': {}
        },
        {
            'nome': "Linear com regularização L2",
            'classe': Ridge,
            'parametros': {
                'alpha': 10 ** np.random.uniform(-3, 3, 10)
            }
        },
        {
            'nome': "Linear com regularização L1",
            'classe': Lasso,
            'parametros': {
                'alpha': 10 ** np.random.uniform(-3, 3, 10)
            }
        },
        {
            'nome': "SVM Linear",
            'classe': LinearSVR,
            'parametros': {
                'epsilon': [0.1, 0.3],
                'C': 2 ** np.random.uniform(-5, 15, 10)
            }
        },
        {
            'nome': "SVM com kernel RBF",
            'classe': SVR,
            'parametros': {
                'kernel': ["rbf"],
                'epsilon': [0.1, 0.3],
                'C': 2 ** np.random.uniform(-5, 15, 10),
                'gamma': 2 ** np.random.uniform(-9, 32, 10)
            }
        },
        {
            'nome': "KNN",
            'classe': KNeighborsRegressor,
            'parametros': {
                'n_neighbors': np.random.uniform(1, 500+1, 10).astype("int32")
            }
        },
        {
            'nome': "MLP",
            'classe': MLPRegressor,
            'parametros': {
                'max_iter': [400],
                'hidden_layer_sizes': np.arange(5, 200+1, 5)
            }
        },
        {
            'nome': "Árvore de decisão",
            'classe': DecisionTreeRegressor,
            'parametros': {
                'ccp_alpha': np.random.uniform(0.0, 0.04, 10)
            }
        },
        {
            'nome': "GBM",
            'classe': GradientBoostingRegressor,
            'parametros': {
                'n_estimators': np.random.uniform(5, 100+1, 10).astype("int32"),
                'learning_rate': np.random.uniform(0.01, 0.3, 10),
                'max_depth': [2, 3, 4, 5]
            }
        }
    ]

    # Lendo os dados de entrada
    dados = Dados(ARQUIVO_ENTRADA, PORCENT_TREINAMENTO, PORCENT_MEDIDA)

    melhor_num_entradas = None
    melhor_modelo = None
    melhor_parametros = None
    melhor_score = float("inf")
    tempo_inicial = time.time()

    for num_entradas in np.arange(2, 15+1, 1):
        print("\n====================================================================================================")
        print("Número de entradas -> %d" % num_entradas)

        x_treino, y_treino, x_medida, y_medida = dados.get_dados_previsao(num_entradas, faz_cs=True, faz_pca=True)

        treinador = Treinador()
        treinador.treinar_modelo(x_treino, y_treino, x_medida, y_medida, MODELOS, scoring="neg_root_mean_squared_error", maior_melhor=False)

        # Verficando qual o melhor modelo dentre todos
        if treinador.melhor_score < melhor_score:
            melhor_num_entradas = num_entradas
            melhor_modelo = treinador.melhor_modelo
            melhor_parametros = treinador.melhor_parametros
            melhor_score = treinador.melhor_score

        print("\n")

    # Calculando tempo total
    tempo_final = time.time()
    tempo_total = tempo_final - tempo_inicial

    # Imprimindo melhor resultado entre todos números de entradas
    print("====================================================================================================")
    print("====================================================================================================")
    print("====================================================================================================")
    print("Tempo execução: %s" % format_tempo(tempo_total))
    print("Melhor número entradas: %d" % melhor_num_entradas)
    print("Melhor modelo: %s" % melhor_modelo['nome'])
    print("Melhores parâmetros: %s" % dict_string(melhor_parametros))
    print("Melhor RMSE conjunto medida: %.6f" % melhor_score)
