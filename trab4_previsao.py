import numpy as np
import random

from trab4.dados import Dados
from trab4.treinador import Treinador
from trab4.lib import dict_string
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
        # {
        #     'nome': "Regressão Linear",
        #     'classe': LinearRegression,
        #     'parametros': {}
        # },
        # {
        #     'nome': "Linear com regularização L2",
        #     'classe': Ridge,
        #     'parametros': {
        #         'alpha': 10 ** np.random.uniform(-3, 3, 10)
        #     }
        # },
        # {
        #     'nome': "Linear com regularização L1",
        #     'classe': Lasso,
        #     'parametros': {
        #         'alpha': 10 ** np.random.uniform(-3, 3, 10)
        #     }
        # },
        # {
        #     'nome': "SVM Linear",
        #     'classe': LinearSVR,
        #     'parametros': {
        #         'epsilon': [0.1, 0.3],
        #         'C': 2 ** np.random.uniform(-5, 15, 10)
        #     }
        # },
        # {
        #     'nome': "SVM com kernel RBF",
        #     'classe': SVR,
        #     'parametros': {
        #         'kernel': ["rbf"],
        #         'epsilon': [0.1, 0.3],
        #         'C': 2 ** np.random.uniform(-5, 15, 1),
        #         'gamma': 2 ** np.random.uniform(-9, 32, 1)
        #     }
        # },
        # {
        #     'nome': "KNN",
        #     'classe': KNeighborsRegressor,
        #     'parametros': {
        #         'n_neighbors': np.random.uniform(1, 1000+1, 10)
        #     }
        # },
        # {
        #     'nome': "MLP",
        #     'classe': MLPRegressor,
        #     'parametros': {
        #         'hidden_layer_sizes': np.arange(5, 20+1, 5)
        #     }
        # },
        # {
        #     'nome': "Árvore de decisão",
        #     'classe': DecisionTreeRegressor,
        #     'parametros': {
        #         'ccp_alpha': np.random.uniform(0.0, 0.04, 10)
        #     }
        # },
        {
            'nome': "GBM",
            'classe': GradientBoostingRegressor,
            'parametros': {
                'n_estimators': [35],#np.random.uniform(5, 100+1, 5).astype("int32"),
                'learning_rate': [0.12405787499398713],#np.random.uniform(0.01, 0.3, 5),
                'max_depth': [2]#[2, 3, 4, 5]
            }
        }
    ]

    # Lendo os dados de entrada
    dados = Dados(ARQUIVO_ENTRADA, PORCENT_TREINAMENTO, PORCENT_MEDIDA)

    melhor_num_entradas = None
    melhor_modelo = None
    melhor_parametros = None
    melhor_score = float("inf")

    for num_entradas in [14]:#np.arange(2, 15+1, 1):
        print("\n====================================================================================================")
        print("Número de entradas -> %d" % num_entradas)

        x_treino, y_treino, x_medida, y_medida = dados.get_dados_previsao(num_entradas)

        treinador = Treinador()
        treinador.treinar_modelo(x_treino, y_treino, x_medida, y_medida, MODELOS, scoring="neg_root_mean_squared_error", maior_melhor=False)

        # Verficando qual o melhor modelo dentre todos
        if treinador.melhor_score < melhor_score:
            melhor_num_entradas = num_entradas
            melhor_modelo = treinador.melhor_modelo
            melhor_parametros = treinador.melhor_parametros
            melhor_score = treinador.melhor_score

        print("\n")

    # Imprimindo melhor resultado entre todos números de entradas
    print("====================================================================================================")
    print("====================================================================================================")
    print("====================================================================================================")
    print("Melhor número entradas: %d" % melhor_num_entradas)
    print("Melhor modelo: %s" % melhor_modelo['nome'])
    print("Melhores parâmetros: %s" % dict_string(melhor_parametros))
    print("Melhor RMSE conjunto medida: %.6f" % melhor_score)
