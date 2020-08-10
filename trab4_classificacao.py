import numpy as np
import random

from trab4.dados import Dados
from trab4.treinador import Treinador
from trab4.lib import dict_string
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# DEFINIÇÕES
ARQUIVO_ENTRADA = "./dados4.csv"
PORCENT_MEDIDA = 10
PORCENT_TREINAMENTO = 100 - PORCENT_MEDIDA


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    # DEFININDO OS MODELOS QUE SERÃO TESTADOS
    MODELOS = [
        {
            'nome': "Regressão Logística (sem regularização)",
            'classe': LogisticRegression,
            'parametros': {}
        },
        {
            'nome': "Regressão Logística com Regularização L2",
            'classe': LogisticRegression,
            'parametros': {
                'penalty': ["l2"],
                'max_iter': [200],
                'C': 10 ** np.random.uniform(-3, 3, 10)
            }
        },
        {
            'nome': "LDA",
            'classe': LinearDiscriminantAnalysis,
            'parametros': {}
        },
        {
            'nome': "QDA",
            'classe': QuadraticDiscriminantAnalysis,
            'parametros': {}
        },
        # {
        #     'nome': "SVM Linear",
        #     'classe': LinearSVC,
        #     'parametros': {
        #         'C': 2 ** np.random.uniform(-5, 15, 10)
        #     }
        # },
        # {
        #     'nome': "SVM com kernel RBF",
        #     'classe': SVC,
        #     'parametros': {
        #         'kernel': ["rbf"],
        #         'C': 2 ** np.random.uniform(-5, 15, 5),
        #         'gamma': 2 ** np.random.uniform(-9, 3, 5)
        #     }
        # },
        # {
        #     'nome': "Naive Bayes",
        #     'classe': GaussianNB,
        #     'parametros': {}
        # },
        # {
        #     'nome': "KNN",
        #     'classe': KNeighborsClassifier,
        #     'parametros': {
        #         'n_neighbors': random.choices(np.arange(1, 302, 2), k=10)
        #     }
        # },
        # {
        #     'nome': "MLP",
        #     'classe': MLPClassifier,
        #     'parametros': {
        #         'max_iter': [400],
        #         'hidden_layer_sizes': np.arange(5, 100+1, 5)
        #     }
        # },
        # {
        #     'nome': "Arvore de decisão",
        #     'classe': DecisionTreeClassifier,
        #     'parametros': {
        #         'ccp_alpha': np.random.uniform(0, 0.04, 10)
        #     }
        # },
        # {
        #     'nome': "GBM",
        #     'classe': GradientBoostingClassifier,
        #     'parametros': {
        #         'n_estimators': np.random.uniform(5, 100, 5).astype("int32"),
        #         'learning_rate': np.random.uniform(0.01, 0.3, 5),
        #         'max_depth': [2, 3, 4, 5]
        #     }
        # }
    ]

    # Lendo os dados de entrada
    dados = Dados(ARQUIVO_ENTRADA, PORCENT_TREINAMENTO, PORCENT_MEDIDA)

    melhor_num_entradas = None
    melhor_modelo = None
    melhor_parametros = None
    melhor_score = float("-inf")

    for num_entradas in np.arange(1, 5+1, 1):
        print("\n====================================================================================================")
        print("Número de entradas -> %d" % num_entradas)

        x_treino, y_treino, x_medida, y_medida = dados.get_dados_classificacao(num_entradas)

        treinador = Treinador()
        treinador.treinar_modelo(x_treino, y_treino, x_medida, y_medida, MODELOS, scoring="accuracy", maior_melhor=True)

        # Verficando qual o melhor modelo dentre todos
        if treinador.melhor_score > melhor_score:
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
    print("Melhor acurária conjunto medida: %.6f" % melhor_score)