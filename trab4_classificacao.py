import argparse
from trab4.dados import Dados

# DEFINIÇÕES
ARQUIVO_ENTRADA = "./dados4.csv"
PORCENT_MEDIDA = 10
PORCENT_TREINAMENTO = 100 - PORCENT_MEDIDA


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    dados = Dados(ARQUIVO_ENTRADA, PORCENT_TREINAMENTO, PORCENT_MEDIDA)
    x_treino, y_treino, x_medida, y_medida = dados.get_dados_classificacao(5)
    