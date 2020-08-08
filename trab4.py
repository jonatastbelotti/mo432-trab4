import argparse
import numpy as np
import pandas as pd

# DEFINIÇÕES
ARQUIVO_ENTRADA = "./dados4.csv"
PORCENT_MEDIDA = 10
PORCENT_TESTE = 10
PORCENT_TREINAMENTO = 100 - PORCENT_TESTE - PORCENT_MEDIDA


# Função que faz a leitura do arquivo de entrada
def ler_arquivo_csv(arquivo):
    df = pd.read_csv(arquivo).drop(["Data"], axis=1)
    df = df.replace(to_replace="-", value=np.nan)
    resp = df["Taxa"].to_numpy(dtype=np.number, na_value=None)

    resp = np.flip(resp)

    print("Arquivo de entrada: %s com %d registros" % (arquivo, resp.shape[0]))

    return resp


# Função que completa os dados que estão faltando
def completar_dados(dados):
    num = 0

    # Todo valor nulo é preenchido com a média do anterior e próximo
    for i, val in enumerate(dados):
        if np.isnan(val):
            val_anterior, val_proximo = None, None
            num += 1

            for i_anterior in range(i, 0, -1):
                if not np.isnan(dados[i_anterior]):
                    val_anterior = dados[i_anterior]
                    break

            for i_proximo in range(i+1, dados.shape[0]):
                if not np.isnan(dados[i_proximo]):
                    val_proximo = dados[i_proximo]
                    break
            
            if val_anterior is not None and val_proximo is not None:
                dados[i] = (val_anterior + val_proximo) / 2.0
            else:
                dados[i] = val_proximo if val_proximo is not None else dados[i]
                dados[i] = val_anterior if val_anterior is not None else dados[i]
    
    print("Foram corrigidos %d registros nulos" % num)

    return dados


# Função que separa os dados em treinamento, teste e medida
def separar_dados(dados, porcent_trei, porcent_teste, porcent_medida):
    registros_medida = int(dados.shape[0] * (porcent_medida/100))
    registros_teste = int(dados.shape[0] * (porcent_teste/100))
    registros_treino = int(dados.shape[0] - registros_medida - registros_teste)

    treino = dados[0:registros_treino]
    teste = dados[registros_treino:registros_treino+registros_teste]
    medida = dados[registros_treino+registros_teste:registros_treino+registros_teste+registros_medida]

    return treino, teste, medida


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    dados = ler_arquivo_csv(ARQUIVO_ENTRADA)
    dados = completar_dados(dados)
    dados_treino, dados_teste, dados_medida = separar_dados(dados, PORCENT_TREINAMENTO, PORCENT_TESTE, PORCENT_MEDIDA)
