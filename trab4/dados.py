import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use('fivethirtyeight')

class Dados():

    def __init__(self, arquivo, porcent_trei, porcent_medida):
        self.arquivo = arquivo
        self.porcent_trei = porcent_trei
        self.porcent_medida = porcent_medida

        self.ler_arquivo_csv()
        self.completar_dados()
        self.separar_dados()


    def ler_arquivo_csv(self):
        """Função que faz a leitura do arquivo de entrada"""
        df = pd.read_csv(self.arquivo)
        df = df.replace(to_replace="-", value=np.nan)
        resp = df["Taxa"].to_numpy(dtype=np.number, na_value=None)

        self.dados = np.flip(resp)
        self.datas = np.flip(df["Data"].to_numpy())

        print("Arquivo de entrada: %s com %d registros" % (self.arquivo, self.dados.shape[0]))


    def completar_dados(self):
        """Função que completa os dados que estão faltando"""
        num = 0

        # Todo valor nulo é preenchido com a média do anterior e próximo
        for i, val in enumerate(self.dados):
            if np.isnan(val):
                val_anterior, val_proximo = None, None
                num += 1

                for i_anterior in range(i, 0, -1):
                    if not np.isnan(self.dados[i_anterior]):
                        val_anterior = self.dados[i_anterior]
                        break

                for i_proximo in range(i+1, self.dados.shape[0]):
                    if not np.isnan(self.dados[i_proximo]):
                        val_proximo = self.dados[i_proximo]
                        break

                if val_anterior is not None and val_proximo is not None:
                    self.dados[i] = (val_anterior + val_proximo) / 2.0
                else:
                    self.dados[i] = val_proximo if val_proximo is not None else self.dados[i]
                    self.dados[i] = val_anterior if val_anterior is not None else self.dados[i]

        print("Foram corrigidos %d registros nulos" % num)


    def separar_dados(self):
        """Função que separa os dados em treinamento e medida"""
        registros_medida = int(self.dados.shape[0] * (self.porcent_medida/100))
        registros_treino = int(self.dados.shape[0] - registros_medida)

        self.dados_treino = self.dados[0:registros_treino]
        self.dados_medida = self.dados[registros_treino:registros_treino+registros_medida]

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler.fit(self.dados_treino)
        # self.dados_treino = scaler.transform(self.dados_treino)
        # self.dados_medida = scaler.transform(self.dados_medida)

        # self.dados_treino = self.dados_treino.reshape(-1)
        # self.dados_medida = self.dados_medida.reshape(-1)


    def get_dados_classificacao(self, num_entradas, faz_cs, faz_pca):
        """
        Função que separa os conjuntos de entradas e de saídas para classificação.
        """
        return self.__get_dados(num_entradas, classificacao=True, faz_cs=faz_cs, faz_pca=faz_pca)


    def get_dados_previsao(self, tam_janela, faz_cs, faz_pca):
        """
        Função que separa os conjuntos de entradas e de saídas para previsão.
        """
        return self.__get_dados(tam_janela, classificacao=False, faz_cs=faz_cs, faz_pca=faz_pca)


    def __get_dados(self, num_entradas, classificacao=False, faz_cs=True, faz_pca=True):
        """
        Função que separa os conjuntos de entradas e de saídas para classificação e previsão.
        Se classificação 1 significa que aumentou em relação ao dia anterior, 0 que não aumentou.
        Recebe como parâmetro o número de entradas desejado.
        Os primeiros registros do conjunto de treinamento que não contém o número de entradas necessárias são ignorados.
        """
        x_treino, y_treino = list(), list()
        x_medida, y_medida = list(), list()

        # Montando o conjunto com as entradas e saídas do terinamento
        for i, val in enumerate(self.dados_treino):
            entradas = list()

            for i_aux in range(i-num_entradas, i):
                if i_aux >= 0 and i_aux < self.dados_treino.shape[0]:
                    entradas.append(self.dados_treino[i_aux])
                else:
                    break

            if len(entradas) > 0:
                x_treino.append(entradas)
                if classificacao:
                    y_treino.append(1 if val > entradas[-1] else 0)
                else:
                    y_treino.append(val)

        # Montando o conjunto com as entradas e saídas dos dados de medida
        for i, val in enumerate(self.dados_medida):
            entradas = list()

            for i_aux in range(i-num_entradas, i):
                if i_aux >= 0 and i_aux < self.dados_medida.shape[0]:
                    entradas.append(self.dados_medida[i_aux])
                else:
                    entradas.append(self.dados_treino[i_aux])

            if len(entradas) > 0:
                x_medida.append(entradas)
                if classificacao:
                    y_medida.append(1 if val > entradas[-1] else 0)
                else:
                    y_medida.append(val)

        # Convertendo em Arrays Numpy
        x_treino, y_treino = np.array(x_treino), np.array(y_treino)
        x_medida, y_medida = np.array(x_medida), np.array(y_medida)

        # Aplicando o centering and Scaling
        if faz_cs:
            scaler = StandardScaler()
            scaler.fit(x_treino)
            x_treino, x_medida = scaler.transform(x_treino), scaler.transform(x_medida)

        # Aplicando PCA
        if faz_pca:
            pca = PCA(0.9)
            pca.fit(x_treino)
            x_treino = pca.transform(x_treino)
            x_medida = pca.transform(x_medida)
            print("O PCA deixou %d atributos" % x_treino.shape[1])


        return x_treino, y_treino, x_medida, y_medida


    def plotar_grafico_completo(self, caminho="./graficos/dados_completo.png"):
        indices_x, labels_x = list(), list()
        d_antiga = None
        for i, d in enumerate(self.datas):
            d_atual = d.split("-")[0]
            if d_atual != d_antiga:
                indices_x.append(i+1)
                labels_x.append(d_atual)
                d_antiga = d_atual

        plt.figure(figsize=(16, 9))
        plt.title('Dados da série completa')
        plt.plot(self.dados, lw=2)
        plt.xticks(indices_x, labels_x)
        plt.xlabel('Data', fontsize=18)
        plt.ylabel('Taxa de cambio', fontsize=18)
        plt.savefig(caminho, dpi=300)


    def plot_grafico_previsao(self, previsao, caminho="./graficos/previsao.png"):
        indices_x, labels_x = list(), list()
        d_antiga = "05/2018"
        for i, d in enumerate(self.datas[-(previsao.shape[0]):]):
            d_atual = "%s/%s" % (d.split("-")[1], d.split("-")[0])
            if d_atual != d_antiga and any(txt in d_atual for txt in ["05/", "07/", "09/", "11/", "01/", "03/"]):
                indices_x.append(i+1)
                labels_x.append(d_atual)
                d_antiga = d_atual

        plt.figure(figsize=(16, 9))
        plt.title('Valor real vs Valor Previsto')
        plt.plot(self.dados_medida, lw=3)
        plt.plot(previsao, lw=2)
        plt.xticks(indices_x, labels_x)
        plt.xlabel('Data', fontsize=18)
        plt.ylabel('Taxa de cambio', fontsize=18)
        plt.legend(['Valor real', 'Valor previsto'], loc='upper right')
        plt.savefig(caminho, dpi=300)

