import time

from trab4.lib import dict_string, format_tempo
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error

# Número de núcleos que serão utilizados, -1 usa todos, -2 deixa um sem usar
N_JOBS = -1


class Treinador():
    """
    Classe que implementa as rotinas para treinar os modelos de classificação e de previsão.
    Responsável por testar os hiperparâmetros de cada modelo e medir a versão final no conjunto de medida.
    """

    melhor_parametros = None
    melhor_score = None
    melhor_modelo = None
    melhor_resultado = None

    def treinar_modelo(self, x_treino, y_treino, x_medida, y_medida, modelos, scoring="accuracy", maior_melhor=True):
        """
        Função que realiza o treino dos modelos, verificando o melhor número de entradas e os melhores parâmetros.
        """
        self.melhor_parametros = None
        self.melhor_score = float("-inf") if maior_melhor else float("inf")
        self.melhor_modelo = None
        tempo_inicial = time.time()

        for modelo in modelos:
            # Testando todas as combinações de todos os parâmetros
            grid = GridSearchCV(estimator=modelo['classe'](), param_grid=modelo['parametros'], scoring=scoring, cv=TimeSeriesSplit(n_splits=5), n_jobs=N_JOBS)
            grid.fit(x_treino, y_treino)
            score_treino = abs(grid.best_score_)
            parametros = grid.best_params_

            # Vendo resultados no conjunto de medida
            modelo_final = modelo['classe'](**parametros)
            modelo_final = modelo_final.fit(x_treino, y_treino)
            resultado_final = modelo_final.predict(x_medida)
            if scoring == "accuracy":
                score_final = accuracy_score(y_medida, resultado_final)
            elif scoring == "neg_root_mean_squared_error":
                score_final = mean_squared_error(y_medida, resultado_final, squared=False)

            print("%s: %s -> (%.6f, %.6f)" % (modelo['nome'], dict_string(grid.best_params_), score_treino, score_final))

            # Verificando se o modelo atual é melhor que os anteriores
            if (maior_melhor and score_final > self.melhor_score) or (not maior_melhor and score_final < self.melhor_score):
                self.melhor_parametros = parametros
                self.melhor_score = score_final
                self.melhor_modelo = modelo
                self.melhor_resultado = resultado_final

        # Calculando tempo total
        tempo_final = time.time()
        tempo_total = tempo_final - tempo_inicial

        # Imprimindo resultados
        print("\n")
        print("Tempo execução: %s" % format_tempo(tempo_total))
        print("Melhor modelo: %s" % self.melhor_modelo['nome'])
        print("Parâmetros: %s" % dict_string(self.melhor_parametros))
        print("Score conjunto medida: %.6f" % self.melhor_score)
