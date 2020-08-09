from trab4.lib import dict_string
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


class Treinador():
    """
    Classe que implementa as rotinas para treinar os modelos de classificação e de previsão.
    Responsável por testar os hiperparâmetros de cada modelo e medir a versão final no conjunto de medida.
    """

    melhor_parametros = None
    melhor_score = None
    melhor_modelo = None

    def treinar_modelo(self, x_treino, y_treino, x_medida, y_medida, modelos, scoring="accuracy", classificacao=True, maior_melhor=True):
        """
        Função que realiza o treino dos modelos, verificando o melhor número de entradas e os melhores parâmetros.
        """
        self.melhor_parametros = None
        self.melhor_score = float("-inf") if maior_melhor else float("inf")
        self.melhor_modelo = None

        for modelo in modelos:
            # Testando todas as combinações de todos os parâmetros
            grid = GridSearchCV(estimator=modelo['classe'](), param_grid=modelo['parametros'], scoring=scoring, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1)
            grid.fit(x_treino, y_treino)
            score_treino = grid.best_score_
            parametros = grid.best_params_

            # Vendo resultados no conjunto de medida
            modelo_final = modelo['classe'](**parametros)
            modelo_final = modelo_final.fit(x_treino, y_treino)
            score_final = modelo_final.score(x_medida, y_medida)

            print("%s: %s -> (%.6f, %.6f)" % (modelo['nome'], dict_string(grid.best_params_), score_treino, score_final))

            # Verificando se o modelo atual é melhor que os anteriores
            if (maior_melhor and score_final > self.melhor_score) or (not maior_melhor and score_final < self.melhor_score):
                self.melhor_parametros = parametros
                self.melhor_score = score_final
                self.melhor_modelo = modelo

        print("\nMelhor modelo: %s" % self.melhor_modelo['nome'])
        print("Parâmetros: %s" % dict_string(self.melhor_parametros))
        print("Score: %.6f" % self.melhor_score)
