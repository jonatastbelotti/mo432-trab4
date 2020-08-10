# Função que transforma um dicionário Python em uma String com as chaves e os valores
def dict_string(dicionario={}):
    if dicionario is None or not dicionario.keys():
        return ""

    partes = list()
    for k, v in dicionario.items():
        partes.append("%s=%s" % (k, str(v)))

    return ", ".join(partes)


# Função que recebe um float com uma quantidade de segundos e retorna uma String com esse tempo formatado
def format_tempo(segundos):
    temp = segundos
    unidade = "segundos"

    if temp >= 60:
        temp = temp / 60.0
        unidade = "minutos"

    if temp >= 60:
        temp = temp / 60.0
        unidade = "horas"

    return "%.2f %s" % (temp, unidade)