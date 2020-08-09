# Função que transforma um dicionário Python em uma String com as chaves e os valores
def dict_string(dicionario={}):
    if dicionario is None or not dicionario.keys():
        return ""

    partes = list()
    for k, v in dicionario.items():
        partes.append("%s=%s" % (k, str(v)))

    return ", ".join(partes)
