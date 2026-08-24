def combination(lista):
    if not lista:
        return [[]]

    restantes = combination(lista[1:])
    resultados = [element + [lista[0]] for element in restantes]

    return restantes + resultados

def test_empty_set():
    assert combination([]) == [[]]


def test_unique_element():
    assert combination(["a"]) == [[], ["a"]]


def test_two_elements():

    assert combination(["a", "b"]) == [[], ["b"], ["a"], ["b", "a"]]