
def combination(lista):
    if not lista:
        return [[]] # our base case when the list does not have more elements

    restantes = combination(lista[1:]) # recurisivity over the elements of the list except the first one
    resultados = [element + [lista[0]] for element in restantes] # adding the first element of the list to ones of restantes

    return restantes + resultados # we finally get the the combinations
