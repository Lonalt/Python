#Conjunto S

S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

#Contruir tupla com os dados do conjunto S

dados = []

for i in S:
    for j in S:
        for k in S:
            if i + j + k == 20:
                dados.append((i, j, k))


#total de tuplas

print('Total de tuplas: {}'.format(len(dados)))

#ordenar tuplas em ordem crescente no primeiro elemento

dados = sorted(dados, key=lambda x: x[0])

print(dados)
print()






