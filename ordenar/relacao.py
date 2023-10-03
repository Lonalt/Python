from random import randint
from random import choice

#conjunto S

S = {1, 2, 3, 4, 5}

#Contruir tupla com os dados do conjunto S

baseDados = []


#checar simetria 
def simetria(S, dados):
    for i in S:
        for j in S:
            if (i, j) in dados and (j, i) not in dados:
                return False
            else:
                return True

#checar reflexividade
def reflexividade(S, dados):
    for i in S:
        if (i, i) not in dados:
            return False
        else:
            return True
        
#checar transitividade
def transitividade(S, dados):
    for i in S:
        for j in S:
            for k in S:
                if (i, j) in dados and (j, k) in dados and (i, k) not in dados:
                    return False
                else:
                    return True
                
#checar anti-simetria
def anti_simetria(S, dados):
    for i in S:
        for j in S:
            if (i, j) in dados and (j, i) in dados and i != j:
                return False
            else:
                return True


#gerar conjunto de pares ordenados       
def gerarDupla(S, dados):
    base = set()  
    # Adição aleatória de elementos
    for i in range(0, len(S)*10):
        base.add(choice(S), choice(S))  
    dados.extend(list(base))


gerarDupla(S, baseDados)

print(baseDados)
    
        
