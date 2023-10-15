import pandas
import csv
import os 

# Gera um arquivo .csv com uma lista de entrada
def gerar_csv(lista, nome_arquivo, indice):
    current_directory = os.getcwd()
    caminho = os.path.join(current_directory, "Arquivos", nome_arquivo + indice + ".csv") 
    with open(caminho, 'w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo, delimiter=';')
        for linha in lista:
            escritor.writerow(linha)

# Abre o arquivo para leitura
def abrir_arquivo_base():
    current_directory = os.getcwd()
    caminho = os.path.join(current_directory, "Arquivos", "Base.csv")
    with open(caminho, 'r', newline='', encoding='utf-8') as arquivo:
        leitor = csv.reader(arquivo, delimiter=';')
        next(leitor, None)
        Base_dados = []
        for linha in leitor:
            frase, sentimento = linha
            Base_dados.append((frase.strip(), sentimento.strip()))
    arquivo.close()
    return Base_dados

def abrir_arquivo_teste(indice):
    current_directory = os.getcwd()
    caminho = os.path.join(current_directory, "Arquivos", "teste" + indice + ".csv")
    with open(caminho, 'r', newline='', encoding='utf-8') as arquivo:
        leitor = csv.reader(arquivo, delimiter=';')
        next(leitor, None)
        teste = []
        for linha in leitor:
            frase, sentimento = linha
            teste.append((frase.strip(), sentimento.strip()))
    arquivo.close()
    return teste

# Coleta as palavras da base de dados
def coletar_palavras(frases):
    todasPalavras = []
    for (palavras, sentimento) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras
