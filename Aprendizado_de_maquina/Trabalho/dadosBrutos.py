import pandas as pandas
import csv

# Abre o arquivo para leitura
def abrir_arquivo():
    with open("/home/lonalt/Área de Trabalho/Python/Aprendizado_de_maquina/Trabalho/Base.csv", 'r', newline='', encoding='utf-8') as arquivo:
        leitor = csv.reader(arquivo, delimiter=';')
        next(leitor, None)
        Base_dados = []
        for linha in leitor:
            frase, sentimento = linha
            Base_dados.append((frase.strip(), sentimento.strip()))
    return Base_dados

# Cria um DataFrame com os dados
def criar_dataframe(Base_dados):
    base = pandas.DataFrame(Base_dados, columns=['Frase', 'Sentimento'])
    return base

# Coleta as palavras da base de dados
def coletar_palavras(frases):
    todasPalavras = []
    for (palavras, sentimento) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras

# Gera um arquivo .csv com uma lista de entrada
def gerar_csv(lista, nome_arquivo):
    caminho = "/home/lonalt/Área de Trabalho/Python/Aprendizado_de_maquina/Trabalho/" + nome_arquivo + ".csv"
    with open(caminho, 'w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo, delimiter=';')
        for linha in lista:
            escritor.writerow(linha)

if __name__ == "__main__":
    print("Fim do programa dadosBrutos.py")

