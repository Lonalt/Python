import Dados as dadosBrutos
import Ferramentas as ferramentas
import os
import nltk
from nltk.metrics import ConfusionMatrix

def gerarBase():
    data_Set = dadosBrutos.abrir_arquivo()
    treinamento, teste = ferramentas.separar_dados(data_Set)
    dadosBrutos.gerar_csv(treinamento, "treinamento")
    dadosBrutos.gerar_csv(teste, "teste")
    return treinamento, teste

def tratamento(treinamento, teste):
    stopWords = ferramentas.criar_stopwords()
    frases_treinamento = ferramentas.aplicar_stemmer(treinamento)
    frases_teste = ferramentas.aplicar_stemmer(teste)
    Palavras_treinamento = dadosBrutos.coletar_palavras(frases_treinamento)
    Palavras_teste = dadosBrutos.coletar_palavras(frases_teste)
    frequencia_treinamento = ferramentas.frequencia_palavras(Palavras_treinamento)
    frequencia_teste = ferramentas.frequencia_palavras(Palavras_teste)
    base_completa_treinamento = nltk.classify.apply_features(ferramentas.extrair_palavras, frases_treinamento)
    base_completa_teste = nltk.classify.apply_features(ferramentas.extrair_palavras_teste, frases_teste)
    return base_completa_treinamento, base_completa_teste

def classificador(base_completa_treinamento):
    classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)
    return classificador

def errosTotais(classificador, base_completa_teste):
    erros = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        if resultado != classe:
            erros.append((classe, resultado, frase))
    print('Total de erros:', len(erros))
    print()

def calcular_acuracia(esperado, previsto):
    corretos = 0
    for i in range(len(esperado)):
        if esperado[i] == previsto[i]:
            corretos += 1
    acuracia = corretos / len(esperado)
    print('Acuracia: {:.5f}'.format(acuracia))

def calcular_precisao(esperado, previsto):
    verdadeiros_positivos = 0
    falsos_positivos = 0
    for i in range(len(esperado)):
        if esperado[i] == previsto[i]:
            verdadeiros_positivos += 1
    for i in range(len(esperado)):
        if esperado[i] == 'feliz' and (previsto[i] == 'neutro' or previsto[i] == 'triste'):
            falsos_positivos += 1
        if esperado[i] == 'neutro' and previsto[i] == 'triste':
            falsos_positivos += 1
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
    print('Precisão: {:.5f}'.format(precisao))

def matrizConfusao(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    matriz = ConfusionMatrix(esperado, previsto)
    print(matriz)
    print()
    calcular_acuracia(esperado, previsto)
    calcular_precisao(esperado, previsto)

def tags(classificador, solicitado):
    print(classificador.labels())
    print(classificador.show_most_informative_features(solicitado))
    print()

def probabilidade(classificador, teste):
    distribuição = classificador.prob_classify(teste)
    for classe in distribuição.samples():
        print("%s: %f" % (classe, distribuição.prob(classe)))
        print()

def testeAutomatico(classificador):
    teste = [
                'Viagens são bem simples e baratas',
                'Amores vem e vão, mas o que fica são as lembranças',
                'Em uma manhã ensolarada de domingo, reuni meus amigos em um café à beira-mar para celebrar meu aniversário.'
            ]
    for i in teste:
        print(i)
        testeStemming = []
        stemmer = nltk.stem.RSLPStemmer()
        for (palavras) in i.split():
            comStem = [p for p in palavras.split()]
            testeStemming.append(str(stemmer.stem(comStem[0])))
        novo = ferramentas.extrair_palavras(testeStemming)
        distribuição = classificador.prob_classify(novo)
        for classe in distribuição.samples():
            print("%s: %f" % (classe, distribuição.prob(classe)))
        print()

def AnalisadorManual(classificador):
    print('Digite a quantidade de testes que deseja realizar:')
    solicitado = int(input())
    for i in range(solicitado):
        print('Digite a frase que deseja testar:')
        teste = input()
        testeStemming = []
        stemmer = nltk.stem.RSLPStemmer()
        for (palavras) in teste.split():
            comStem = [p for p in palavras.split()]
            testeStemming.append(str(stemmer.stem(comStem[0])))
        novo = ferramentas.extrair_palavras(testeStemming)
        probabilidade(classificador, novo)
        print()

os.system('clear')
treinamento, teste = gerarBase()
base_completa_treinamento, base_completa_teste = tratamento(treinamento, teste)
classificador = classificador(base_completa_treinamento)
errosTotais(classificador, base_completa_teste)
matrizConfusao(classificador, base_completa_teste)
tags(classificador, 10)
testeAutomatico(classificador)
