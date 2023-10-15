import Dados as dadosBrutos
import Ferramentas as ferramentas
import os
import nltk
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import classification_report

def tratamento_treinamento(treinamento):
    stopWords = ferramentas.criar_stopwords()
    frases_treinamento = ferramentas.aplicar_stemmer(treinamento)
    Palavras_treinamento = dadosBrutos.coletar_palavras(frases_treinamento)
    frequencia_treinamento = ferramentas.frequencia_palavras(Palavras_treinamento)
    base_completa_treinamento = nltk.classify.apply_features(ferramentas.extrair_palavras, frases_treinamento)
    return base_completa_treinamento

def tratamento_teste(teste):
    stopWords = ferramentas.criar_stopwords()
    frases_teste = ferramentas.aplicar_stemmer(teste)
    Palavras_teste = dadosBrutos.coletar_palavras(frases_teste)
    frequencia_teste = ferramentas.frequencia_palavras(Palavras_teste)
    base_completa_teste = nltk.classify.apply_features(ferramentas.extrair_palavras_teste, frases_teste)
    return base_completa_teste

def classificador(base_completa_treinamento):
    classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)
    return classificador

def erros_totais(classificador, base_completa_teste):
    erros = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        if resultado != classe:
            erros.append((classe, resultado, frase))
    return len(erros)

def calcular_acuracia(classificador, base_completa_teste):
    return nltk.classify.accuracy(classificador, base_completa_teste)

def calcular_precisao(matriz, tag):
    return ConfusionMatrix.precision(matriz, tag)

def calcular_recall(matriz, tag):
    return ConfusionMatrix.recall(matriz, tag)

def calcular_f1(matriz, tag):
    return ConfusionMatrix.f_measure(matriz, tag)

def relatorio(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    print(classification_report(esperado, previsto, target_names=['feliz', 'triste', 'neutro']))

def matriz_confusao(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    matriz = ConfusionMatrix(esperado, previsto)
    return matriz

def analisador_manual(classificador, frase):
    testeStemming = []
    stemmer = nltk.stem.RSLPStemmer()
    for (palavras) in frase.split():
        comStem = [p for p in palavras.split()]
        testeStemming.append(str(stemmer.stem(comStem[0])))
    novo = ferramentas.extrair_palavras(testeStemming)
    distribuicao = classificador.prob_classify(novo)
    for classe in distribuicao.samples():
        print(f"{classe}: {distribuicao.prob(classe):.5}")
    print()


