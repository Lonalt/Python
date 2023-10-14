import Dados as dadosBrutos
import Ferramentas as ferramentas
import os
import nltk
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import classification_report

def tratamento_treinamento_sem_stemming(treinamento):
    stopWords = ferramentas.criar_stopwords()
    frases_treinamento = treinamento
    Palavras_treinamento = dadosBrutos.coletar_palavras(frases_treinamento)
    frequencia_treinamento = ferramentas.frequencia_palavras(Palavras_treinamento)
    base_completa_treinamento = nltk.classify.apply_features(ferramentas.extrair_palavras, frases_treinamento)
    return base_completa_treinamento

def tratamento_teste_sem_stemming(teste):
    stopWords = ferramentas.criar_stopwords()
    frases_teste = teste
    Palavras_teste = dadosBrutos.coletar_palavras(frases_teste)
    frequencia_teste = ferramentas.frequencia_palavras(Palavras_teste)
    base_completa_teste = nltk.classify.apply_features(ferramentas.extrair_palavras_teste, frases_teste)
    return base_completa_teste

def classificador_sem_stemming(base_completa_treinamento):
    classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)
    return classificador

def erros_totais_sem_stemming(classificador, base_completa_teste):
    erros = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        if resultado != classe:
            erros.append((classe, resultado, frase))
    return len(erros)

def calcular_acuracia_sem_stemming(classificador, base_completa_teste):
    return nltk.classify.accuracy(classificador, base_completa_teste)

def calcular_precisao_sem_stemming(matriz, tag):
    return ConfusionMatrix.precision(matriz, tag)

def calcular_recall_sem_stemming(matriz, tag):
    return ConfusionMatrix.recall(matriz, tag)

def calcular_f1_sem_stemming(matriz, tag):
    return ConfusionMatrix.f_measure(matriz, tag)

def relatorio_sem_stemming(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    print(classification_report(esperado, previsto, target_names=['feliz', 'triste', 'neutro']))

def matriz_confusao_sem_stemming(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    matriz = ConfusionMatrix(esperado, previsto)
    return matriz

def analisador_manual_sem_stemming(classificador, frase):
    teste_sem_stemming = []
    for (palavras) in frase.split():
        semStem = [p for p in palavras.split()]
        teste_sem_stemming.append(str(semStem[0]))
    novo = ferramentas.extrair_palavras_teste(teste_sem_stemming)
    distribuicao = classificador.prob_classify(novo)
    for classe in distribuicao.samples():
        print(f"{classe}: {distribuicao.prob(classe):.5}")
    print()


