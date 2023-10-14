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

def teste_automatico_sem_stemming(classificador):
    teste = [
        'Viagens são bem simples e baratas.', #neutro
        'Amores vem e vão, mas o que fica são as lembranças.', #triste
        'Em uma manhã ensolarada de domingo, reuni meus amigos em um café à beira-mar para celebrar meu aniversário.', #feliz
        'Depois de tanto esforço, finalmente consegui realizar meu sonho.', #feliz
        'Viver é uma arte, e nem todos são artistas.', #neutro
        'O céu estava nublado, mas o ar estava fresco e agradável para um passeio no parque.', #neutro
        'Aquele dia foi o mais feliz da minha vida.', #feliz
        'Ao olhar para trás, só consigo ver os momentos que perdi e as oportunidades que deixei escapar.', #triste
        'Meu coração está partido.', #triste
    ]
    for i in teste:
        print(i)
        novo = ferramentas.extrair_palavras(i.split())
        distribuição = classificador.prob_classify(novo)
        for classe in distribuição.samples():
            print(f"{classe}: {distribuicao.prob(classe):.5}\n")


def analisador_manual_sem_stemming(classificador, frase):
    novo = ferramentas.extrair_palavras(frase.split())
    distribuição = classificador.prob_classify(novo)
    for classe in distribuição.samples():
        print(f"{classe}: {distribuicao.prob(classe):.5}\n")


