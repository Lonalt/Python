import nltk
import sklearn
import numpy as np
from scipy import stats
import Dados as dadosBrutos

# Separa a base de dados em treinamento e teste
def separar_dados(indice):
    Base_dados = dadosBrutos.abrir_arquivo_base()
    treinamento, teste = sklearn.model_selection.train_test_split(Base_dados, test_size=0.3)
    dadosBrutos.gerar_csv(teste, "teste", indice)
    return treinamento, teste

# Cria uma lista com as stopwords
def criar_stopwords():
    stopWords = nltk.corpus.stopwords.words('portuguese')
    return stopWords

# Aplica o algoritmo de Stemming
def aplicar_stemmer(frases):
    stemmer = nltk.stem.RSLPStemmer()
    stopWords = criar_stopwords()
    frasesStemming = []
    for (palavras, sentimento) in frases:
        comStemming = [str(stemmer.stem(p)) for p in palavras.split() if stemmer.stem(p) not in stopWords]
        frasesStemming.append((comStemming, sentimento))
    return frasesStemming

def aplicar_tratamento_sem_stemmer(frases):
    stopWords = criar_stopwords()
    frases_tratadas = []
    for (palavras, sentimento) in frases:
        palavras_sem_stemming = [p for p in palavras.split() if p not in stopWords]
        frases_tratadas.append((palavras_sem_stemming, sentimento))
    return frases_tratadas

# frequencia das palavras
def frequencia_palavras(palavras):
    return nltk.FreqDist(palavras)

# bucar as palavras mais frequentes
def buscar_palavras_frequentes(frequencia):
    return frequencia.keys()

# Extrair as palavras das frases
def extrair_palavras(documento):
    doc = set(documento)
    caracteristicas = {}
    frequencia = frequencia_palavras(doc)
    palavras_frequentes = buscar_palavras_frequentes(frequencia)
    for palavras in palavras_frequentes:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def extrair_palavras_teste(documento):
    doc = set(documento)
    caracteristicas = {}
    frequencia = frequencia_palavras(doc)
    palavras_frequentes = buscar_palavras_frequentes(frequencia)
    for palavras in palavras_frequentes:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# calcula media de vetor

def media_vetor(vetor):
    media = np.mean(vetor)
    media = round(media, 2)
    return media

# calcula desvio padrao de vetor

def desvio_padrao(vetor):
    dp = np.std(vetor)
    dp = round(dp, 2)
    return dp

# Aplica o teste t

def teste_t(vetor1, vetor2):
    t_statistic, p_value = stats.ttest_ind(vetor1, vetor2)
    t_statistic = round(t_statistic, 4)
    p_value = round(p_value, 12)
    print("\n\t\tTeste T:\n")
    print(f"\tT-Statistic: {t_statistic}")
    print(f"\tP-Value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("\n\tAs médias são estatisticamente diferentes.")
    else:
        print("\n\tNão há diferença estatística significativa entre as médias.")

