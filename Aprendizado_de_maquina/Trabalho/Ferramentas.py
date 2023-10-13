import nltk
import sklearn

# Separa a base de dados em treinamento e teste
def separar_dados(Base_dados):
    treinamento, teste = sklearn.model_selection.train_test_split(Base_dados, test_size=0.3)
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

