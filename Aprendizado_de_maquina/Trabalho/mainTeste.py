import dadosBrutos as dadosBrutos
import ferramentas as ferramentas
import os
import nltk
from nltk.metrics import ConfusionMatrix

# Limpa a tela
os.system('clear')

# Abre arquivo
data_Set = dadosBrutos.abrir_arquivo()

# Separa os dados em treinamento e teste
treinamento, teste = ferramentas.separar_dados(data_Set)

# Cria stopwords
stopWords = ferramentas.criar_stopwords()

# Gerar arquivo .csv com os dados de treinamento e teste
dadosBrutos.gerar_csv(treinamento, "treinamento")
dadosBrutos.gerar_csv(teste, "teste")

# Aplica Stemmer
frases_treinamento = ferramentas.aplicar_stemmer(treinamento)
frases_teste = ferramentas.aplicar_stemmer(teste)

# Coleta as palavras
Palavras_treinamento = dadosBrutos.coletar_palavras(frases_treinamento)
Palavras_teste = dadosBrutos.coletar_palavras(frases_teste)

# Frequencia das palavras
frequencia_treinamento = ferramentas.frequencia_palavras(Palavras_treinamento)
frequencia_teste = ferramentas.frequencia_palavras(Palavras_teste)

# extrair as palavras das frases

base_completa_treinamento = nltk.classify.apply_features(ferramentas.extrair_palavras, frases_treinamento)
base_completa_teste = nltk.classify.apply_features(ferramentas.extrair_palavras_teste, frases_teste)

# Cria o classificador
classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)

#print(classificador.labels())
#print(classificador.show_most_informative_features(10))
#print(nltk.classify.accuracy(classificador, base_completa_teste))
#print()

erros = []
for (frase, classe) in base_completa_teste:
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))

esperado = []
previsto = []
for (frase, classe) in base_completa_teste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

matriz = ConfusionMatrix(esperado, previsto)
print(matriz)

teste = 'Viagens são bem simples e baratas'
print(teste)
testeStemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras) in teste.split():
    comStem = [p for p in palavras.split()]
    testeStemming.append(str(stemmer.stem(comStem[0])))

novo = ferramentas.extrair_palavras(testeStemming)

distribuição = classificador.prob_classify(novo)
for classe in distribuição.samples():
    print("%s: %f" % (classe, distribuição.prob(classe)))

print()

teste = 'Amores vem e vão, mas o que fica são as lembranças'
print(teste)
testeStemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras) in teste.split():
    comStem = [p for p in palavras.split()]
    testeStemming.append(str(stemmer.stem(comStem[0])))

novo = ferramentas.extrair_palavras(testeStemming)

distribuição = classificador.prob_classify(novo)
for classe in distribuição.samples():
    print("%s: %f" % (classe, distribuição.prob(classe)))

print()

teste = 'Em uma manhã ensolarada de domingo, reuni meus amigos em um café à beira-mar para celebrar meu aniversário.'
print(teste)
testeStemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras) in teste.split():
    comStem = [p for p in palavras.split()]
    testeStemming.append(str(stemmer.stem(comStem[0])))

novo = ferramentas.extrair_palavras(testeStemming)

distribuição = classificador.prob_classify(novo)
for classe in distribuição.samples():
    print("%s: %f" % (classe, distribuição.prob(classe)))
