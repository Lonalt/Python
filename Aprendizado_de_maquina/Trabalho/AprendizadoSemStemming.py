import Dados as dadosBrutos
import Ferramentas as ferramentas
import nltk
from nltk.metrics import ConfusionMatrix

def tratamento_SemStemming(treinamento, teste):
    stopWords = ferramentas.criar_stopwords()
    frases_treinamento = treinamento
    frases_teste = teste
    Palavras_treinamento = dadosBrutos.coletar_palavras(frases_treinamento)
    Palavras_teste = dadosBrutos.coletar_palavras(frases_teste)
    frequencia_treinamento = ferramentas.frequencia_palavras(Palavras_treinamento)
    frequencia_teste = ferramentas.frequencia_palavras(Palavras_teste)
    base_completa_treinamento = nltk.classify.apply_features(ferramentas.extrair_palavras, frases_treinamento)
    base_completa_teste = nltk.classify.apply_features(ferramentas.extrair_palavras_teste, frases_teste)
    return base_completa_treinamento, base_completa_teste

def classificador_SemStemming(base_completa_treinamento):
    classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)
    return classificador

def errosTotais_SemStemming(classificador, base_completa_teste):
    erros = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        if resultado != classe:
            erros.append((classe, resultado, frase))
    print('Total de erros:', len(erros))
    print()

def calcular_acuracia_SemStemming(classificador, base_completa_teste):
    print("Acurácia {:.2}".format(nltk.classify.accuracy(classificador, base_completa_teste)))

def calcular_precisao_SemStemming(matriz):
    print("Precisão Feliz {:.2}".format(ConfusionMatrix.precision(matriz, 'feliz')))
    print("Precisão Triste {:.2}".format(ConfusionMatrix.precision(matriz, 'triste')))
    print("Precisão Neutro {:.2}".format(ConfusionMatrix.precision(matriz, 'neutro')))

def calcular_recall_SemStemming(matriz):
    print("Recall Feliz {:.2}".format(ConfusionMatrix.recall(matriz, 'feliz')))
    print("Recall Triste {:.2}".format(ConfusionMatrix.recall(matriz, 'triste')))
    print("Recall Neutro {:.2}".format(ConfusionMatrix.recall(matriz, 'neutro')))

def calcular_f1_SemStemming(matriz):
    print("F1 Feliz {:.2}".format(ConfusionMatrix.f_measure(matriz, 'feliz')))
    print("F1 Triste {:.2}".format(ConfusionMatrix.f_measure(matriz, 'triste')))
    print("F1 Neutro {:.2}".format(ConfusionMatrix.f_measure(matriz, 'neutro')))

def relatorio_SemStemming(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    print(classification_report(esperado, previsto, target_names=['feliz', 'triste', 'neutro']))

def matriz_Confusao_SemStemming(classificador, base_completa_teste):
    esperado = []
    previsto = []
    for (frase, classe) in base_completa_teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)
    matriz = ConfusionMatrix(esperado, previsto)
    return matriz

def tags_SemStemming(classificador, solicitado):
    print(classificador.labels())
    print(classificador.show_most_informative_features(solicitado))
    print()

def testeAutomatico_SemStemming(classificador):
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
            print("{}: {:.5}".format(classe, distribuição.prob(classe)))
        print()

def AnalisadorManual_SemStemming(classificador):
    print('Digite a quantidade de testes que deseja realizar:')
    solicitado = int(input())
    for i in range(solicitado):
        print('Digite a frase que deseja testar:')
        frase = input()
        novo = ferramentas.extrair_palavras(frase.split())
        distribuição = classificador.prob_classify(novo)
        for classe in distribuição.samples():
            print("{}: {:.5}".format(classe, distribuição.prob(classe)))
        print()


