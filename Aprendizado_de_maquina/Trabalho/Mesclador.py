from os import system
import pickle
import Aprendizado as apCS
import AprendizadoSemStemming as apSS
import Dados as dB
import Ferramentas as ferramentas
    
def gerador_De_Classificadores():
    for i in range(1, 11):
        treinamento, teste = ferramentas.separar_dados(str(i))
        treinamentoCS = apCS.tratamento_Treinamento(treinamento)
        testeCS = apCS.tratamento_Teste(teste)
        treinamentoSS = apSS.tratamento_Treinamento_SemStemming(treinamento)
        testeSS = apSS.tratamento_Teste_SemStemming(teste)
        classificadorCS = apCS.classificador(treinamentoCS)
        classificadorSS = apSS.classificador_SemStemming(treinamentoSS)

        # Salvando classificador com stemming na pasta Classificadores
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'wb') as f:
            pickle.dump(classificadorCS, f)

        # Salvando classificador sem stemming na pasta Classificadores
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'wb') as f:
            pickle.dump(classificadorSS, f)

        print(f"Classificadores {i} gerados e salvos com sucesso.")

    print("Processo de geração de bases e classificadores concluído.")
    print()

def analiseErros():
    erros_CS = []
    erros_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_Teste(str(i))
            testeCS = apCS.tratamento_Teste(testeCS)
            erros_CS.append(apCS.errosTotais(classificadorCS, testeCS))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_Teste(str(i))
            testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
            erros_SS.append(apSS.errosTotais_SemStemming(classificadorSS, testeSS))
    
    return erros_CS, erros_SS

def analiseAcuracia():
    acuracia_CS = []
    acuracia_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_Teste(str(i))
            testeCS = apCS.tratamento_Teste(testeCS)
            acuracia_CS.append(round(apCS.calcular_acuracia(classificadorCS, testeCS), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_Teste(str(i))
            testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
            acuracia_SS.append(round(apSS.calcular_acuracia_SemStemming(classificadorSS, testeSS), 3))
    
    return acuracia_CS, acuracia_SS

def analisePrecisão(tag):
    precisao_CS = []
    precisao_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_Teste(str(i))
            testeCS = apCS.tratamento_Teste(testeCS)
            matrizCS = apCS.matrizConfusao(classificadorCS, testeCS)
            precisao_CS.append(round(apCS.calcular_precisao(matrizCS, tag), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_Teste(str(i))
            testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
            matrizSS = apSS.matriz_Confusao_SemStemming(classificadorSS, testeSS)
            precisao_SS.append(round(apSS.calcular_precisao_SemStemming(matrizSS, tag), 3))
    
    return precisao_CS, precisao_SS

def analiseRecall(tag):
    recall_CS = []
    recall_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_Teste(str(i))
            testeCS = apCS.tratamento_Teste(testeCS)
            matrizCS = apCS.matrizConfusao(classificadorCS, testeCS)
            recall_CS.append(round(apCS.calcular_recall(matrizCS, tag), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_Teste(str(i))
            testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
            matrizSS = apSS.matriz_Confusao_SemStemming(classificadorSS, testeSS)
            recall_SS.append(round(apSS.calcular_recall_SemStemming(matrizSS, tag), 3))
    
    return recall_CS, recall_SS

def analiseF1(tag):
    f1_CS = []
    f1_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_Teste(str(i))
            testeCS = apCS.tratamento_Teste(testeCS)
            matrizCS = apCS.matrizConfusao(classificadorCS, testeCS)
            f1_CS.append(round(apCS.calcular_f1(matrizCS, tag), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_Teste(str(i))
            testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
            matrizSS = apSS.matriz_Confusao_SemStemming(classificadorSS, testeSS)
            f1_SS.append(round(apSS.calcular_f1_SemStemming(matrizSS, tag), 3))
    
    return f1_CS, f1_SS

def geradorRelatorio(selecao):
    with open(f'Classificadores/classificador_com_stemming_{selecao}.pkl', 'rb') as f:
        classificadorCS = pickle.load(f)
        testeCS = dB.abrir_arquivo_Teste(str(selecao))
        testeCS = apCS.tratamento_Teste(testeCS)
        print("Com stemming:")
        apCS.relatorio(classificadorCS, testeCS)

    with open(f'Classificadores/classificador_sem_stemming_{selecao}.pkl', 'rb') as f:
        classificadorSS = pickle.load(f)
        testeSS = dB.abrir_arquivo_Teste(str(selecao))
        testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
        print("Sem stemming:")
        apSS.relatorio_SemStemming(classificadorSS, testeSS)

def geradorMatriz(selecao):
    with open(f'Classificadores/classificador_com_stemming_{selecao}.pkl', 'rb') as f:
        classificadorCS = pickle.load(f)
        testeCS = dB.abrir_arquivo_Teste(str(selecao))
        testeCS = apCS.tratamento_Teste(testeCS)
        print("Com stemming:")
        matrizCS = apCS.matrizConfusao(classificadorCS, testeCS)
        print(matrizCS)

    with open(f'Classificadores/classificador_sem_stemming_{selecao}.pkl', 'rb') as f:
        classificadorSS = pickle.load(f)
        testeSS = dB.abrir_arquivo_Teste(str(selecao))
        testeSS = apSS.tratamento_Teste_SemStemming(testeSS)
        print("Sem stemming:")
        matrizSS = apSS.matriz_Confusao_SemStemming(classificadorSS, testeSS)
        print(matrizSS)

def analizador_Comparador():
    frases = []
    quantidade = 0
    print("Digite as frases que deseja analisar:")
    quantidade = int(input())
    for i in range(0, quantidade):
        print(f"Digite a frase {i+1}:")
        frases.append(input())
    print()
    for i in frases:
        print(f"Analisando a frase: {i}")
        print("Com stemming:\n")
        for j in range(1, 11):
            with open(f'Classificadores/classificador_com_stemming_{j}.pkl', 'rb') as f:
                classificadorCS = pickle.load(f)
                apCS.AnalisadorManual(classificadorCS, i)
        print("Sem stemming:\n")
        for j in range(1, 11):
            with open(f'Classificadores/classificador_sem_stemming_{j}.pkl', 'rb') as f:
                classificadorSS = pickle.load(f)
                apSS.AnalisadorManual_SemStemming(classificadorSS, i)












