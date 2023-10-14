from os import system
import pickle
import Aprendizado as apCS
import AprendizadoSemStemming as apSS
import Dados as dB
import Ferramentas as ferramentas
    
def gerador_de_classificadores():
    for i in range(1, 11):
        treinamento, teste = ferramentas.separar_dados(str(i))
        treinamentoCS = apCS.tratamento_treinamento(treinamento)
        testeCS = apCS.tratamento_teste(teste)
        treinamentoSS = apSS.tratamento_treinamento_sem_stemming(treinamento)
        testeSS = apSS.tratamento_teste_sem_stemming(teste)
        classificadorCS = apCS.classificador(treinamentoCS)
        classificadorSS = apSS.classificador_sem_stemming(treinamentoSS)

        # Salvando classificador com stemming na pasta Classificadores
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'wb') as f:
            pickle.dump(classificadorCS, f)

        # Salvando classificador sem stemming na pasta Classificadores
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'wb') as f:
            pickle.dump(classificadorSS, f)

        print(f"Classificadores {i} gerados e salvos com sucesso.")

    print("Processo de geração de bases e classificadores concluído.")
    print()
    input("Pressione ENTER para continuar...")

def analise_erros():
    erros_CS = []
    erros_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_teste(str(i))
            testeCS = apCS.tratamento_teste(testeCS)
            erros_CS.append(apCS.erros_totais(classificadorCS, testeCS))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_teste(str(i))
            testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
            erros_SS.append(apSS.erros_totais_sem_stemming(classificadorSS, testeSS))
    
    return erros_CS, erros_SS

def analise_acuracia():
    acuracia_CS = []
    acuracia_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_teste(str(i))
            testeCS = apCS.tratamento_teste(testeCS)
            acuracia_CS.append(round(apCS.calcular_acuracia(classificadorCS, testeCS), 5))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_teste(str(i))
            testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
            acuracia_SS.append(round(apSS.calcular_acuracia_sem_stemming(classificadorSS, testeSS), 5))
    
    return acuracia_CS, acuracia_SS

def analise_precisão(tag):
    precisao_CS = []
    precisao_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_teste(str(i))
            testeCS = apCS.tratamento_teste(testeCS)
            matrizCS = apCS.matriz_confusao(classificadorCS, testeCS)
            precisao_CS.append(round(apCS.calcular_precisao(matrizCS, tag), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_teste(str(i))
            testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
            matrizSS = apSS.matriz_confusao_sem_stemming(classificadorSS, testeSS)
            precisao_SS.append(round(apSS.calcular_precisao_sem_stemming(matrizSS, tag), 3))
    
    return precisao_CS, precisao_SS

def analise_recall(tag):
    recall_CS = []
    recall_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_teste(str(i))
            testeCS = apCS.tratamento_teste(testeCS)
            matrizCS = apCS.matriz_confusao(classificadorCS, testeCS)
            recall_CS.append(round(apCS.calcular_recall(matrizCS, tag), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_teste(str(i))
            testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
            matrizSS = apSS.matriz_confusao_sem_stemming(classificadorSS, testeSS)
            recall_SS.append(round(apSS.calcular_recall_sem_stemming(matrizSS, tag), 3))
    
    return recall_CS, recall_SS

def analise_f1(tag):
    f1_CS = []
    f1_SS = []
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            testeCS = dB.abrir_arquivo_teste(str(i))
            testeCS = apCS.tratamento_teste(testeCS)
            matrizCS = apCS.matriz_confusao(classificadorCS, testeCS)
            f1_CS.append(round(apCS.calcular_f1(matrizCS, tag), 3))

    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            testeSS = dB.abrir_arquivo_teste(str(i))
            testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
            matrizSS = apSS.matriz_confusao_sem_stemming(classificadorSS, testeSS)
            f1_SS.append(round(apSS.calcular_f1_sem_stemming(matrizSS, tag), 3))
    
    return f1_CS, f1_SS

def gerador_relatorio(selecao):
    with open(f'Classificadores/classificador_com_stemming_{selecao}.pkl', 'rb') as f:
        classificadorCS = pickle.load(f)
        testeCS = dB.abrir_arquivo_teste(str(selecao))
        testeCS = apCS.tratamento_teste(testeCS)
        print("Com stemming:")
        apCS.relatorio(classificadorCS, testeCS)

    with open(f'Classificadores/classificador_sem_stemming_{selecao}.pkl', 'rb') as f:
        classificadorSS = pickle.load(f)
        testeSS = dB.abrir_arquivo_teste(str(selecao))
        testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
        print("Sem stemming:")
        apSS.relatorio_sem_stemming(classificadorSS, testeSS)

def gerador_matriz(selecao):
    with open(f'Classificadores/classificador_com_stemming_{selecao}.pkl', 'rb') as f:
        classificadorCS = pickle.load(f)
        testeCS = dB.abrir_arquivo_teste(str(selecao))
        testeCS = apCS.tratamento_teste(testeCS)
        print("Com stemming:")
        matrizCS = apCS.matriz_confusao(classificadorCS, testeCS)
        print(matrizCS)

    with open(f'Classificadores/classificador_sem_stemming_{selecao}.pkl', 'rb') as f:
        classificadorSS = pickle.load(f)
        testeSS = dB.abrir_arquivo_teste(str(selecao))
        testeSS = apSS.tratamento_teste_sem_stemming(testeSS)
        print("Sem stemming:")
        matrizSS = apSS.matriz_confusao_sem_stemming(classificadorSS, testeSS)
        print(matrizSS)

def analizador_comparador():
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
            print(f"Divisão {j}:")
            with open(f'Classificadores/classificador_com_stemming_{j}.pkl', 'rb') as f:
                classificadorCS = pickle.load(f)
                apCS.analisador_manual(classificadorCS, i)
        print("Sem stemming:\n")
        for j in range(1, 11):
            print(f"Divisão {j}:")
            with open(f'Classificadores/classificador_sem_stemming_{j}.pkl', 'rb') as f:
                classificadorSS = pickle.load(f)
                apSS.analisador_manual_sem_stemming(classificadorSS, i)

def analizador_comparador_automatico():
    for i in range(1, 11):
        with open(f'Classificadores/classificador_com_stemming_{i}.pkl', 'rb') as f:
            classificadorCS = pickle.load(f)
            apCS.teste_automatico(classificadorCS)
    print("Sem stemming:\n")
    for i in range(1, 11):
        with open(f'Classificadores/classificador_sem_stemming_{i}.pkl', 'rb') as f:
            classificadorSS = pickle.load(f)
            apSS.teste_automatico_sem_stemming(classificadorSS)













