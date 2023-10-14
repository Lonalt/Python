from os import system
import pickle
import Aprendizado as apCS
import AprendizadoSemStemming as apSS
import Dados as dB
import Ferramentas as ferramentas

def gerador_De_Bases():
    for i in range(1, 11):
        treinamento, teste = ferramentas.separar_dados(str(i))
        dB.gerar_csv(treinamento, "treinamento", str(i))
        dB.gerar_csv(teste, "teste", str(i))

def gerar_Classificadores(indice):
    treinamento = dB.abrir_arquivo_Treinamento(str(indice))
    tCS = apCS.tratamento(treinamento)
    tSS = apSS.tratamento(treinamento)
    classificadorCS = apCS.classificador(tCS)
    classificadorSS = apSS.classificador(tSS)

    # Salvando classificador com stemming
    with open(f'classificador_com_stemming_{indice}.pkl', 'wb') as f:
        pickle.dump(classificadorCS, f)

    # Salvando classificador sem stemming
    with open(f'classificador_sem_stemming_{indice}.pkl', 'wb') as f:
        pickle.dump(classificadorSS, f)

    return classificadorCS, classificadorSS

system("clear")  
gerador_De_Bases()  



# treinamento, teste = ferramentas.gerarBase(str(1))
# treinamentoCS, testeCS = apCS.tratamento(treinamento, teste)
# classificadorCS = apCS.classificador(treinamentoCS)
# #apCS.errosTotais(classificadorCS, testeCS)
# matrizCS = apCS.matrizConfusao(classificadorCS, testeCS)
# print(matrizCS)
# apCS.relatorio(classificadorCS, testeCS)
# apCS.calcular_acuracia(classificadorCS, testeCS)
# apCS.calcular_precisao(matrizCS)
# apCS.calcular_recall(matrizCS)
# apCS.calcular_f1(matrizCS)

# #apCS.tags(classificadorCS, 10)
# #apCS.testeAutomatico(classificadorCS)
# #apCS.AnalisadorManual(classificadorCS)

system("clear")
gerador_De_Bases()


