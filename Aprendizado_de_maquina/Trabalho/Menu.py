import os
import Aprendizado as apCS
import AprendizadoSemStemming as apSS
import Dados as dB
import Ferramentas as ferramentas
import Mesclador as mesclador

class Menu:
    """Menu interativo para manipulação de dados e classificadores de análise de sentimentos."""

    def __init__(self):
        self.opcao = 0

    def menu_principal(self):
        os.system('clear')
        print("---- Menu Principal ----")
        print("\t01. Gerar classificadores")
        print("\t02. Analisar erros")
        print("\t03. Analisar acurácia")
        print("\t04. Analisar precisão")
        print("\t05. Analisar recall")
        print("\t06. Analisar F1")
        print("\t07. Gerar relatório")
        print("\t08. Gerar matriz de confusão")
        print("\t09. Analisar frases")
        print("\t10. Sair")
        print()
        self.opcao = int(input("Digite sua opção: "))

    def executar_opcao(self):
        while self.opcao != 10:
            if self.opcao == 1:
                mesclador.gerador_de_classificadores()
            elif self.opcao == 2:
                os.system('clear')
                erros_CS, erros_SS = mesclador.analise_erros()

                print("\n\t\tErros com stemming:\n")
                print(erros_CS)
                media_CS = ferramentas.media_vetor(erros_CS)
                print(f"\n\tMédia de erros com stemming: {media_CS}")
                dp = ferramentas.desvio_padrao(erros_CS)
                print(f"\tDesvio padrão: {dp}")

                print("\n\t\tErros sem stemming:\n")
                print(erros_SS)
                media_SS = ferramentas.media_vetor(erros_SS)
                print(f"\n\tMédia de erros sem stemming: {media_SS}")
                dp = ferramentas.desvio_padrao(erros_SS)
                print(f"\tDesvio padrão: {dp}")

                input("\nPressione ENTER para continuar...")
            elif self.opcao == 3:
                os.system('clear') 
                acuracia_CS, acuracia_SS = mesclador.analise_acuracia()

                print("\n\t\tAcurácia com stemming:\n")
                print(acuracia_CS)
                media_CS = ferramentas.media_vetor(acuracia_CS)
                print(f"\n\tMédia de acurácia com stemming: {media_CS}")
                dp = ferramentas.desvio_padrao(acuracia_CS)
                print(f"\tDesvio padrão: {dp}")

                print("\n\t\tAcurácia sem stemming:\n")
                print(acuracia_SS)
                media_SS = ferramentas.media_vetor(acuracia_SS)
                print(f"\n\tMédia de acurácia sem stemming: {media_SS}")
                dp = ferramentas.desvio_padrao(acuracia_SS)
                print(f"\tDesvio padrão: {dp}")

                teste_t = ferramentas.teste_t(acuracia_CS, acuracia_SS)

                input("\nPressione ENTER para continuar...")
            elif self.opcao == 4:
                os.system('clear')
                tag = input("Digite a tag a ser analisada: ")
                precisao_CS, precisao_SS = mesclador.analise_precisão(tag)

                print(f"\n\t\tPrecisão com stemming para o tag {tag}:\n")
                print(precisao_CS)
                media_CS = ferramentas.media_vetor(precisao_CS)
                print(f"\n\tMédia de precisão com stemming: {media_CS}")
                dp = ferramentas.desvio_padrao(precisao_CS)
                print(f"\tDesvio padrão: {dp}")

                print(f"\n\t\tPrecisão sem stemming para o tag {tag}:\n")
                print(precisao_SS)
                media_SS = ferramentas.media_vetor(precisao_SS)
                print(f"\n\tMédia de precisão sem stemming: {media_SS}")
                dp = ferramentas.desvio_padrao(precisao_SS)
                print(f"\tDesvio padrão: {dp}")

                teste_t = ferramentas.teste_t(precisao_CS, precisao_SS)

                input("\nPressione ENTER para continuar...")
            elif self.opcao == 5:
                os.system('clear')
                tag = input("Digite a tag a ser analisada: ")
                recall_CS, recall_SS = mesclador.analise_recall(tag)

                print(f"\n\t\tRecall com stemming para o tag {tag}:")
                print(recall_CS)
                media_CS = ferramentas.media_vetor(recall_CS)
                print(f"\n\tMédia de recall com stemming: {media_CS}")
                dp = ferramentas.desvio_padrao(recall_CS)
                print(f"\tDesvio padrão: {dp}")    

                print(f"\n\t\tRecall sem stemming para o tag {tag}:")
                print(recall_SS)
                media_SS = ferramentas.media_vetor(recall_SS)
                print(f"\n\tMédia de recall sem stemming: {media_SS}")
                dp = ferramentas.desvio_padrao(recall_SS)
                print(f"\tDesvio padrão: {dp}")

                teste_t = ferramentas.teste_t(recall_CS, recall_SS)

                input("\nPressione ENTER para continuar...")
            elif self.opcao == 6:
                os.system('clear')
                tag = input("Digite a tag a ser analisada: ")
                f1_CS, f1_SS = mesclador.analise_f1(tag)

                print(f"\n\t\tF1 com stemming para o tag {tag}:")
                print(f1_CS)
                media_CS = ferramentas.media_vetor(f1_CS)
                print(f"\n\tMédia de F1 com stemming: {media_CS}")
                dp = ferramentas.desvio_padrao(f1_CS)
                print(f"\tDesvio padrão: {dp}")

                print(f"\n\t\tF1 sem stemming para o tag {tag}:")
                print(f1_SS)
                media_SS = ferramentas.media_vetor(f1_SS)
                print(f"\n\tMédia de F1 sem stemming: {media_SS}")
                dp = ferramentas.desvio_padrao(f1_SS)
                print(f"\tDesvio padrão: {dp}")

                teste_t = ferramentas.teste_t(f1_CS, f1_SS)

                input("\nPressione ENTER para continuar...")
            elif self.opcao == 7:
                os.system('clear')
                selecao = int(input("Digite a divisão de treino/teste a ser analisada: "))
                mesclador.gerador_relatorio(selecao)
                input("\nPressione ENTER para continuar...")
            elif self.opcao == 8:
                os.system('clear')
                selecao = int(input("Digite a divisão de treino/teste a ser analisada: "))
                mesclador.gerador_matriz(selecao)
                input("\nPressione ENTER para continuar...")
            elif self.opcao == 9:
                os.system('clear')
                mesclador.analizador_comparador()
                input("\nPressione ENTER para continuar...")
            else:
                print("\nOpção inválida!")
                input("\n\nPressione ENTER para continuar...")
            self.menu_principal()


            
                   
if __name__ == "__main__":
    menu = Menu()
    menu.menu_principal()
    menu.executar_opcao() 
 
