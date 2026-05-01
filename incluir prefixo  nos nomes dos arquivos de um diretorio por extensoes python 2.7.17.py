import os # interagir com o sistema operacional

# 1. Define prefixo do arquivo e local da pasta dos arquivos
STRING_PREFIXO = "Herez - "
DIRETORIO = "/root/spot/NewDir/" # Altere para o caminho da sua pasta

#Debug
import pdb
pdb.set_trace()

# Verifica se a pasta existe
if not os.path.exists(DIRETORIO):
    print("Pasta {DIRETORIO} nao encontrada.")
else:
    print("Processando arquivos em: " + DIRETORIO)
    # 2. Lista todos os arquivos na pasta
    for nome_arquivo in os.listdir(DIRETORIO):
        # 3. Verifica se termina com .pdf
        if nome_arquivo.endswith(".pdf") and os.path.isfile(os.path.join(DIRETORIO, nome_arquivo)):
            # 4. Cria o novo nome do arquivo 
            novo_nome = STRING_PREFIXO + nome_arquivo

            # 5. Define os caminhos completos (importante para renomear)
            caminho_antigo = os.path.join(DIRETORIO, nome_arquivo)
            caminho_novo = os.path.join(DIRETORIO, novo_nome)

            # 6. Renomeia o arquivo
            try:
                if not nome_arquivo.startswith(STRING_PREFIXO):
                    os.rename(caminho_antigo, caminho_novo)
                    print("Renomeado:" + nome_arquivo + " para -> "+ novo_nome)
            except Exception as e:
                print("Erro ao renomear " + nome_arquivo + ":" + e)

    print("Processo executado com sucesso.")
