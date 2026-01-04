import os # Importa o módulo para interagir com o sistema operacional

# 1. Define a string a ser adicionada e o diretório
STRING_PREFIXO = "Herez "
DIRETORIO = "/root/Downloads/" # Altere para o caminho da sua pasta

# Verifica se o diretório existe
if not os.path.exists(DIRETORIO):
    print(f"Diretório '{DIRETORIO}' não encontrado.")
else:
    print(f"Processando arquivos em: {DIRETORIO}")
    # 2. Lista todos os arquivos no diretório
    for nome_arquivo in os.listdir(DIRETORIO):
        # 3. Verifica se é um arquivo e se termina com .txt
        if nome_arquivo.endswith(".pdf") and os.path.isfile(os.path.join(DIRETORIO, nome_arquivo)):
            # 4. Cria o novo nome do arquivo usando f-string para concatenar
            novo_nome = f"{STRING_PREFIXO}{nome_arquivo}"

            # 5. Define os caminhos completos (importante para renomear)
            caminho_antigo = os.path.join(DIRETORIO, nome_arquivo)
            caminho_novo = os.path.join(DIRETORIO, novo_nome)

            # 6. Renomeia o arquivo
            try:
                os.rename(caminho_antigo, caminho_novo)
                print(f"Renomeado: '{nome_arquivo}' -> '{novo_nome}'")
            except Exception as e:
                print(f"Erro ao renomear {nome_arquivo}: {e}")

    print("Processo concluído.")

