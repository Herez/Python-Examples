Modelo de regressão capaz de prever o consumo de energia
=============

A base de dados fornecida contém medições históricas de consumo de energia de uma indústria, acompanhadas de diversas variáveis obtidas por sensores instalados no ambiente industrial e no sistema produtivo. O objetivo é construir um modelo de regressão capaz de prever o consumo de energia a partir dessas leituras de sensores, de forma a apoiar análise de eficiência energética, planejamento de produção e tomada de decisão operacional.

Contexto do problema
=============

A indústria registra, em intervalos regulares de tempo, o consumo total de energia e diferentes grandezas físicas associadas ao processo produtivo e às condições ambientais. Essas informações podem ser usadas para entender como o consumo varia ao longo do tempo e em função do nível de produção, das condições ambientais e de outras características operacionais. Espera-se que o participante desenvolva um modelo de regressão que, a partir das variáveis explicativas disponíveis, estime o consumo de energia para novas situações de operação.

dados_consumo_energia.csv
=============

A base contém uma coluna de consumo de energia (variável alvo) e múltiplas colunas de sensores (variáveis explicativas). Em geral, são incluídas medições relacionadas a:

* Nível de atividade produtiva (por exemplo, velocidade de máquinas ou taxa de produção).
* Condições ambientais (por exemplo, temperatura e umidade).
* Condições de operação do sistema (por exemplo, pressão em determinadas linhas, horários de operação e dia da semana).
* Outros sensores auxiliares, que podem ou não ter relação direta com o consumo de energia.

A variável de interesse para previsão é o consumo de energia em cada instante (por exemplo, em kWh), enquanto todas as demais variáveis numéricas e categóricas associadas às leituras de sensores podem ser utilizadas como preditoras no modelo de regressão.


linear regression.py
=============
Define a variável resposta como o consumo de energia e selecionar as demais variáveis como candidatas a preditoras.
Prepara os dados para modelagem, incluindo eventuais transformações necessárias (tratamento de valores extremos, codificação de variáveis categóricas, padronização ou normalização, quando adequado).
Modelo de regressão capaz de prever o consumo de energia com desempenho ótimo.
