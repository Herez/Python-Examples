import pandas as pd             # manipulação de dados (DataFrame)
import numpy as np              # operações numéricas
import matplotlib.pyplot as plt # plotagem dos gráficos
import seaborn as sns           # visualizações estatísticas

from sklearn.model_selection import train_test_split                          # Função para dividir dados em treino/teste
from sklearn.linear_model import LinearRegression                             # Modelo de regressão linear
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Métricas de avaliação: MAE, MSE/RMSE, R²
from sklearn.preprocessing import StandardScaler, OneHotEncoder               # Pré-processamento: padronização e codificação categórica
from sklearn.compose import ColumnTransformer                                 # Aplicar transformações diferentes por coluna
from sklearn.pipeline import Pipeline                                         # Encadear pré-processamento e modelo

# =============================================================================
sns.set(style="darkgrid") # white dark whitegrid darkgrid ticks
# =============================================================================
df = pd.read_csv("/root/spot/Downloads/Python/dados_consumo_energia.csv", sep=",") # Lê o CSV
# =============================================================================

# =============================================================================
# Normalizar separador decimal e tentar converter tipos
# =============================================================================
for col in df.columns: #Para Cada Coluna dentra da coleção de Colunas do DataFrame
    # Converte tudo para string e substitui vírgula por ponto (caso números venham com vírgula)
    df[col] = df[col].astype(str).str.replace(",", ".")
    # Tenta converter a coluna para numérico; se falhar, mantém como string (errors="ignore")
    df[col] = pd.to_numeric(df[col], errors="ignore")

# =============================================================================
# Remover colunas irrelevantes
# =============================================================================
cols_to_drop = []
if "Data_Hora" in df.columns:
    cols_to_drop.append("Data_Hora")

# =============================================================================
# Definir variável alvo
# =============================================================================
target_col = "Consumo_Energia_kWh"
if target_col in cols_to_drop: # Garante que a coluna alvo não seja acidentalmente removida.
    cols_to_drop.remove(target_col)

# =============================================================================
# Preparar X (explicativas) e y (alvo)
# =============================================================================
y = df[target_col]                                # Série com o alvo (consumo)
X = df.drop(columns=[target_col] + cols_to_drop)  # DataFrame com as explicativas (remove alvo e colunas irrelevantes)

# =============================================================================
# Identificar colunas categóricas e numéricas
# =============================================================================
categorical_features = X.select_dtypes(include="object").columns.tolist()
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# =============================================================================
# Pré-processamento com ColumnTransformer
# =============================================================================
# Aplica transformações às colunas de um DataFrame do pandas. 
# Permite que diferentes colunas ou subconjuntos de colunas da entrada sejam transformados separadamente, 
# e as características geradas por cada transformação serão concatenadas para formar um único espaço 
# de características. Isso é útil para dados heterogêneos, para combinar vários mecanismos de extração 
# de características ou transformações em uma única transformação.
# =============================================================================
# Quando aplicamos o StandardScaler aos dados, ele faz uso da distribuição normal e, portanto, 
# transforma cada valor nos dados de forma que a média dos dados padronizados 
# seja zero e o desvio padrão seja igual a um. 
# Essa transformação garante que os dados estejam em uma escala comum, 
# o que é útil para muitos algoritmos de aprendizado de máquina, 
# especialmente aqueles que são sensíveis à escala dos atributos.
# =============================================================================
# OneHotEncoder é uma técnica de pré-processamento em Machine Learning que converte variáveis categóricas 
# em um formato numérico binário (zeros e uns), criando novas colunas para cada categoria única, 
# onde apenas uma coluna recebe '1' (indicando a presença daquela categoria) e as outras recebem '0'. 
# Isso é essencial para algoritmos que exigem entradas numéricas, 
# permitindo que eles interpretem dados textuais ou discretos
# =============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features), # Aplica StandardScaler às colunas numéricas (média 0, desvio 1)
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features) # Aplica OneHotEncoder às categóricas; handle_unknown evita erro com categorias novas
    ],
    remainder="drop"  # Descarta colunas não listadas explicitamente
)

# =============================================================================
# Pipeline: pré-processamento + regressão linear
# =============================================================================
# Pipeline facilita validação cruzada e ajuda na integridade dos dados
model = Pipeline(steps=[
    ("preprocessor", preprocessor),   # Primeiro transforma os dados
    ("regressor", LinearRegression()) # Depois ajusta o modelo linear
])

# =============================================================================
# Divisão treino/teste
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% dos dados para teste; random_state para reprodutibilidade

# =============================================================================
# Treinar modelo
# =============================================================================
model.fit(X_train, y_train) # Ajusta o pipeline: aprende parâmetros do StandardScaler, OneHotEncoder e coeficientes da regressão

# =============================================================================
# Previsões e avaliação
# =============================================================================
y_pred = model.predict(X_test)                     # Previsões no conjunto de teste
mae = mean_absolute_error(y_test, y_pred)          # Erro absoluto médio
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Raiz do erro quadrático médio
r2 = r2_score(y_test, y_pred)                      # Coeficiente de determinação

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# =============================================================================
# Visualizações
# =============================================================================
# 1) Predito vs Real
# =============================================================================
plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
# Plota pontos: eixo x = valor real, eixo y = valor predito
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, '--', color='red', label='y = x')
# Linha y=x serve como referência: pontos próximos indicam boa previsão
plt.xlabel("Consumo real (kWh)")
plt.ylabel("Consumo predito (kWh)")
plt.title("Predito vs Real")
plt.legend()
plt.tight_layout()
plt.show()
# =============================================================================
# 2) Resíduos vs Predito
# =============================================================================
residuals = y_test - y_pred
# Resíduo definido como (real - predito); análise de resíduos ajuda a detectar viés e heterocedasticidade
# onde a variância dos erros (resíduos) não é constante em todos os níveis das variáveis independentes
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')  # Linha horizontal em zero para referência
plt.xlabel("Consumo predito (kWh)")
plt.ylabel("Resíduo (real - predito)")
plt.title("Resíduos vs Predito")
plt.tight_layout()
plt.show()
