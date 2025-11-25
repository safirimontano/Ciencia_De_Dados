# Projeto de regressão linear

#### 1. Objetivo: Prever o preço das casas na Califórnia. 
#### 2. Bibliotecas usadas: 
```Python
import kagglehub
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
```
#### 3. Obtenção dos dados:
O projeto foi realizado usando o Google Colab e o dataset estava no banco de dados do kaggle, então o processo de obtenção é realizado utilizando a biblioteca kagglehub. *path = kagglehub.dataset_download("camnugent/california-housing-prices")*.

#### 4. Analise exploratória dos dados
```Python
print("Informações da base de dados:")
print(df.info())
print("\estatistica:")
print(df.describe())
```
df.info(): Quantas linhas tem, Tipos de dados, Quantos valores nulos por coluna.
df.describe(): Estatísticas numéricas(média, desvio padrão, min / max, quartis).

#### 5. One Hot Encoder
Colunas qualitativas são transformadas em binário para auxiliar o modelo.

#### 6. Validação de correlação
Um gráfico de calor (heatmap) mostrando a correlação entre todas as variáveis numéricas da base. 
É possível visualizar, por exemplo qual variável mais se relaciona com o preço das casas (median_house_value), se existe multicolinearidade ou que atributos valem mais a pena no modelo.
<img width="1046" height="805" alt="image" src="https://github.com/user-attachments/assets/9383cf3b-4e03-4119-97e8-b53d3c8a30bc" />

#### 7. Distribuição das variáveis numéricas
Apresento um  grid de histogramas, um para cada variável numérica do df_final que demonstra como cada variável se distribui (normal, assimétrica, concentrada etc.) e é útil pra perceber outliers e entender a “personalidade” dos dados.

#### 8. Determinação do target (Variável alvo)
O target é o que vamos prever (o preço mediano das casas).

#### 9. Apresentação da correlação dos dados com o target em tabela
Aqui vemos uma série de correlações entre todas as variáveis e o preço das casas, ordenadas da mais forte positiva para a mais forte negativa.
<img width="986" height="590" alt="image" src="https://github.com/user-attachments/assets/47ade301-9e77-43a9-b088-64f10fc53f97" />

#### 10. Determinando a variável preditora
Essa parte determina o X = média de renda > variável explicativa e y = valor das casas > variável alvo.
O modelo de regressão vai tentar prever o preço da casa usando só a renda média do bairro.

#### 11. Divisão entre treino e teste
Aqui dividimos em:
X_train: 80% dos valores de renda (median_income)
X_test: 20% dos valores de renda
y_train: 80% dos preços das casas
y_test: 20% dos preços das casas
Isso aqui vai separar os dados pra evitar que o modelo “decore” tudo.

#### 12. Criação e treinamento
É apresentado aqui o coeficiente (a inclinação da reta) e o o intercepto (onde a reta cruza o eixo Y).
O cálculo é o `^​=β0​+β1​⋅median_income`

#### 13. Previsão
A previsão retorna um vetor com os valores previstos de preço das casas, usando a renda dos bairros do conjunto de teste.
Em resumo ele faz um: “Com essa renda, eu estimo que a casa valha X.”

#### 13. Validação do resultado
Usei as seguintes validações:
- MAE (Erro Absoluto Médio)
Quanto o modelo erra em média, em valores diretos.
- MSE (Erro Quadrático Médio)
Mais sensível a erros grandes — costuma parecer gigante.
- RMSE
MSE “arrumado” pra ficar na mesma escala do target. Mais interpretável.
- R²
Diz quanto da variação nos preços das casas o modelo explica.

#### 13. Exibição dos resultados do teste e previsão.
Um DataFrame com median_income > renda do bairro, target_teste > preço real da casa e target_previsto > preço previsto pelo seu modelo
Isso mostra claramente quando o modelo chegou perto, quando passou longe, e como a renda realmente influencia o valor.

#### 14. Visualização da regressão em um gráfico
<img width="876" height="624" alt="image" src="https://github.com/user-attachments/assets/30f7da5d-5d67-4e6c-916f-22e6c4a9aaa4" />

