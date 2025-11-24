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
#### 5. One Hot Encoder
#### 6. Validação de correlação
<img width="1046" height="805" alt="image" src="https://github.com/user-attachments/assets/9383cf3b-4e03-4119-97e8-b53d3c8a30bc" />

#### 7. Distribuição das variáveis numéricas
#### 8. Determinação do target (Variável alvo)
#### 9. Apresentação da correlação dos dados com o target em tabela
<img width="986" height="590" alt="image" src="https://github.com/user-attachments/assets/47ade301-9e77-43a9-b088-64f10fc53f97" />

#### 10. Determinando a variável preditora
#### 11. Divisão entre treino e teste
#### 12. Criação e treinamento
#### 13. Previsão
#### 13. Validação do resultado
#### 13. Exibição dos resultados do teste e previsão.
#### 14. Visualização da regressão em um gráfico
<img width="876" height="624" alt="image" src="https://github.com/user-attachments/assets/30f7da5d-5d67-4e6c-916f-22e6c4a9aaa4" />

