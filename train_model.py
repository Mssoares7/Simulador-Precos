# Imports
import joblib
import sklearn
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Carregar os Dados
df = pd.read_csv('./data/carros.csv')

# Pré-processamento dos Dados
df = df.dropna()  # Removendo valores nulos
# Reset do índice
df.reset_index(drop=True)
df = df.drop(['Model', 'Origin', 'Invoice'], axis=1)  # Remove colunas irrelevantes

# Ajustamos a coluna target
df['MSRP'] = df['MSRP'].map(lambda x: x.lstrip('$').replace(',',''))
# Convertemos para tipo numérico
df['MSRP'] = pd.to_numeric(df['MSRP'])

# One-hot encoding
df = pd.get_dummies(df, columns = ['Make', 'Type', 'DriveTrain'])

# Separa x e y
X = df.drop('MSRP', axis = 1)
y = df['MSRP']
X = X.to_numpy()
y = y.to_numpy()


# Divisão dos dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Cria o modelo
modelo = GradientBoostingRegressor(n_estimators = 5000, 
                                   learning_rate = 0.1,
                                   max_depth = 10,
                                   min_samples_leaf = 3,
                                   max_features = 0.1,
                                   loss = 'lad',
                                   random_state = 0)

# Treinamento do modelo
modelo.fit(X_treino, y_treino)

# Previsões
previsoes = modelo.predict(X_teste)
print("O R2 Score do Modelo é:", r2_score(y_teste, previsoes) * 100)

# Salvando o Modelo
joblib.dump(modelo, './modelo/modelo.pkl')
print("Modelo salvo com sucesso em './modelo/modelo.pkl'")
