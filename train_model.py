# Importação de Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

# Carregar os Dados
df = pd.read_csv('./data/carros.csv')

# Pré-processamento dos Dados
df = df.dropna()  # Removendo valores nulos
df.reset_index(drop=True, inplace=True)  # Resetando índices
df['MSRP'] = df['MSRP'].map(lambda x: x.lstrip('$').replace(',', ''))  # Limpando valores monetários
df['MSRP'] = pd.to_numeric(df['MSRP'])  # Convertendo para número
df = pd.get_dummies(df, columns=['Make', 'Type', 'DriveTrain'])  # One-hot encoding

# Separação de X e y
X = df.drop('MSRP', axis=1)
y = df['MSRP']

# Divisão em Treino e Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# Treinamento do Modelo
modelo = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    min_samples_leaf=3,
    max_features=0.1,
    loss='lad',  # Loss para robustez a outliers
    random_state=0
)
modelo.fit(X_treino, y_treino)

# Avaliação do Modelo
previsoes = modelo.predict(X_teste)
r2 = r2_score(y_teste, previsoes)
print(f"O R2 Score do Modelo é: {r2 * 100:.2f}")

# Salvando o Modelo
joblib.dump(modelo, './modelo/modelo.pkl')
print("Modelo salvo com sucesso em './modelo/modelo.pkl'")
