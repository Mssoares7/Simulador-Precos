# Simulador de Preço de Carros com Machine Learning

Este projeto é um simulador que prevê o preço de veículos com base em seus atributos utilizando Machine Learning.

## Tecnologias Utilizadas:
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Render (para deploy)
- Git e GitHub

## Estrutura do Projeto:
├── app.py               # Arquivo principal da aplicação Flask
├── tools/car.py         # Classe responsável pela preparação e previsão
├── modelo/modelo.pkl    # Modelo treinado
├── data/carros.csv      # Dados de entrada
├── templates/index.html # Interface HTML
├── requirements.txt     # Dependências do projeto

Funcionalidades:
Previsão de preço de veículos com base em características como:
Tamanho do motor
Número de cilindros
Tipo de tração  (4WD, FWD, RWD)
Tipo de veículo (SUV, Sedan, etc.)
Montadora

Interface amigável desenvolvida com Flask.

Modelo treinado com Gradient Boosting Regressor.

Deploy:
A aplicação foi implantada na plataforma Render e está disponível no seguinte link: [Simulador de Preços](https://simulador-precos.onrender.com/)

Os dados utilizados para treinar o modelo foram retirados do dataset Auto MPG Cars, disponível no UCI Machine Learning Repository.

Licença:
Este projeto está licenciado sob a MIT License.
