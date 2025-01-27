# Classe que define um carro

# Imports
import joblib
import numpy as np

# Classe
class Car:
    """
    Classe para preparação de dados e previsão de preços de carros.
    """

    def __init__(self, car):
        """
        Inicializa a classe com os dados do carro fornecidos.
        :param car: Lista contendo os atributos do carro.
        """
        if not isinstance(car, list) or len(car) != 11:
            raise ValueError("Os dados do carro devem ser uma lista com exatamente 11 valores.")
        self.car = car

    def prepare(self):
        """
        Prepara os dados do carro para previsão.
        :return: Lista numpy formatada para entrada no modelo.
        """
        # Lista de resultados inicializada com zeros
        result = np.zeros(55)

        try:
            # Extraindo atributos numéricos
            result[0] = float(self.car[0])  # Engine Size
            result[1] = float(self.car[1])  # Cylinders
            result[2] = float(self.car[2])  # Horse Power
            result[3] = float(self.car[3])  # MPG City
            result[4] = float(self.car[4])  # MPG Highway
            result[5] = float(self.car[5])  # Weight
            result[6] = float(self.car[6])  # Wheel Base
            result[7] = float(self.car[7])  # Length
        except ValueError:
            raise ValueError("Um dos valores numéricos fornecidos não é válido. Certifique-se de que sejam números.")

        # Mapeamento de categorias
        make = {
            'acura': 8, 'audi': 9, 'bmw': 10, 'buick': 11, 'cadillac': 12, 'chevrolet': 13, 'chrysler': 14, 'dodge': 15, 
            'ford': 16, 'gmc': 17, 'honda': 18, 'hummer': 19, 'hyundai': 20, 'infiniti': 21, 'isuzu': 22, 'jaguar': 23, 
            'jeep': 24, 'kia': 25, 'land_rover': 26, 'lexus': 27, 'lincoln': 28, 'mini': 29, 'mazda': 30, 'mercedes-benz': 31, 
            'mercury': 32, 'mitsubishi': 33, 'nissan': 34, 'oldsmobile': 35, 'pontiac': 36, 'porsche': 37, 'saab': 38, 
            'saturn': 39, 'scion': 40, 'subaru': 41, 'suzuki': 42, 'toyota': 43, 'volkswagen': 44, 'volvo': 45
        }
        body_type = {'hybrid': 46, 'suv': 47, 'sedan': 48, 'sports': 49, 'truck': 50, 'wagon': 51}
        drive = {'4wd': 52, 'fwd': 53, 'rwd': 54}

        try:
            # Mapeando valores categóricos
            result[make[self.car[8].lower()]] = 1  # Fabricante
            result[body_type[self.car[9].lower()]] = 1  # Tipo de carroceria
            result[drive[self.car[10].lower()]] = 1  # Tipo de tração
        except KeyError as e:
            raise KeyError(f"O valor '{e.args[0]}' não é válido. Verifique as opções para fabricante, tipo de veículo ou tração.")

        return result

    def predict(self, car):
        """
        Faz a previsão do preço do carro com base nos dados preparados.
        :param car: Dados preparados do carro.
        :return: Valor previsto.
        """
        car_to_predict = [car]
        try:
            # Carregando o modelo treinado
            model = joblib.load('modelo/modelo.pkl')
        except FileNotFoundError:
            raise FileNotFoundError("O modelo treinado não foi encontrado no diretório especificado.")

        # Fazendo a previsão
        predicted_car_value = model.predict(car_to_predict)
        return predicted_car_value[0]
