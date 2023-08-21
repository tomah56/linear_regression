from predict_price import Predict
from sklearn.preprocessing import StandardScaler
import numpy as np


def calculate_mean(numbers):
    """calculates the mean"""
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    return mean

def calculate_standard_deviation(numbers):
    """calculates the std"""
    variance = calculate_variance(numbers)
    std_deviation = variance ** 0.5
    return std_deviation


def calculate_variance(numbers):
    """calculates the vairance"""
    mean = calculate_mean(numbers)
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    variance = squared_diff_sum / len(numbers)
    return variance


class DataProcess():
    """normalising a given dataset adn saving its key values
    have method to normalise any other value outside of the original dataset
    and also have a method to denormalize it
    """
    def __init__(self, dataset) -> None:
        self.mean = calculate_mean(dataset)
        self.std = calculate_standard_deviation(dataset)
        self.norm_dataset = (dataset - self.mean) / self.std
    
    def normalize(self, value):
        return (value - self.mean) / self.std

    def de_normalize(self, value):
        return value * self.std + self.mean
    

def standarization(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    X = normalized_data[:, :-1]  # All columns except the last one
    y = normalized_data[:, -1]




class TrainModell():
    def __init__(self, data) -> None:
        self.prices = data["price"]
        self.kms = data["km"]
        self.data_length = len(self.kms)
        self.norm_prices = DataProcess(self.prices)
        self.norm_kms = DataProcess(self.kms)
        self.norm_my_modell = Predict()
        self.my_modell = Predict()
        self.predicted = None
        self.norm_predicted = None
        self.train_nodell()
        self.denorm_modell()

    def __str__(self) -> str:
        return f"<Theta_0: {self.my_modell.theta_0}, Theta_1: {self.my_modell.theta_1}>"

    def train_nodell(self):
        num_samples =  self.data_length
        num_iterations = 1000
        learning_rate = 0.001
        theta0 = 0
        theta1 = 0
        for _ in range(num_iterations):
            predicted_prices = [self.norm_my_modell.estimate_price(mileage) for mileage in self.norm_kms.norm_dataset]
            # predicted_prices = self.norm_my_modell.estimate_price(self.norm_kms.norm_dataset)

            error0 = sum(predicted_price - true_price for predicted_price, true_price in zip(predicted_prices, self.norm_prices.norm_dataset))
            error1 = sum((predicted_price - true_price) * mileage for predicted_price, true_price, mileage in zip(predicted_prices, self.norm_prices.norm_dataset, self.norm_kms.norm_dataset))

            theta0 -= learning_rate * (1/num_samples) * error0
            theta1 -= learning_rate * (1/num_samples) * error1
            self.norm_my_modell.setTheta_0(theta0)
            self.norm_my_modell.setTheta_1(theta1)
    
    def denorm_modell(self):
        theta_0_normalized = self.norm_my_modell.theta_0
        theta_1_normalized = self.norm_my_modell.theta_1
        print((f"Normalised:\nIntercept:{theta_0_normalized} Coefficients: {theta_1_normalized}"))
        self.norm_predicted = self.norm_my_modell.estimate_price(self.norm_kms.norm_dataset)

        theta_0_denormalized = theta_0_normalized * self.norm_prices.std + self.norm_prices.mean
        theta_1_denormalized = (theta_1_normalized * self.norm_prices.std) / self.norm_kms.std
        print((f"Original:\nIntercept:{theta_0_denormalized} Coefficients: {theta_1_denormalized}"))

        self.my_modell.setTheta_0(theta_0_denormalized)
        self.my_modell.setTheta_1(theta_1_denormalized)
        # self.predicted = self.my_modell.estimate_price(self.kms)
        self.predicted =  [self.my_modell.estimate_price(mileage) for mileage in self.kms]
        # print(self.predicted)