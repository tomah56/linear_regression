import pandas as pd
from load_csv import load
from predict_price import Predict
import matplotlib.pyplot as plt


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

def normalize(data):
    return (data - calculate_mean(data)) / calculate_standard_deviation(data)


def num_norm(number, data):
    return (number - calculate_mean(data)) / calculate_standard_deviation(data)

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



class TrainModell():
    def __init__(self, data) -> None:
        self.prices = data["price"]
        self.kms = data["km"]
        self.data_length = len(self.kms)
        self.norm_prices = DataProcess(self.prices)
        self.norm_kms = DataProcess(self.kms)
        self.norm_my_modell = Predict()
        self.my_modell = Predict()
        self.train_nodell()
        self.denorm_modell()

    def __str__(self) -> str:
        return f"<Theta_0: {self.my_modell.theta_0}, Theta_1: {self.my_modell.theta_1}>"


    def train_nodell(self):
        num_samples =  self.data_length
        num_iterations = 100
        learning_rate = 0.01
        theta0 = 0.02
        theta1 = -0.2
        for _ in range(num_iterations):
            predicted_prices = [self.norm_my_modell.estimate_price(int(mileage)) for mileage in self.norm_kms.norm_dataset]

            error0 = sum(predicted_price - true_price for predicted_price, true_price in zip(predicted_prices, self.norm_prices.norm_dataset))
            error1 = sum((predicted_price - true_price) * mileage for predicted_price, true_price, mileage in zip(predicted_prices, self.norm_prices.norm_dataset, self.norm_kms.norm_dataset))

            theta0 -= learning_rate * (1/num_samples) * error0
            theta1 -= learning_rate * (1/num_samples) * error1
            self.norm_my_modell.setTheta_0(theta0)
            self.norm_my_modell.setTheta_1(theta1)
    
    def denorm_modell(self):
        theta_0_normalized = self.norm_my_modell.theta_0
        theta_1_normalized = self.norm_my_modell.theta_1
        print((f"Normalised: {theta_0_normalized} and {theta_1_normalized}"))

        theta_0_denormalized = theta_0_normalized * self.norm_prices.std + self.norm_prices.mean + 2000
        theta_1_denormalized = (theta_1_normalized * self.norm_prices.std) / self.norm_kms.std
        print((f"Original: {theta_0_denormalized} and {theta_1_denormalized}"))

        self.my_modell.setTheta_0(theta_0_denormalized)
        self.my_modell.setTheta_1(theta_1_denormalized)


def norm_plot(test):
    plt.scatter(test.norm_kms.norm_dataset, test.norm_prices.norm_dataset, color='black', label='original')

    # Calculate the corresponding normalized predictions
    y_line_normalized = test.norm_my_modell.theta_0 + test.norm_my_modell.theta_1 * test.norm_kms.norm_dataset

    # Plot the normalized linear regression line
    plt.plot(test.norm_kms.norm_dataset, y_line_normalized, color='red', label='Normalized Linear Regression Line')

    new_km_normalized = test.norm_kms.normalize(98000)

    predicted_price_normalized = test.norm_my_modell.estimate_price(new_km_normalized)
    # Plot the predicted point
    plt.scatter(new_km_normalized, predicted_price_normalized, color='green', label='Predicted Point')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Price Estimation based on Kilometers')
    plt.legend()
    plt.show()

def plot_original(test: TrainModell):
    plt.scatter(test.kms, test.prices , color='black', label='original')

    # Calculate the corresponding normalized predictions
    y_line = test.my_modell.theta_0 + test.my_modell.theta_1 * test.kms

    # Plot the normalized linear regression line
    plt.plot(test.kms, y_line, color='red', label='Linear Regression Line')

    # new_km_normalized = test.norm_kms.normalize(98000)

    # predicted_price_normalized = test.norm_my_modell.estimate_price(new_km_normalized)
    # # Plot the predicted point
    # plt.scatter(new_km_normalized, predicted_price_normalized, color='green', label='Predicted Point')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Price Estimation based on Kilometers')
    plt.legend()
    plt.show()



def start(data):
    test = TrainModell(data)
    # print(test)
    norm_plot(test)
    plot_original(test)

    return test


def old(data):
    # Intercept 8811
    theta0 = 0
    # coefficients -0.02
    theta1 = 0
    o_pricelist = data["price"]
    o_km_list = data['km']
    pricelist = normalize(o_pricelist)
    km_list = normalize(o_km_list)
    new_km_normalized = num_norm(160000, o_km_list)
    num_samples =  len(km_list)
    num_iterations = 1000
    learning_rate = 0.1
    modell = Predict()
 
    for _ in range(num_iterations):  # You'll need to decide the number of iterations
        predicted_prices = [modell.estimate_price(int(mileage)) for mileage in km_list]

        error0 = sum(predicted_price - true_price for predicted_price, true_price in zip(predicted_prices, pricelist))
        error1 = sum((predicted_price - true_price) * mileage for predicted_price, true_price, mileage in zip(predicted_prices, pricelist, km_list))

        theta0 -= learning_rate * (1/num_samples) * error0
        theta1 -= learning_rate * (1/num_samples) * error1
        modell.setTheta_0(theta0)
        modell.setTheta_1(theta1)

    predicted_price_normalized = modell.estimate_price(new_km_normalized)



    # plt.plot(range(num_iterations), losses)
    # plt.plot(range(num_iterations), losses, marker='o', color='blue', label='Loss')
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Learning Curve")
    # plt.show()
    # # print(estimate_price(85223))

    # print("km")
    # print(km_list)
    # print("price")
    # print(pricelist)
    plt.scatter(km_list, pricelist, color='black', label='original')
    # plt.scatter(X_test, y_test, color='blue', label='Actual')

    # Normalize the 'km' values for plotting the line
    # x_line_normalized = (X - mean_km) / std_km
    # x_line_normalized_sorted = km_list.sort_values(by='km')

    # Calculate the corresponding normalized predictions
    y_line_normalized = theta0 + theta1 * km_list

    # Plot the normalized linear regression line
    plt.plot(km_list, y_line_normalized, color='red', label='Normalized Linear Regression Line')


    # Plot the predicted point
    plt.scatter(new_km_normalized, predicted_price_normalized, color='green', label='Predicted Point')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Price Estimation based on Kilometers')
    plt.legend()
    plt.show()

    return modell