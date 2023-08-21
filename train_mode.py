import pandas as pd
from load_csv import load
from predict_price import Predict
import matplotlib.pyplot as plt
from t_m_utils import TrainModell
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


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
    plt.scatter(test.kms, test.predicted , color='green', label='predicted')

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
    # plot_original(test)
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(data)
    # print(normalized_data)
 

    print("---- Original ----")
    rmse = mean_squared_error(test.kms, test.predicted)
    r2 = r2_score(test.kms, test.predicted)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    print("---- Normalised ----")
    rmse = mean_squared_error(test.norm_kms.norm_dataset, test.norm_predicted)
    r2 = r2_score(test.norm_kms.norm_dataset, test.norm_predicted)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    newBob = newModell()

    newBob.standardnorm(data)
    # printing values

    return test

class newModell():
    def __init__(self) -> None:
        self.theta_0 = 0
        self.theta_1 = 0

    def estimate_price(self, mileage):
        return self.theta_0 + (self.theta_1 * mileage)


    def train_nodell(self, d_price, d_kms):
        num_samples =  len(d_kms)
        num_iterations = 100
        learning_rate = 0.5
        theta0 = 0
        theta1 = 0
        for _ in range(num_iterations):
            predicted_prices = [self.estimate_price(mileage) for mileage in d_kms]
            error0 = sum(predicted_price - true_price for predicted_price, true_price in zip(predicted_prices, d_price))
            error1 = sum((predicted_price - true_price) * mileage for predicted_price, true_price, mileage in zip(predicted_prices, d_price, d_kms))

            theta0 -= learning_rate * (1/num_samples) * error0
            theta1 -= learning_rate * (1/num_samples) * error1
            self.theta_1 = theta1
            self.theta_0 = theta0

    def standardnorm(self, data):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        X = normalized_data[:, :-1]  # All columns except the last one
        y = normalized_data[:, -1]
        self.train_nodell(X, y)

        #original
        prices = data["price"]
        kms = data["km"]
        print("\n---- Standard Normalised ----\n")

        print(self.theta_0, self.theta_1)
        norm_predicted = self.estimate_price(X)
        rmse = mean_squared_error(X, norm_predicted)
        r2 = r2_score(X, norm_predicted)
        print('Root mean squared error: ', rmse)
        print('R2 score: ', r2)

        predicted_prices_original_scale = scaler.inverse_transform(np.column_stack((kms, norm_predicted)))[:, -1]
        # Plot the original data and the predicted prices
        plt.scatter(kms, prices, label='Original Data')
        plt.plot(kms, predicted_prices_original_scale, color='red', label='Predicted Prices')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.legend()
        plt.title('Original Data vs. Predicted Prices')
        plt.show()

        plt.scatter(X, y, color='black', label='original')
        plt.scatter(X, norm_predicted , color='green', label='predicted')
        # Calculate the corresponding normalized predictions
        y_line = self.theta_0 + self.theta_1 * X
        # Plot the normalized linear regression line
        plt.plot(X, y_line, color='red', label='Linear Regression Line')
        plt.xlabel('Kilometers')
        plt.ylabel('Price')
        plt.title('Standard New Norm')
        plt.legend()
        plt.show()