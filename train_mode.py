import pandas as pd
from load_csv import load
from predict_price import Predict
import matplotlib.pyplot as plt
from t_m_utils import TrainModell
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


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
    # norm_plot(test)
    # plot_original(test)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    print(normalized_data)

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

    # printing values

    return test
