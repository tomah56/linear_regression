import matplotlib.pyplot as plt
from t_m_utils import TrainModell


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
    plt.figure(figsize=(8, 6))
    plt.scatter(test.kms, test.prices , color='black', label='original')
    plt.scatter(test.kms, test.predicted , color='green', label='predicted')
    plt.plot(test.kms, test.predicted , color='blue', label='Linear Regression Line')
    plt.errorbar(test.kms, test.predicted, test.error_points, fmt='none', color='red', capsize=4, label='Error bars')
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Price Estimation based on Kilometers')
    plt.legend()
    plt.grid(True)
    plt.show()


def start(data):
    test = TrainModell(data)
    norm_plot(test)
    plot_original(test)
    
    return test
