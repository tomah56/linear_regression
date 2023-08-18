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


def start(data):
    theta0 = 0
    theta1 = 0
    o_pricelist = data["price"]
    o_km_list = data['km']
    pricelist = normalize(o_pricelist)
    km_list = normalize(o_km_list)
    new_km_normalized = num_norm(160000, o_km_list)

    # print(normalize(km_list))
    # print(normalize(pricelist))

    num_samples =  len(km_list)
    num_iterations = 1000
    learning_rate = 0.1
    modell = Predict()
    losses = []
    losses2 = []
    # print(modell.estimate_price(4562))

    for _ in range(num_iterations):  # You'll need to decide the number of iterations
        predicted_prices = [modell.estimate_price(int(mileage)) for mileage in km_list]
        # print(predicted_prices)
        # predicted_prices = [estimate_price(mileage) for mileage in km_list]
        error0 = sum(predicted_price - true_price for predicted_price, true_price in zip(predicted_prices, pricelist))
        error1 = sum((predicted_price - true_price) * mileage for predicted_price, true_price, mileage in zip(predicted_prices, pricelist, km_list))

        theta0 -= learning_rate * (1/num_samples) * error0
        theta1 -= learning_rate * (1/num_samples) * error1
        # losses.append(error0)
        # losses2.append(error1)
        # losses.append(theta0)
        # losses2.append(theta1)
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
