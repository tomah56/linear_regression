import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def test(data):
    X = data[['km']]
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    coefficients = model.coef_
    intercept = model.intercept_

    print("\n----- SKLEARN ------\n")
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # model evaluation
    # Calculate the mean squared error
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # printing values
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # Plot the data and the linear regression line
    plt.scatter(X, y, color='black', label='original')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('SkLearn control plot')
    plt.legend()
    plt.show()


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