import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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