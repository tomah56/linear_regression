import json
import matplotlib.pyplot as plt


class Predict():
    def __init__(self) -> None:
        self.theta_0 = 0
        self.theta_1 = 0
        self.avg_error = 0

    def setTheta_0(self, newdata):
        self.theta_0 = newdata

    def setTheta_1(self, newdata):
        self.theta_1 = newdata

    def setError(self, newdata):
        self.avg_error = newdata
    
    def estimate_price(self, mileage):
        return self.theta_0 + (self.theta_1 * mileage)


def plot(predict: Predict, esti, user):
    x1 = 9999
    if user <= x1:
        x1 -= 5000
    x2 = 250000
    if user >= x2:
        x2 += 5000
    x3 = x1 + 5000
    y1 = predict.estimate_price(x1)
    y2 = predict.estimate_price(x2)
    y3 = predict.estimate_price(x3)
    km = [x1 , x2]
    price = [y1, y2]
    err = predict.avg_error
    # plt.figure(figsize=(8, 6))
    plt.scatter(user, esti , color='green', label='predicted')
    plt.plot(km, price , color='blue', label='Linear Regression Line')
    plt.errorbar(user, esti, err, fmt='none', color='red', capsize=4, label='Error bars')
    text = f'slope: {predict.theta_1:.4f}\nintersect: {int(predict.theta_0)}'
    plt.annotate(text, xy=(x3, y3), xytext=(x3 - 5000, y3 - 2000),
             arrowprops=dict(facecolor='black', shrink=1))
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title(f'Your estimated Price: {int(esti)} for km: {user}')
    plt.legend()
    plt.show()

def main():
    try:
        with open('thetas.json', 'r') as f:
            data = json.load(f)
        theta0 = data['theta0']
        theta1 = data['theta1']
        err = data['avg_error']
        user_input = 89000
        if not user_input:
            user_input = input("Please enter a km (mileage): ")
        program = Predict()
        program.setTheta_0(theta0)
        program.setTheta_1(theta1)
        program.setError(err)
        estimated_price = program.estimate_price(int(user_input))
        print(int(estimated_price))
        plot(program, estimated_price, int(user_input))
    except (KeyError, Exception) as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
