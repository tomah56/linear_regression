from predict_price import Predict


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


class TrainModell():
    """This class contains all methodes and data for a liner regression training and estimation"""
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
        self.error_points = None
        self.train_nodell()
        self.denorm_modell()

    def __str__(self) -> str:
        return f"<Theta_0: {self.my_modell.theta_0}, Theta_1: {self.my_modell.theta_1}>"

    def train_nodell(self):
        num_samples =  self.data_length
        num_iterations = 100
        learning_rate = 0.5
        theta0 = 0
        theta1 = 0
        for _ in range(num_iterations):
            predicted_prices = [self.norm_my_modell.estimate_price(mileage) for mileage in self.norm_kms.norm_dataset]
            error0 = sum(predicted_price - true_price for predicted_price, true_price in zip(predicted_prices, self.norm_prices.norm_dataset))
            error1 = sum((predicted_price - true_price) * mileage for predicted_price, true_price, mileage in zip(predicted_prices, self.norm_prices.norm_dataset, self.norm_kms.norm_dataset))
            theta0 -= learning_rate * (1/num_samples) * error0
            theta1 -= learning_rate * (1/num_samples) * error1
            self.norm_my_modell.setTheta_0(theta0)
            self.norm_my_modell.setTheta_1(theta1)
    
    def denorm_modell(self):
        # theta_0_normalized = self.norm_my_modell.theta_0
        # theta_1_normalized = self.norm_my_modell.theta_1
        # print((f"Normalised:\nIntercept:{theta_0_normalized} Coefficients: {theta_1_normalized}"))
        self.norm_predicted = self.norm_my_modell.estimate_price(self.norm_kms.norm_dataset)

        self.predicted =  self.norm_prices.de_normalize(self.norm_predicted)
        x1  = self.kms[0]
        y1  = self.predicted[0]
        x2  = self.kms[len(self.kms) - 1]
        y2  = self.predicted[len(self.kms) - 1]

        theta_1_denormalized = (y2 - y1) / (x2 - x1)
        theta_0_denormalized = y1 - theta_1_denormalized * x1
        print((f"Original:\nIntercept:{theta_0_denormalized} Coefficients: {theta_1_denormalized}"))
        self.my_modell.theta_0 = theta_0_denormalized
        self.my_modell.theta_1 = theta_1_denormalized
        self.error_points = [abs(original - predicted) for original, predicted in zip(self.prices, self.predicted)]

    def mean_squared_error(self):
        squared_diff = [(original - predicted)**2 for original, predicted in zip(self.prices, self.predicted)]
        mse = sum(squared_diff) / len(self.prices)
        return mse
