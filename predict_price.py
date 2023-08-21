class Predict():
    def __init__(self) -> None:
        self.theta_0 = 0
        self.theta_1 = 0

    def setTheta_0(self, newdata):
        self.theta_0 = newdata

    def setTheta_1(self, newdata):
        self.theta_1 = newdata
    
    def estimate_price(self, mileage):
        return self.theta_0 + (self.theta_1 * mileage)
