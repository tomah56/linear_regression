from load_csv import load
from train_mode import start
import matplotlib.pyplot as plt

from test_mdoel_lib import test



def main():
    try:
        data_car = load("data.csv")
        trained_model = start(data_car)
        # print(trained_model.estimate_price(30000))
        # test(data_car)
        # print(data_car)
    except (KeyError, Exception) as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
