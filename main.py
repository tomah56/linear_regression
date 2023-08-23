from load_csv import load
from train_mode import start
# from test_mdoel_lib import test


def main():
    try:
        data_car = load("data.csv")
        trained_model = start(data_car)
        print("---- Precision of the modell ----")
        print('mean squared error: ', trained_model.mean_squared_error())

    except (KeyError, Exception) as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
