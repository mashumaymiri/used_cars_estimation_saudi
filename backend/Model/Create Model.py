import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def results_evaluation(y_test, y_pred):
    np.place(y_test, y_test<1, [1])
    mape = mean_absolute_percentage_error(y_test, y_pred); print(f"MAPE: {round(mape, 3)}")
    mae = mean_absolute_error(y_test, y_pred); print(f"MAE: {round(mae, 1)}")
    mdape_ = np.median((np.abs(np.subtract(y_test, y_pred)/ y_test))); print(f"MdAPE: {round(mdape_, 3)}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    print("R^2 Score: {:.4f}".format(r2_score(y_test, y_pred)))

def main():
    df = pd.read_csv(__file__[:-15]+"/data/UsedCarsSA_Clean_EN.csv", delimiter=",", encoding="utf_8")

    for column in df.columns:
                    df[column] = pd.Categorical(df[column])

    df["Price"] = df["Price"].astype(int)
    df["Mileage"] = df["Mileage"].astype(int)

    df= df[df['Price'] != 0]

    print(df)
    print(df.columns)

    y = df["Price"].copy()
    X = df.drop("Price", axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42) 

    clf = xgb.XGBRegressor(
        tree_method="gpu_hist", enable_categorical=True, use_label_encoder=True
    )

    clf.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False)

    y_true = np.array(y_test, dtype=float)
    y_pred = np.array(clf.predict(X_test), dtype=float)

    results_evaluation(y_true, y_pred)

    clf.save_model(__file__[:-15]+"\models\model_v1.json")


if __name__ == "__main__":
    main()