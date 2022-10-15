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

    df = df[df["Region"] == "Riyadh"]
    df = df[df["Price"] != 0]

    df = df.reset_index(drop=True)
        

    df = df.drop(["Negotiable", "Region"], axis=1)

    df["Origin"] = df["Origin"].replace({"Saudi" : 3, "Gulf Arabic": 2, "Other": 1, "Unknown": 0})
    df["Options"] = df["Options"].replace({"Full" : 2, "Semi Full": 1, "Standard": 0})
    df["Fuel_Type"] = df["Fuel_Type"].replace({"Hybrid" : 2, "Diesel": 1, "Gas": 0})
    df["Gear_Type"] = df["Gear_Type"].replace({"Automatic": 1, "Manual": 0})

    from sklearn.preprocessing import OneHotEncoder
    oneHotEnc = OneHotEncoder()
    for column in ["Color", "Make"]:#, "Region"]:
        encDf = pd.DataFrame(oneHotEnc.fit_transform(np.array(df[column]).reshape(-1, 1)).toarray(), columns=oneHotEnc.get_feature_names_out([column]))
        df = df.join(encDf)
        df = df.drop(column, axis=1)

    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    df["Type"] = encoder.fit_transform(df["Type"].array.reshape(-1, 1))

    for column in df.columns:
            df[column] = df[column].astype(float)
        
    print(df)
    print(df.columns)

    y = df["Price"].copy()
    X = df.drop("Price", axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42) 

    model = xgb.XGBRegressor(
        tree_method="gpu_hist"
    )

    model.fit(X_train, y_train, verbose=False)

    y_true = np.array(y_test, dtype=float)
    y_pred = np.array(model.predict(X_test), dtype=float)

    results_evaluation(y_true, y_pred)

    model.save_model(__file__[:-15]+"\models\model_v1.json")


if __name__ == "__main__":
    main()