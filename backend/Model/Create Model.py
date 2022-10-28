import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def results_evaluation(y_test, y_pred):
    np.place(y_test, y_test<1, [1])
    mae = mean_absolute_error(y_test, y_pred); print(f"MAE: {round(mae, 1)}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    mse = mean_squared_error(y_test, y_pred); print(f"MSE: {round(mse, 3)}")
    print("R^2 Score: {:.4f}".format(r2_score(y_test, y_pred)))

def main():
    df = pd.read_csv(__file__[:-15]+"/data/UsedCarsSA_Clean_EN.csv", delimiter=",", encoding="utf_8")

    print(df)
    df = df[df["Price"] != 0]
    print(df)

    df = df.reset_index(drop=True)
        

    df = df.drop(["Negotiable"], axis=1)

    df["Origin"] = df["Origin"].replace({"Saudi" : 3, "Gulf Arabic": 2, "Other": 1, "Unknown": 0})
    df["Options"] = df["Options"].replace({"Full" : 2, "Semi Full": 1, "Standard": 0})
    df["Fuel_Type"] = df["Fuel_Type"].replace({"Hybrid" : 2, "Diesel": 1, "Gas": 0})
    df["Gear_Type"] = df["Gear_Type"].replace({"Automatic": 1, "Manual": 0})

    from sklearn.preprocessing import OneHotEncoder
    oneHotEnc = OneHotEncoder()
    for column in ["Color", "Make", "Region"]:
        encDf = pd.DataFrame(oneHotEnc.fit_transform(np.array(df[column]).reshape(-1, 1)).toarray(), columns=oneHotEnc.get_feature_names_out([column]))
        df = df.join(encDf)
        df = df.drop(column, axis=1)

    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    df["Type"] = encoder.fit_transform(df["Type"].array.reshape(-1, 1))
    
    f = open("TypesEncoder.txt", "w", encoding="utf-8")
    for Type in encoder.categories_[0]:
        f.write(Type+",")
    f.close()

    for column in df.columns:
        df[column] = df[column].astype(float)
    
    f = open("columnNames.txt", "w", encoding="utf-8")
    for column in df.columns:
        f.write(column+",")
    f.close()
    
    # # Standrize
    mean = df["Price"].mean()
    sd = df["Price"].std()
    
    df = df[(df["Price"] <= mean+(1*sd))]


    y = df["Price"].copy()
    X = df.drop("Price", axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42) 

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    model = xgb.XGBRegressor( 
      n_estimators = 150
    )
    # model = RandomForestRegressor(criterion="poisson", n_estimators=250, random_state=42, n_jobs=-1)
    # model = LinearRegression(n_jobs = -1)
    model.fit(X_train, y_train, verbose=False)

    y_true = np.array(y_test, dtype=float)
    y_pred = np.array(model.predict(X_test), dtype=float)

    results_evaluation(y_true, y_pred)

    model.save_model(__file__[:-15]+"\models\model_v1.json")


    ## Code for evaluating the best model using kfold cross validation
    # k = 5
    # from sklearn.model_selection import KFold
    # kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.linear_model import LinearRegression
    # from sklearn import svm
    # from xgboost.sklearn import XGBRegressor
    # from catboost import CatBoostRegressor
    # from sklearn.kernel_ridge import KernelRidge
    # from sklearn.linear_model import ElasticNet
    # from sklearn.linear_model import BayesianRidge
    # from sklearn.ensemble import GradientBoostingRegressor
    # from sklearn.ensemble import RandomForestRegressor
    # results = []
    # for reg in [KNeighborsRegressor(), LinearRegression(), svm.SVR(), XGBRegressor(), KernelRidge(), ElasticNet(), BayesianRidge(), GradientBoostingRegressor(), RandomForestRegressor()]:
    #     print(reg)
    #     Mmae, Mrmse, Mr2 = TrainModel(X, y, reg, kf)
    #     print(f"MAE: {sum(Mmae)/k}")
    #     print(f"RMSE: {sum(Mrmse)/k}")
    #     print(f"R^2 Score: {sum(Mr2)/k}") 
    #     results.append([Mmae, Mrmse, Mr2, reg])

    # bestres = []
    # tres = sum(results[0][0])/k
    # for res in results:
    #     if sum(res[0])/k < tres:
    #         tres = sum(res[0])/k
    #         bestres = res
    # print(bestres)

def TrainModel(X, y, regressor, kf):
    Mmae = []; Mrmse = []; Mr2 = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)

        Mmae.append(mean_absolute_error(y_test, y_pred)) 
        Mrmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        Mr2.append(r2_score(y_test, y_pred))
    return Mmae, Mrmse, Mr2

if __name__ == "__main__":
    main()