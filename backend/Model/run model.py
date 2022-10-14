import pandas as pd 
import xgboost as xgb

def runModel():
    df = pd.DataFrame(columns=['Make', 'Type', 'Year', 'Origin', 'Color', 'Options', 'Engine_Size', 'Fuel_Type', 'Gear_Type', 'Mileage', 'Region', 'Price', 'Negotiable'])
    
    df.loc[len(df.index)] = ['Make', 'Type', 'Year', 'Origin', 'Color', 'Options', 'Engine_Size', 'Fuel_Type', 'Gear_Type', 100, 'Region', 0, 'Negotiable']

    for column in df.columns:
                    df[column] = pd.Categorical(df[column])

    df["Price"] = df["Price"].astype(int)
    df["Mileage"] = df["Mileage"].astype(int)

    y = df["Price"].copy()
    X = df.drop("Price", axis=1).copy()

    dtest = xgb.DMatrix(X,label=y, enable_categorical=True)
    
    model = xgb.Booster()
    print("loding model")
    
    model.load_model(fname=__file__[:-12]+"/models/model_v1.json")

    print("predicting")
    # y_true = np.array(y_test, dtype=float)
    y_pred = model.predict(dtest)
    # Main.results_evaluation(y_true, y_pred)
    print(y_pred)
    return y_pred

runModel()