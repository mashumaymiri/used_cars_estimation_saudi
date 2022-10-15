import sys
import pandas as pd 
import xgboost as xgb
import numpy as np

def runModel(args):
    Make, Type, Year, Origin, Color, Options, Engine_Size, Fuel_Type, Gear_Type, Mileage, Region = args
    print(Make, Type, Year, Origin, Color, Options, Engine_Size, Fuel_Type, Gear_Type, Mileage, Region)
    
    f = open("columnNames.txt", "r", encoding="utf-8")
    columns = f.read()
    f.close()
    columns = columns[:-1].split(",")
    df = pd.DataFrame(columns=columns)
    df.loc[0] = 0

    df["Make"] = Make
    #
    f = open("TypesEncoder.txt", "r", encoding="utf-8")
    TypeA = f.read()
    f.close()
    TypeA = TypeA[:-1].split(",")
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder()
    enc.fit(np.array([TypeA]).reshape(-1, 1))
    TypeR = enc.transform(np.array([Type]).reshape(-1, 1))
    df["Type"] = TypeR[0]
    #
    df["Year"] = Year
    df["Origin"] = Origin
    # df["Color"] = Color
    df["Color_"+Color] = 1
    df["Options"] = Options
    df["Engine_Size"] = Engine_Size
    df["Fuel_Type"] = Fuel_Type
    df["Gear_Type"] = Gear_Type
    df["Mileage"] = Mileage
    # df["Region"] = Region
    df["Price"] = 0
    df.fillna(0)


    # df.loc[len(df.index)] = [Make, Type, Year, Origin, Options, Engine_Size, Fuel_Type, Gear_Type, Mileage, 0]
    print(df)
    # df = df.drop(["Region"], axis=1)

    df["Origin"] = df["Origin"].replace({"Saudi" : 3, "Gulf Arabic": 2, "Other": 1, "Unknown": 0})
    df["Options"] = df["Options"].replace({"Full" : 2, "Semi Full": 1, "Standard": 0})
    df["Fuel_Type"] = df["Fuel_Type"].replace({"Hybrid" : 2, "Diesel": 1, "Gas": 0})
    df["Gear_Type"] = df["Gear_Type"].replace({"Automatic": 1, "Manual": 0})

    

    # dataframe = pd.concat([dataframe, dummies], axis=1)
    # dataframe["حي_"+district] = 1

    # df = pd.DataFrame(columns=['Make', 'Type', 'Year', 'Origin', 'Color', 'Options', 'Engine_Size', 'Fuel_Type', 'Gear_Type', 'Mileage', 'Region', 'Price', 'Negotiable'])
    

    print(df)

    for column in df.columns:
        if column in ["Make"]:
            df[column] = pd.Categorical(df[column])
        else:
            df[column] = df[column].astype(float)

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

# runModel(sys.argv[1:])
runModel([
    'Kia',       'Optima',
    2014,        'Saudi',
    'Black',     'Standard',
    1.8,         'Gas',
    'Automatic', 60000,
    'Riyadh'
  ])