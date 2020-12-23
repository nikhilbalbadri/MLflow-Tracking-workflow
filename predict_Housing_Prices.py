import numpy as np
import pandas as pd
import warnings
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def fill_NAN(data):
    data['PoolQC'] = data['PoolQC'].replace(np.nan,"No Pool")
    data['Fence'] = data['Fence'].replace(np.nan,"No Fence")
    data['MiscFeature'] = data['MiscFeature'].replace(np.nan,"No MiscFeature")
    data['Alley'] = data['Alley'].replace(np.nan,"No Alley")
    data['GarageType'] = data['GarageType'].replace(np.nan,"No Garage")
    data['GarageFinish'] = data['GarageFinish'].replace(np.nan,"No Garage")
    data['GarageQual'] = data['GarageQual'].replace(np.nan,"No Garage")
    data['GarageCond'] = data['GarageCond'].replace(np.nan,"No Garage")
    data['FireplaceQu'] = data['FireplaceQu'].replace(np.nan,"No Fireplace")
    data['BsmtQual'] = data['BsmtQual'].replace(np.nan,"No Basement")
    data['BsmtCond'] = data['BsmtCond'].replace(np.nan,"No Basement")
    data['BsmtExposure'] = data['BsmtExposure'].replace(np.nan,"No Basement")
    data['BsmtFinType1'] = data['BsmtFinType1'].replace(np.nan,"No Basement")
    data['BsmtFinType2'] = data['BsmtFinType2'].replace(np.nan,"No Basement")
    data['MasVnrType'] = data['MasVnrType'].replace(np.nan,"None")
    data['MasVnrArea'] = data['MasVnrArea'].replace(np.nan,0.0)
    impute = SimpleImputer(strategy='mean')
    data['LotFrontage'] = impute.fit_transform(data[['LotFrontage']])
    data = data[~data['Electrical'].isnull()]
    data = data.drop(columns=['Id','GarageYrBlt'])
    return data

def encoding_variables(data):
    data['MSZoning'] = data['MSZoning'].apply(lambda x: "C" if x == "C (all)" else x)
    data['LandContour'] = data['LandContour'].apply(lambda x: ['Low', 'HLS', 'Bnk', 'Lvl'].index(x))
    data['LotShape'] = data['LotShape'].apply(lambda x: ['IR3', 'IR2', 'IR1', 'Reg'].index(x))
    data['Alley'] = data['Alley'].apply(lambda x: ['No Alley', 'Grvl', 'Pave'].index(x))
    data['Street'] = data['Street'].apply(lambda x: 0 if x == "Grvl" else 1)
    data['MSZoning'] = data['MSZoning'].apply(lambda x: ['A', 'FV', 'I', 'RL','RP','RM','C','RH'].index(x))
    data['Utilities'] = data['Utilities'].apply(lambda x: ['Elo', 'NoSeWa', 'NoSewr', 'AllPub'].index(x))
    data['LotConfig'] = data['LotConfig'].apply(lambda x: ['FR3', 'FR2', 'CulDSac', 'Corner','Inside'].index(x))
    data['LandSlope'] = data['LandSlope'].apply(lambda x: ['Sev', 'Mod', 'Gtl'].index(x))
    Neighborhood_Order = list(data.groupby('Neighborhood')['SalePrice'].mean().sort_values().keys())
    data['Neighborhood'] = data['Neighborhood'].apply(lambda x: Neighborhood_Order.index(x))
    Condition1_Order = list(data.groupby('Condition1')['SalePrice'].mean().sort_values().keys())
    data['Condition1'] = data['Condition1'].apply(lambda x: Condition1_Order.index(x))
    Condition2_Order = list(data.groupby('Condition2')['SalePrice'].mean().sort_values().keys())
    data['Condition2'] = data['Condition2'].apply(lambda x: Condition2_Order.index(x))
    data['BldgType'] = data['BldgType'].apply(lambda x: ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'].index(x))
    data['HouseStyle'] = data['HouseStyle'].apply(lambda x: ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'].index(x))
    data['RoofStyle'] = data['RoofStyle'].apply(lambda x: ['Shed', 'Mansard', 'Hip', 'Gambrel','Gable','Flat'].index(x))
    data['RoofMatl'] = data['RoofMatl'].apply(lambda x: ['WdShngl', 'WdShake', 'Tar&Grv', 'Roll', 'Metal', 'Membran', 'CompShg', 'ClyTile'].index(x))
    Exterior1st_Order = list(data.groupby('Exterior1st')['SalePrice'].mean().sort_values().keys())
    data['Exterior1st'] = data['Exterior1st'].apply(lambda x: Exterior1st_Order.index(x))
    Exterior2nd_Order = list(data.groupby('Exterior2nd')['SalePrice'].mean().sort_values().keys())
    data['Exterior2nd'] = data['Exterior2nd'].apply(lambda x: Exterior2nd_Order.index(x))
    data['MasVnrType'] = data['MasVnrType'].apply(lambda x: ['None', 'CBlock', 'BrkCmn', 'BrkFace','Stone'].index(x))
    data['ExterQual'] = data['ExterQual'].apply(lambda x: ['Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['ExterCond'] = data['ExterCond'].apply(lambda x: ['Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    Foundation_Order = list(data.groupby('Foundation')['SalePrice'].mean().sort_values().keys())
    data['Foundation'] = data['Foundation'].apply(lambda x: Foundation_Order.index(x))
    data['BsmtQual'] = data['BsmtQual'].apply(lambda x: ['No Basement','Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['BsmtCond'] = data['BsmtCond'].apply(lambda x: ['No Basement','Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['BsmtExposure'] = data['BsmtExposure'].apply(lambda x: ['No Basement','No', 'Mn', 'Av', 'Gd'].index(x))
    data['BsmtFinType1'] = data['BsmtFinType1'].apply(lambda x: ['No Basement','Unf', 'LwQ', 'Rec', 'BLQ','ALQ','GLQ'].index(x))
    data['BsmtFinType2'] = data['BsmtFinType2'].apply(lambda x: ['No Basement','Unf', 'LwQ', 'Rec', 'BLQ','ALQ','GLQ'].index(x))
    Heating_Order = list(data.groupby('Heating')['SalePrice'].mean().sort_values().keys())
    data['Heating'] = data['Heating'].apply(lambda x: Heating_Order.index(x))
    data['HeatingQC'] = data['HeatingQC'].apply(lambda x: ['Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['CentralAir'] = data['CentralAir'].apply(lambda x: 0 if x == "No" else 1)
    Electrical_Order = list(data.groupby('Electrical')['SalePrice'].mean().sort_values().keys())
    data['Electrical'] = data['Electrical'].apply(lambda x: Electrical_Order.index(x))
    data['KitchenQual'] = data['KitchenQual'].apply(lambda x: ['Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['Functional'] = data['Functional'].apply(lambda x: ['Sal','Sev','Maj2', 'Maj1', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'].index(x))
    data['FireplaceQu'] = data['FireplaceQu'].apply(lambda x: ['No Fireplace','Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['GarageType'] = data['GarageType'].apply(lambda x: ['No Garage','Detchd', 'CarPort', 'BuiltIn', 'Basment','Attchd', '2Types'].index(x))
    data['GarageFinish'] = data['GarageFinish'].apply(lambda x: ['No Garage','Unf', 'RFn', 'Fin'].index(x))
    data['GarageQual'] = data['GarageQual'].apply(lambda x: ['No Garage','Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['GarageCond'] = data['GarageCond'].apply(lambda x: ['No Garage','Po', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['PavedDrive'] = data['PavedDrive'].apply(lambda x: ['N','P', 'Y'].index(x))
    data['PoolQC'] = data['PoolQC'].apply(lambda x: ['No Pool', 'Fa', 'TA', 'Gd','Ex'].index(x))
    data['Fence'] = data['Fence'].apply(lambda x: ['No Fence', 'MnWw','GdWo', 'MnPrv', 'GdPrv'].index(x))
    data['MiscFeature'] = data['MiscFeature'].apply(lambda x: ['No MiscFeature', 'Shed', 'Gar2', 'Othr','Elev','TenC'].index(x))
    SaleType_Order = list(data.groupby('SaleType')['SalePrice'].mean().sort_values().keys())
    data['SaleType'] = data['SaleType'].apply(lambda x: SaleType_Order.index(x))
    SaleCondition_Order = list(data.groupby('SaleCondition')['SalePrice'].mean().sort_values().keys())
    data['SaleCondition'] = data['SaleCondition'].apply(lambda x: SaleCondition_Order.index(x))
    return data

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/nikhilbalbadri/MLflow_workflow/main/Data/train.csv"
    )
    try:
        data = pd.read_csv(csv_url)
    except Exception as e:
        logger.exception(
            "Unable to download CSV, check your internet connection. Error: %s", e
        )
    
    regression_model = sys.argv[1] if len(sys.argv) > 1 else "linear"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    intercept = sys.argv[3] if len(sys.argv) > 3 else True
    feature_count = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    split_ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
    
    data = fill_NAN(data)
    data = encoding_variables(data)
    fm = data.drop(columns=['SalePrice'])
    tv = data['SalePrice']
    fm_train, fm_test, tv_train, tv_test = train_test_split(fm, tv, test_size=split_ratio, random_state = 10)
    
    if feature_count > 0:
        fs = SelectKBest(score_func=f_regression, k=feature_count)
        fs.fit(fm_train, tv_train)
        fm_train = fs.transform(fm_train)
        fm_test = fs.transform(fm_test)
    else:
        feature_count = len(fm_train.columns)

    with mlflow.start_run():
        if regression_model == "linear":
            modelName = "Linear"
            model = LinearRegression(fit_intercept=intercept)
        elif regression_model == "ridge":
            modelName = "Ridge"
            model = Ridge(alpha=aplha, fit_intercept=intercept, random_state=20)
        else:
            modelName = "Lasso"
            model = Lasso(aplha=aplha, fit_intercept=intercept, random_state=20)
            
        model.fit(fm_train, tv_train)

        predicted_price = model.predict(fm_test)

        (rmse, mae, r2) = eval_metrics(tv_test, predicted_price)

        print(modelName, " model")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("Alpha", alpha)
        mlflow.log_param("Model", regression_model)
        mlflow.log_param("Intercept", intercept)
        mlflow.log_param("Features count for SelectKBest", feature_count)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name=modelName)
        else:
            mlflow.sklearn.log_model(model, "model")