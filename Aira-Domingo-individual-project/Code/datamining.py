import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression




datasetm = pd.read_csv("NFA_2018.csv", delimiter=",")
# subsetting observations from records EFConsTotGHA
footprintm = datasetm[datasetm.record == 'EFConsTotGHA']  # create new dataframe
# rearrange columns
footprintm = footprintm[
    ['country', 'ISO alpha-3 code', 'UN_region', 'UN_subregion', 'year', 'record', 'crop_land', 'grazing_land',
     'forest_land', 'fishing_ground', 'built_up_land', 'population', 'carbon', 'total', 'Percapita GDP (2010 USD)']]
# nan missing data
footprintm = footprintm.replace(0, np.NaN)
# drop rows with no data and data from world
footprintm = footprintm[footprintm.crop_land.notna()]
footprintm = footprintm[footprintm.country != 'World']
# fill nan with 0
footprintm = footprintm.fillna(0)
# split dataset into X and Y
values = footprintm.values
if value == 'crop':
    var = 6
elif value == 'grazing':
    var = 7
elif value == 'forest':
    var = 8
elif value == 'fishing':
    var = 9
else:
    var = 10
X = values[:, var]
Y = values[:, 12]
# train test split 70/30
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=50)
# make into dataframes
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
# impute missing value with mean
imputer = SimpleImputer(missing_values=0, strategy='mean')
# impute training set
imp_x_train = imputer.fit_transform(x_train)
imp_x_test = imputer.transform(x_test)
imp_y_train = imputer.fit_transform(y_train)
imp_y_test = imputer.transform(y_test)
# scale data
scaler = MinMaxScaler(feature_range=(0, 100))
scaled_x_train = scaler.fit_transform(imp_x_train)
scaled_x_test = scaler.transform(imp_x_test)
scaled_y_train = scaler.fit_transform(imp_y_train)
scaled_y_test = scaler.transform(imp_y_test)
# convert to dataframe
scaled_x_train = pd.DataFrame(scaled_x_train)
scaled_x_test = pd.DataFrame(scaled_x_test)
scaled_y_train = pd.DataFrame(scaled_y_train)
scaled_y_test = pd.DataFrame(scaled_y_test)
# train with linear regression model using training sets
model = LinearRegression(normalize=True)
model.fit(scaled_x_train, scaled_y_train)
carbon_y_pred = model.predict(scaled_x_test)
test_r2 = r2_score(scaled_y_test, carbon_y_pred)
test_error = mean_squared_error(scaled_y_test, carbon_y_pred)
model.intercept_
model.coef_
