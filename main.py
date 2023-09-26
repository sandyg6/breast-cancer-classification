# -*- coding: utf-8 -*-

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpy
import seaborn as sea
import pickle
#import warnings
from sklearn.model_selection import train_test_split
def prediction(user_id, radius_mean, texture_mean, perimeter_mean,area_mean, smoothness_mean, compactness_mean, concavity_mean,concave_points_mean, symmetry_mean, fractal_dimension_mean):
    df = pd.read_csv("cancerdataset.csv")
    df['diagnosis'] = df['diagnosis'].replace(to_replace='M', value=0.0)
    df['diagnosis'] = df['diagnosis'].replace(to_replace='B', value=1.0)

    benign_df =df[df['diagnosis']==0.0][0:200]
    malignant_df =df[df['diagnosis']==1.0][0:200]
    axes = benign_df.plot(kind = 'scatter',x='area_mean',y= 'perimeter_mean', color='blue',label='Benign')
    malignant_df.plot(kind = 'scatter',x='area_mean',y= 'perimeter_mean', color='red',label='Malignant', ax=axes)

    df= df[pd.to_numeric(df['user_id'],errors='coerce').notnull()]

    df['user_id'].astype('float')

    feature_df = ['user_id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']
#Independent var
    x = np.asarray(feature_df)
#dependent var
    y = np.asarray(df['diagnosis'])

    x = df.drop("diagnosis",axis=1)
    y = df["diagnosis"]

    x= x.values
    y= y.values
    train_x = x[:80]
    train_y = y[:80]
    test_x = x[80:]
    test_y = y[80:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#model training and testing
    from sklearn.ensemble import RandomForestClassifier
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(x_train,y_train)

    random_forest_classifier.score(x_test,y_test)

#pickle file
    pickle.dump(random_forest_classifier,open('model.pkl','wb'))
    model = pickle.load(open('model.pkl','rb'))
    hp = [[user_id, radius_mean, texture_mean, perimeter_mean,area_mean, smoothness_mean, compactness_mean, concavity_mean,concave_points_mean, symmetry_mean, fractal_dimension_mean]]
    predicts = model.predict(hp)
    if predicts == [0.]:
        predicts = "The predicted cancer is Malignant"
    elif predicts == [1.]:
        predicts = "The predicted cancer is Benign"
    return predicts
