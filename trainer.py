"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import ember
import argparse
import os
import sys
import pickle
import jsonlines
import utility
from lightgbm import plot_importance # ?
from sklearn.ensemble import forest as rf # ?
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

class ModelType(object):
    def train(self):
        raise (NotImplemented)

    def save(self):
        raise (NotImplemented)

class Gradientboosted(ModelType):
    """
    Train the LightGBM model from the vectorized features
    """
    def __init__(self, datadir, rows, dim):
        self.datadir = datadir
        self.rows = rows
        self.dim = dim
        self.model = None

    """
    Run Gradientboost algorithm which in lightgbm
    """
    def train(self):
        """
        Train
        """
        X, y = ember.read_vectorized_features(self.datadir, self.rows, self.dim)
        
        # train
        lgbm_dataset = lgb.Dataset(X, y)
        
        params = {
            "application": "binary",
            "boosting": "gbdt",
            "objective": "binary",
            "num_iterations": 1000,
            "learning_rate": 0.05,
            "num_leaves": 2048,
            "max_depth": 16,
            "min_data_in_leaf": 1000,
            "feature_fraction": 0.5,
            "verbose": -1  # log option
        }

        self.model = lgb.train(params, lgbm_dataset)

    def save(self):
        """
        Save a model using a pickle package
        """
        print('[GradientBoosted] start save')
        #logger.debug(self.model)
        if self.model:
            self.model.save_model(os.path.join(self.datadir, 'GradientBoosted_model.txt')) 
        #logger.debug('[GradientBoosted] finish save')

class X_Gradientboosted(ModelType):
    """
    Train the XGBoost model from the vectorized features
    """

    def __init__(self, datadir, rows, dim):
        self.datadir = datadir
        self.rows = rows
        self.dim = dim
        self.model = None

    #XGBoost
    def train(self):
        """
        Train
        """
        X, y= ember.read_vectorized_features(self.datadir, self.rows, self.dim)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        self.model = xgb.XGBClassifier(silent=False,
                                      booster='gbtree',
                                      scale_pos_weight=1,
                                      learning_rate=0.01,
                                      colsample_bytree=0.4,
                                      subsample=0.8,
                                      objective='binary:logistic',
                                      n_estimators=100,
                                      max_depth=50,
                                      gamma=10,
                                      seed=777)

        #파라미터 튜닝 : https://www.kaggle.com/lifesailor/xgboost 참조

        # train, Prediction
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        #Accuracy
        accuracy = accuracy_score(y_pred, y_test)
        print("X_GradientBoosted : ", accuracy)
    
    def save(self):
        """
        Save a model using a pickle package
        """
        print('[X_GradientBoosted] start save')
        # logger.debug(self.model)
        if self.model:
            joblib.dump(self.model, os.path.join(self.datadir, 'X_GradientBoosted_model.txt'))
            #self.model.save_model(os.path.join(self.datadir, 'X_GradientBoosted_model.txt'))
            # logger.debug('[X_GradientBoosted] finish save')

class RandomForestCF(ModelType):
    """
    Train the RandomForest model from the vectorized features
    """

    def __init__(self, datadir, rows, dim):
        self.datadir = datadir
        self.rows = rows
        self.dim = dim
        self.model = None


    # RandomForest
    def train(self):
        """
        Train
        """
        X, y = ember.read_vectorized_features(self.datadir, self.rows, self.dim)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        self.model = RandomForestClassifier(n_estimators=100,
                                          min_samples_leaf=25,
                                          max_features=0.5,
                                          n_jobs=-1,
                                          oob_score=False)
        
        #파라미터 튜닝 : https://injo.tistory.com/30 참조

        # train, Prediction
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        # Accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_pred, y_test)
        print("RandomForest : ", accuracy)

    def save(self):
        """
        Save a model using a pickle package
        """
        print('[RandomForest] start save')
        # logger.debug(self.model)
        if self.model:
            joblib.dump(self.model, os.path.join(self.datadir, 'RandomForest_model.txt'))
            #load_model = joblib.load('RandomForest_model.txt') #predict method , 
            #load_model.predict(X)
            # logger.debug('[RandomForest] finish save')

class Trainer:
    def __init__(self, jsonlpath, output):
        self.jsonlpath = jsonlpath
        self.output = output
        self.rows = 0
        self.model = None
        featurelist = utility.readonelineFromjson(jsonlpath)
        featuretype = utility.FeatureType()
        self.features = featuretype.parsing(featurelist)
        self.dim = sum([fe.dim for fe in self.features])

    def vectorize(self):
        # To do Error check 
        # if file is jsonl file
        if self.rows == 0:
            #logger.info('[Error] Please check if jsonl file is empty ...')
            return -1
        
        ember.create_vectorized_features(self.jsonlpath, self.output, self.rows, self.features, self.dim)

    def update_rows(self):
        """
        Update a rows variable
        """
        with jsonlines.open(self.jsonlpath) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                self.rows += 1

    def removeExistFile(self):
        """
        Remove Files
        """
        path_X = os.path.join(self.output, "X.dat")
        path_y = os.path.join(self.output, "y.dat")

        if os.path.exists(path_X):
            os.remove(path_X)
        if os.path.exists(path_y):
            os.remove(path_y)
    
        with open(path_X, 'w') as f:
            pass
        with open(path_y, 'w') as f:
            pass    

    def run(self):
        """
        Training
        """
        #self.removeExistFile()
        self.update_rows()
        if self.vectorize() == -1: 
            return
        class_list = [
        #Gradientboosted(self.output, self.rows, self.dim),
        X_Gradientboosted(self.output, self.rows, self.dim),
        RandomForestCF(self.output, self.rows, self.dim)
        ]
        for cl in class_list:
            cl.train()
            #cl.save()
        
        

