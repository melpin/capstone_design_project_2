"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
import tqdm
import ember
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from collections import OrderedDict
import utility
import pickle
import joblib
from features import PEFeatureExtractor

#util
def format_spliter(filename):
    if "." in filename:
        return filename.split(".")[0]
    else:
        return filename

class Predictor:
    def __init__(self, testdir, features, output):
        # load model with pickle to predict
        self.testdir = testdir
        self.output = output
        self.features = features
        self.modellist = dict()
        self.extractor = None
        self.err = 0
    
    def lgbmodel_load(self, modelpath):
        with open(modelpath, 'rb') as f: # cause error check?
            self.modellist["LGB"] = lgb.Booster(model_file=modelpath)

    def xgbmodel_load(self, modelpath):
        self.modellist["XGB"] = joblib.load(modelpath)
    
    def rfmodel_load(self, modelpath):
        self.modellist["RF"] = joblib.load(modelpath)
    
    def extract_data(self, file_data, featurelist):
        if self.extractor == None:
            self.extractor = PEFeatureExtractor(featurelist)
        features = np.array(self.extractor.feature_vector(file_data), dtype=np.float32)
        return features
    
    def predict_sample(self, modelname, features, y_list):
        try:
            y_list.append(self.modellist[modelname].predict(features.reshape(1,-1))[0])
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(modelname+' error')
            print(e)
            y_list.append(0)
            self.err += 1
    
    def run(self):
        lgby = []
        name = []
        end = len(next(os.walk(self.testdir))[2])
        

        for sample in tqdm.tqdm(utility.directory_generator(self.testdir), total=end):
            fullpath = os.path.join(self.testdir, sample)

            if os.path.isfile(fullpath):
                binary = open(fullpath, "rb").read()
                name.append(format_spliter(sample))
                features = self.extract_data(binary, self.features)
                self.predict_sample("LGB", features, lgby)
                #self.predict_sample("XGB", features, xgby)
                #self.predict_sample("RF", features, rfy)
            
        
        lgby = np.where(np.array(lgby) > 0.7, 1, 0)
        #other model already classifyed
        
        series = OrderedDict([
                    ('ID', name),
                    ('Class', lgby),
                            ])
        r = pd.DataFrame.from_dict(series)
        r.to_csv(self.output, index=False)#, header=None)
        
        print('{} error is occured'.format(self.err))
