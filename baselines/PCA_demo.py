#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
sys.path.append('../')
from loglizer.models import PCA
from loglizer import preprocessing
from util import dataloader
import time
import argparse

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument("-d", "--domain", type=str, required=True, help='lab, Cons, or rxout')
parser.add_argument("-th", "--threshold", type=int, required=True)
# Parse the argument
args = parser.parse_args()

threshold = args.threshold
domain = args.domain
path_log = 'data_preprocessed/'+domain+'/'
log_normal = path_log + 'normal_{}.csv'.format(threshold)
log_abnormal = path_log + 'anomaly_{}.csv'.format(threshold)
train_ratio = 0.9
path_results = './results/'+domain+'/PCA_{}_{}/'.format(train_ratio, threshold)
os.makedirs(path_results, exist_ok=True)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(log_normal,
                                                                log_abnormal,
                                                                window='session',
                                                                train_ratio=train_ratio,
                                                                split_type='uniform')
    start_time = time.time()
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    model = PCA()
    model.fit(x_train)

    training_time = time.time() - start_time
    print("--- Training process: %s seconds ---" % (training_time))
    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    P, R, F1 = model.evaluate(x_test, y_test)

    performance = {'Values': [P, R, F1, training_time]}
    df_p = pd.DataFrame(performance, index=['P',
                                            'R',
                                            'F1',
                                            'Training_time(s)'])
    df_p.to_csv(path_results + 'performance.csv')


