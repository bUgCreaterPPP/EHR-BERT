#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer import preprocessing
from util import dataloader
import time
threshold = 50
path_log = 'data_preprocessed/Cons/'
log_normal = path_log + 'normal_{}.csv'.format(threshold)
log_abnormal = path_log + 'anomaly_{}.csv'.format(threshold)
train_ratio = 0.9
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.1 # the threshold for anomaly detection
path_results = './results/Cons/LogCluster_{}_{}/'.format(train_ratio, threshold)
os.makedirs(path_results, exist_ok=True)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(log_normal,
                                                                log_abnormal,
                                                                window='session', 
                                                                train_ratio=train_ratio,
                                                                split_type='uniform')
    start_time = time.time()
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

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
