#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
sys.path.append('../')
from loglizer.models import InvariantsMiner
from loglizer import preprocessing
from util import dataloader
import time

path_log = '../data_preprocessed/OpenStackLog/'
log_normal = path_log + 'sequence_normal.csv'
log_abnormal = path_log + 'sequence_abnormal.csv'
train_ratio = 0.50
path_results = '../results/IVM_OpenStackLog/IVM_' + str(train_ratio) + '/'
os.makedirs(path_results, exist_ok=True)
epsilon = 0.5

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_OpenStackLog(log_normal,
                                                                log_abnormal,
                                                                window='session',
                                                                train_ratio=train_ratio,
                                                                split_type='uniform')
    start_time = time.time()
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    model = InvariantsMiner(epsilon=epsilon)
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
