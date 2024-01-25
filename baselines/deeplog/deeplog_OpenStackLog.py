#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
# sys.path.append('../')

from logdeep.models.lstm import deeplog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters
# training_ratio = 0.5

options = dict()
options['window_size'] = 10
options['device'] = "cpu"
options['training_ratio'] = 0.5

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 37 # number of event template

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = '../../results/deeplog_OpenStackLog/' + str(options['training_ratio']) + '/'
options['raw_data_dir'] = '../../data_preprocessed/OpenStackLog/'
options['data_dir'] = '../../data_preprocessed/OpenStackLog_split/' + str(options['training_ratio']) + '/'

# Predict
options['model_path'] = '../../results/deeplog_OpenStackLog/' + str(options['training_ratio']) + '/deeplog_last.pth'
options['num_candidates'] = 15

seed_everything(seed=1234)


def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
