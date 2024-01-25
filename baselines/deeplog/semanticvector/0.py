# -*- coding: utf-8 -*-
"""
Created on Wed Oct 06 13:09 2021

@author: hniu1

the code is used to generate template.txt file
"""

import pandas as pd

import chakin

chakin.download(number=2, save_dir='./')

# path_data = '../../../data_preprocessed/Drain_results/'
#
# df_tem = pd.read_csv(path_data + 'HDFS.log_templates.csv', index_col=False)
#
# list_tem = df_tem['EventTemplate'].tolist()
#
# with open(path_data + 'template.txt', 'w') as f:
#     for item in list_tem:
#         f.write(item + '\n')

print('test finished')
