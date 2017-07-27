#!/usr/bin/env Python
# coding=utf-8

from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import numpy as np
import random
import sys
import time

#datapath = 'test_dataset'
#filename = '500_test.csv'
#mypath = os.getcwd() + '\\' + datapath + '\\' + filename
mypath = '/home/longxj/tornado/test_dataset/' + '/500_test.csv'
f = open(mypath)
df = pd.read_csv(f)
date = np.array(df['date'])
ans = []
if sys.argv[1] == 'test':
	for i in range(len(date) - 1):
		a = 0
		while a == 0:
			a = random.randint(-1, 1)
		ans.append(a)
elif sys.argv[1] == 'pred':
	a = 0
	while a == 0:
		a = random.randint(-1, 1)
	ans.append(a)
elif sys.argv[1] == 'train':
	time.sleep(60)
	ans.append(True)

print(ans)
