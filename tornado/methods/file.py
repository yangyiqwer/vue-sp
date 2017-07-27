#!/usr/bin/env Python
# coding=utf-8

from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import numpy as np

class File(object):
	"""docstring for File"""
	def __init__(self):
		pass

	def getFileList(self, mypath):
		files = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
		return files

	def getModelList(self):
		mypath = os.getcwd() + '/model'
		files = self.getFileList(mypath)
		dic = {}
		res = []
		for f in files:
			dic['model_value'] = f.split('.')[0]
			dic['model_label'] = f.split('.')[0]
			res.append(dic.copy())
		return res

	def getDatasetList(self, type):
		mypath = os.getcwd() + '/' + type
		files = self.getFileList(mypath)
		dic = {}
		res = []
		for f in files:
			dic['data_value'] = f
			dic['data_label'] = f
			res.append(dic.copy())
		return res


	def getModelData(self, modeltype):
		mypath = os.getcwd() + '/model_data/' + modeltype
		files = self.getFileList(mypath)
		res = []
		dic = {}
		for f in files:
			dic['modeldata_value'] = f
			dic['modeldata_label'] = f
			res.append(dic.copy())
		return res

	def getCSVdata(self, datapath, filename):
		mypath = os.getcwd() + '/' + datapath + '/' + filename
		f = open(mypath)
		df = pd.read_csv(f)
		date = np.array(df['date'])
		price = np.array(df['price'])
		for i in range(len(price)):
			price[i] = round(price[i],3)
		res = [{'date':date.tolist(), 'price':price.tolist()}]
		del df
		del date
		del price
		return res