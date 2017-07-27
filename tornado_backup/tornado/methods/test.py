#!/usr/bin/env Python
# coding=utf-8

from methods.file import File
import sys
import os
import subprocess

class Test(File):
	"""docstring for Test"""
	def __init__(self):
		super(Test, self).__init__()
		pass

	def getTestdataset(self):
		return self.getDatasetList('test_dataset')

	def getTestResult(self, modeltype, test_data, model_data):
		res = []
		data = self.getCSVdata('test_dataset', test_data)
		result = []
		realtrend = []
		wrongpred = []
		mypath = os.getcwd() + '\\model\\' + modeltype +'.py'
		child = subprocess.Popen("python " + mypath + " test",  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
		(stdout,stderr) = child.communicate()
		output = str(stdout)
		output = output.strip("b'[")
		output = output.rstrip()
		output = output.rstrip(']\\r\\n')
		'''
		output = os.popen(mypath + ' test')
		output = output.read()
		output = output.strip('[')
		output = output.rstrip()
		output = output.rstrip(']')
		'''

		outputlist = [int(i) for i in output.split(',')]
		dic = {}
		'''
		for i in range(len(data[0]['date']) - 1):
			if(data[0]['price'][i+1] - data[0]['price'][i] > 0):
				dic['name'] = 'rise'
				dic['itemStyle'] = {'normal':{'color':'#ff0000'}}
			elif(data[0]['price'][i+1] - data[0]['price'][i] < 0):
				dic['name'] = 'fall'
				dic['itemStyle'] = {'normal':{'color':'#008000'}}
			else:
				dic['name'] = 'hold'
				dic['itemStyle'] = {'normal':{'color':'#FFFFFF'}}
			dic['value'] = [1, data[0]['date'][i], data[0]['date'][i+1], 1]
			result.append(dic.copy())
			if(outputlist[i] == 1):
				dic['name'] = 'rise'
				dic['itemStyle'] = {'normal':{'color':'#ff0000'}}
			else:
				dic['name'] = 'fall'
				dic['itemStyle'] = {'normal':{'color':'#008000'}}
			dic['value'] = [0, data[0]['date'][i], data[0]['date'][i+1], 1]
			result.append(dic.copy())
		'''
		predright = 0
		for i in range(len(data[0]['date']) - 1):
			if(data[0]['price'][i+1] - data[0]['price'][i] > 0):
				realtrend.append(1)
			elif(data[0]['price'][i+1] - data[0]['price'][i] < 0):
				realtrend.append(-1)
			else:
				realtrend.append(0)
			if(realtrend[-1] == outputlist[i]):
				wrongpred.append(0)
			else:
				wrongpred.append(1)
				predright += 1
		tmp = data[0]['date'].copy()
		del tmp[0]
		text = [{'total':len(outputlist),'Correct_rate': round(((predright / len(outputlist)) * 100), 2)}]
		res.append({'Trend':{'date':tmp,'realtrend':realtrend, 'predtrend': outputlist, 'wrongpred': wrongpred}, 'data': data, 'text': text})
		return res