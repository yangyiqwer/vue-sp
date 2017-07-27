#!/usr/bin/env Python
# coding=utf-8

from methods.file import File
import sys
import os
import subprocess

class Pred(File):
	"""docstring for Test"""
	def __init__(self):
		super(Pred, self).__init__()
		pass

	def getPreddataset(self):
		return self.getDatasetList('test_dataset')
	
	def getPredResult(self, modeltype, Pred_data, model_data):
		mypath = os.getcwd() + '\\model\\' + modeltype +'.py'
		'''
		output = os.popen(mypath + ' pred')
		output = output.read()
		output = output.strip('[')
		output = output.rstrip()
		output = output.rstrip(']')
		'''

		'''
		subprocess.check_output(args, *, stdin=None, stderr=None, shell=False, universal_newlines=False)
		'''
		
		child = subprocess.Popen("python " + mypath + " pred",  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
		(stdout,stderr) = child.communicate()
		output = str(stdout)
		output = output.strip("b'[")
		output = output.rstrip()
		output = output.rstrip(']\\r\\n')
		outputlist = int(output)
		data = self.getCSVdata('test_dataset', Pred_data)
		return [{'pred' : outputlist, 'data': data}]
