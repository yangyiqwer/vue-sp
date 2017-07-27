#!/usr/bin/env Python
# coding=utf-8

from methods.file import File
import sys
import os
import subprocess

class Train(File):
	"""docstring for Test"""
	traininglist = []
	finishlist = []

	def __init__(self):
		super(Train, self).__init__()
		pass

	def getTraindataset(self):
		return self.getDatasetList('train_dataset')
	
	def getTrainResult(self, modeltype, train_data, save_name):
		mypath = os.getcwd() + '\\model\\' + modeltype +'.py'
		self.traininglist.append({'child' : subprocess.Popen("python " + mypath + " train",  stdout=None, stderr=None, shell = True), \
			'model': modeltype, 'dataset':train_data, 'modeldata_name':save_name})
		return [{'result': True}];

	def getTraininglist(self):
		traininglistres = []
		finishlistres = []
		res = []
		dic = {}
		for i in range(len(self.traininglist)-1, -1, -1):
			if self.traininglist[i]['child'].poll() == 0:
				dic['model'] = self.traininglist[i]['model']
				dic['dataset'] = self.traininglist[i]['dataset']
				dic['modeldata_name'] = self.traininglist[i]['modeldata_name']
				self.finishlist.append(dic.copy())
				del self.traininglist[i]

		for i in range(len(self.traininglist)):
			'''
			dic = self.traininglist[i].copy()
			'''
			dic['model'] = self.traininglist[i]['model']
			dic['dataset'] = self.traininglist[i]['dataset']
			dic['modeldata_name'] = self.traininglist[i]['modeldata_name']
			dic['index'] = i + 1;
			traininglistres.append(dic.copy())
		for i in range(len(self.finishlist)):
			dic = self.finishlist[i].copy()
			dic['index'] = i + 1;
			finishlistres.append(dic.copy())
		res = [{'traininglist': traininglistres, 'finishlist': finishlistres}]
		return res