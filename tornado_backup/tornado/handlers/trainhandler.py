#!/usr/bin/env Python
# coding=utf-8

import tornado.web
from methods.train import Train
import tornado.escape

class TrainHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		self.set_header("Access-Control-Allow-Origin", "*")
		self.set_header("Access-Control-Allow-Headers", "x-requested-with")
		self.set_header('Access-Control-Allow-Methods', 'POST, GET')

	def get(self):
		t = Train()
		request_name = self.get_argument('name', ' ')

		if(request_name == 'modelset'):
			request_list = t.getModelList()
		elif(request_name == 'dataset'):
			request_list = t.getTraindataset()
		elif(request_name == 'train'):
			modeltype = self.get_argument('model', ' ')
			train_data = self.get_argument('data', ' ')
			save_name = self.get_argument('save', ' ')
			request_list = t.getTrainResult(modeltype, train_data, save_name)
		elif(request_name == 'trainlist'):
			request_list = t.getTraininglist()
		else:
			request_list = ['haha']

		respon_json = tornado.escape.json_encode(request_list)
		self.write(respon_json)
