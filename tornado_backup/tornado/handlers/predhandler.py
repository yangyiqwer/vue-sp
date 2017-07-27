#!/usr/bin/env Python
# coding=utf-8

import tornado.web
from methods.pred import Pred
import tornado.escape

class PredHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		self.set_header("Access-Control-Allow-Origin", "*")
		self.set_header("Access-Control-Allow-Headers", "x-requested-with")
		self.set_header('Access-Control-Allow-Methods', 'POST, GET')

	def get(self):
		p = Pred()
		request_name = self.get_argument('name', ' ')

		if(request_name == 'modelset'):
			request_list = p.getModelList()
		elif(request_name == 'dataset'):
			request_list = p.getPreddataset()
		elif(request_name == 'modeldataset'):
			modeltype = self.get_argument('model', ' ')
			request_list = p.getModelData(modeltype)
		elif(request_name == 'pred'):
			modeltype = self.get_argument('model', ' ')
			pred_data = self.get_argument('data', ' ')
			model_data = self.get_argument('modeldata', ' ')
			request_list = p.getPredResult(modeltype, pred_data, model_data)
		else:
			request_list = ['']

		respon_json = tornado.escape.json_encode(request_list)
		self.write(respon_json)
