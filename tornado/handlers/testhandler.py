#!/usr/bin/env Python
# coding=utf-8

import tornado.web
from methods.test import Test
import tornado.escape

class TestHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		self.set_header("Access-Control-Allow-Origin", "*")
		self.set_header("Access-Control-Allow-Headers", "x-requested-with")
		self.set_header('Access-Control-Allow-Methods', 'POST, GET')

	def get(self):
		t = Test()
		request_name = self.get_argument('name', ' ')

		if(request_name == 'modelset'):
			request_list = t.getModelList()
		elif(request_name == 'dataset'):
			request_list = t.getTestdataset()
		elif(request_name == 'modeldataset'):
			modeltype = self.get_argument('model', ' ')
			request_list = t.getModelData(modeltype)
		elif(request_name == 'test'):
			modeltype = self.get_argument('model', ' ')
			test_data = self.get_argument('data', ' ')
			model_data = self.get_argument('modeldata', ' ')
			request_list = t.getTestResult(modeltype, test_data, model_data)
		else:
			request_list = ['']

		respon_json = tornado.escape.json_encode(request_list)
		self.write(respon_json)
