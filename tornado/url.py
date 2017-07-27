#!/usr/bin/env Python
# coding=utf-8
"""
the url structure of website
"""

'''
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
'''

from handlers.trainhandler import TrainHandler
from handlers.testhandler import TestHandler
from handlers.predhandler import PredHandler

url = [
	(r'/train/', TrainHandler),
	(r'/test/', TestHandler),
	(r'/pred/', PredHandler),
]
