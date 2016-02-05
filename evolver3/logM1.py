
# import math
# import random
import inspect
# import time
# #import numpy
# #import copy
# #import types
# from scipy import spatial
# from pylab import figure, show


		

class Log :
	def __init__ (self) :
		self.indentNum = 0

	def pr (self, stringIn ) :
		print self.indentNum*'\t' + str (stringIn)

	def prV (self, nameOfVar,  ) :
		nameOfVar = str (nameOfVar)
		frame = inspect.currentframe ()
		if nameOfVar in frame.f_back.f_locals :
			self.pr (nameOfVar + " = " + str (frame.f_back.f_locals[nameOfVar]))

		else :
			self.pr (nameOfVar + " = ???")

	def indent (self) :
		self.indentNum += 1

	def dedent (self) :
		self.indentNum -= 1
		#print

l = Log()

