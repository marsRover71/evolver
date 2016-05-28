
import math
import time

def targetFunc (x) :
	#return math.sqrt(x)*30.0+x*0.05+0.0125*x**3
	return 2*x + 3*x**2 + x**3
# 	return .5 +   (.2 * x) +     0.4 *   (x ** 3.0) + math.sin(x) * 60

# def func (x) :
# 	#return math.sin (x*2) #.5 +   .2 * x + 100*math.sin (-x) +   .3 * (x ** 3)
# 	return .5 +   (.2 * x) +     0.4 *   (x ** 3.0) + math.sin (x*2) * 600 * math.cos(math.sqrt(x))


parameters = {	"numInstructions" : 10,
				"numMutatePoints" : 150, #150
				"numInputs" : 1,
				"maxSymbols" : 20,
				"gracePeriod" : 10,
				"numElites" : 5,

				"minimizeOrMaximize" : "minimize",
				"maxNumGenerations" : 400,
				"apocalypseCycle" : 50,

				"maxFitnessSimilarity" : 1,
				"initialPopulationSize" : 10, # >= 2 primordial
				"maxPopulationSize" : 20,

				"numOffspring" : 2,

				"vizFileBasename" : "/Users/rhett/workspace/evolver/fitnessViz/fitness",

				"realMax" : 100.0,
				"realMin" : -100.0,

				"inputMax" : 1,
				"intMax" : 20,
				"intMin" : -20,
				"targetFunction" : targetFunc,
				"symbolMax" : 4,
				"randomSeed" : time.time (),
				"overflowMessagesOn" : False,
				"argumentDeltaInit": 10.0
		}

