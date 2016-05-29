
import time

def targetFunc (x) :
	return 2*x + 3*x**2 + x**3

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

