

parameters = {	"numInstructions" : 7,
				"numMutatePoints" : 100,
				"numInputs" : 1,
				"maxSymbols" : 20,

	#			"probabilityOfFeatureMutation" : 2,
				"minimizeOrMaximize" : "minimize",
				"maxNumGenerations" : 100,

				"fitnessTarget" : .01,
				"initialPopulationSize" : 20, # >= 2 primordial
				"maxPopulationSize" : 10,

				"numOffspring" : 2,
				"diversityTestRadius" : 0.001,   # higher = less diversity

				"vizFileBasename" : "/Users/rhett/workspace/genetic1/fitnessViz/fitness",

				"realMax" : 20.0,
				"inputMax" : 1,
				"intMax" : 20,
				"symbolMax" : 4  #len (self.phenotype.symbolTable)-1
		}