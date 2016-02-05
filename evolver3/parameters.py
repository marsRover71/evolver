

parameters = {	"numInstructions" : 5,
				"numMutatePoints" : 20,

				"probabilityOfFeatureMutation" : 2,
				"minimizeOrMaximize" : "minimize",
				"maxNumGenerations" : 100,

				"fitnessTarget" : .01,
				"initialPopulationSize" : 20, # >= 2 primordial
				"maxPopulationSize" : 30,

				"numOffspring" : 1,
				"diversityTestRadius" : 0.001,   # higher = less diversity

				"vizFileBasename" : "/Users/rhett/workspace/genetic1/fitnessViz/fitness",

				"realMax" : 3.0,
				"inputMax" : 1,
				"intMax" : 3,
				"symbolMax" : 4  #len (self.phenotype.symbolTable)-1
		}