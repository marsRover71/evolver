
import random
# import sys
# import types
# import time
# import copy
# import math

import parameters as p

from pylab import figure #, show

#sys.path.append ('/Users/rhett/workspace/evolver')

import virtualMachine as vm

# def func (x) :
# 	#return math.sin (x*2) #.5 +   .2 * x + 100*math.sin (-x) +   .3 * (x ** 3)
# 	return .5 +   (.2 * x) +     0.4 *   (x ** 3.0) + math.sin (x*2) * 600 * math.cos(math.sqrt(x))


unusedID = 0


def plotFunction (sampleRange, processedResults, targetFuncResults=None, primordialResults=None, generation=0) :

	fig = figure (1)

	ax1 = fig.add_subplot (1,1,1)

	x = [_ * .1 for _ in xrange (sampleRange[0], sampleRange[1])]

	if targetFuncResults is not None :
		ax1.plot (x,targetFuncResults)

	ax1.plot (x,processedResults)

	ax1.grid (True)
	minP = min (processedResults)
	maxP = max (processedResults)
	margin = (maxP - minP) * 0.1
	ax1.set_ylim ((minP - margin,  maxP + margin))
	ax1.set_xlim ((min (x) - max (x) * 0.1,  max (x) * 1.1))

	fig.savefig('/Users/rhett/workspace/evolver/plots/individual.' + str (generation) + '.png', bbox_inches='tight')
	fig.clf ()

class Simulation :

	def __init__ (self,individual) :
		self.numSamples = 40
		self.currentSample = 0
		self.sampleRange = [0,self.numSamples]
		self.state = None
		self.individual = individual
		self.results = []
		self.processedResults = []
		self.targetFuncResults = []
		self.targetUpperBound =  0.0
		self.targetLowerBound =  0.0
		self.targetDomain = 0.0
		self.resultsUpperBound = 0.0
		self.resultsLowerBound = 0.0
		self.resultsDomain = 0.0
		self.resultsScaleFactor = 0.0
		self.yOffset = 0.0

	def postProcess (self) :

		self.targetUpperBound = max (self.targetFuncResults)
		self.targetLowerBound = min (self.targetFuncResults)
		self.targetDomain = self.targetUpperBound - self.targetLowerBound

		######### results are tuples, need to get the values out
		resultValues = []
		for value in self.results :
			if value is not None :
				resultValues.append (value[0])

			else :
				resultValues.append (0)

		self.resultsUpperBound = max (resultValues)
		self.resultsLowerBound = min (resultValues)
		self.resultsDomain = self.resultsUpperBound - self.resultsLowerBound

		if self.resultsDomain != 0.0 :

			self.resultsScaleFactor = self.targetDomain / float (self.resultsDomain)

			if self.resultsScaleFactor != 0.0 :
				self.yOffset = self.targetLowerBound - self.resultsLowerBound
				self.processedResults = [(r-self.resultsLowerBound) * self.resultsScaleFactor + self.targetLowerBound for r in resultValues]

			else :
				self.processedResults = [(r-self.resultsLowerBound) + self.targetLowerBound for r in resultValues]

		else :
			self.processedResults = [(r-self.resultsLowerBound) + self.targetLowerBound for r in resultValues]

	def computeFitness (self) :

		resultSum = 0
		for index,sample in enumerate (self.processedResults) :
			resultSum += abs (self.targetFuncResults[index] - sample)

		self.individual.fitness = resultSum

	def run (self, generation) :

		func = p.parameters["targetFunction"]
		self.results = []
		self.targetFuncResults = []
		for _sample in range (self.sampleRange[0], self.sampleRange[1]) :

			sample = (_sample/float (self.sampleRange[1]-self.sampleRange[0]))*20
			result = self.individual.phenotype.execute(inputs = [sample])

			self.results.append (result)
			self.targetFuncResults.append (func (sample))

		if len (self.results) > 0 :
			self.postProcess ()
			self.computeFitness ()

class Individual :

	opCodePrec = vm.Instruction.opCodePrecision
	operTypePrec = vm.Instruction.argTypePrecision
	argumentSetPrec = vm.Instruction.argumentPrecision

	def __init__ (self, parentA=None, parentB=None, initGeneration=None) :

		global unusedID

		self.numLines = p.parameters["numInstructions"]

		self.ID = unusedID
		self.phenotype = vm.Program (randomSeed=self.ID, expressionSize=self.numLines)

		if (parentA is None and
			parentB is None ) :
			self.parents = [None, None]

		elif (parentA is None and
			parentB is not None) :
			raise Exception, "Individual: ParentA is None"

		elif (parentA is not None and
			parentB is None) :
			raise Exception, "Individual: ParentB is None"

		else :
			self.parents = [parentA, parentB]

		self.initGeneration = initGeneration

		self.globalSeed = 4 + self.ID

# 		self.POS = None
		self.fitness = None
		self.fitnessRank = -1
		self.closestNeighborDistance = None
		self.closestNeighborIndex = None
		self.diversity = 0.0
		self.active = True
		self.genotype = None # binary string

		self.simulation = Simulation (self)

		unusedID += 1

	def mutate (self, probabilityOfFeatureMutation, progress) :

		binaryStr = self.phenotype.genotype

		strToList = [bool (int (_)) for _ in binaryStr]

		opCodePart = vm.Instruction ().opCodePrecision + 30
		for _ in xrange (p.parameters["numMutatePoints"]) :
			bitPlace = opCodePart + int (random.random () * (len(binaryStr) - opCodePart))
			strToList[bitPlace] = not strToList[bitPlace] 

		binaryStr = ''
		for bit in strToList :
			binaryStr += str (int (bit))

		self.phenotype.genotype = binaryStr
		self.expressGenotypeToPhenotype ()

	def expressGenotypeToPhenotype (self) :
		self.phenotype.flatBinaryToInstructions (self.phenotype.genotype)

if False :
	for generation in xrange (10) :
		print "Gen " + str (generation)
		i = Individual ()
		i.phenotype.synthesize ()
		sim = Simulation (i)
		i.expressGenotypeToPhenotype ()
		print "Primordial"
		sim.run (generation)

		i.mutate (1, 0)
		i.expressGenotypeToPhenotype ()

		mutatedSim = Simulation (i)
		mutatedSim.run (generation)
		print "MUTATED"
		plotFunction(mutatedSim.sampleRange, mutatedSim.processedResults, mutatedSim.targetFuncResults, sim.processedResults)

