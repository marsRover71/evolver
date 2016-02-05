
import random
import sys
import types
import time
import copy
import math

from pylab import figure, show

sys.path.append ('/Users/rhettcollier/workspace/genetic1/src')

import programM1 as program

import logM1

# if l doesn't exist, make it otherwise use the existing one
try :
	l

except :
	l = logM1.Log()

def func (x) :
	return math.sin (x*2) #.5 +   .2 * x + 100*math.sin (-x) +   .3 * (x ** 3)
#fitness invert to get around local maxima



unusedID = 0


def plotFunction (sampleRange, processedResults, targetFuncResults=None, primordialResults=None) :
	#print "plot function"
	#samples = []

	#t = arange (0.0, 1.0, 0.01)

	fig = figure (1)

	ax1 = fig.add_subplot (1,1,1)

	#ax1.text(40, 0, expression)

	#ax1.plot (t, sin (2*pi*t))
	x = [_ * .1 for _ in range (sampleRange[0], sampleRange[1])]
	print x
	#hf = [func (_) for _ in x]
	#print "hf"
	#print hf
	#print "results"
	#print self.results
	
	#print "max(self.processedResults) - min(self.processedResults)"
	#print max(self.processedResults) - min(self.processedResults)
	
	if max(processedResults) - min(processedResults) == 0.0 :
		return
	
	#print self.targetFuncResults
	if targetFuncResults is not None :
		print targetFuncResults
		ax1.plot (x,targetFuncResults)
	#print len(x)
	#print len(self.results)
# 			hfUpperBound = max (hf)
# 			hfLowerBound = min (hf)
# 			hfDomain = hfUpperBound - hfLowerBound
# 			
# 			resultsUpperBound = max (self.results)
# 			resultsLowerBound = min (self.results)
# 			resultsDomain = resultsUpperBound - resultsLowerBound
# 			
# 			if resultsDomain != 0.0 :
# 				scaleFactor = hfDomain / resultsDomain
# 				if scaleFactor != 0.0 :
# 					yOffset = hfLowerBound - resultsLowerBound
# 					results = [(r-resultsLowerBound) * scaleFactor + hfLowerBound for r in self.results]
# 					diff = sum ([abs (results[index] - hf[index]) for index in range (len (x))])
# 					print "diff = ", diff
# 					ax1.plot (x,results)
# 
# 				else :
# 					ax1.plot (x,self.results)
# 					
# 			else :
	#print self.processedResults
	ax1.plot (x,processedResults)
	
	if primordialResults is not None :
		print primordialResults
		ax1.plot (x,primordialResults)  

				
	ax1.grid (True)
	minP = min (processedResults)
	maxP = max (processedResults)
	margin = (maxP - minP) * 0.1
	ax1.set_ylim ((minP - margin,  maxP + margin))
	ax1.set_xlim ((min (x) - max (x) * 0.1,  max (x) * 1.1))
	#ax1.set_ylabel('fitness')
	#ax1.set_xlabel('generation')
	#ax1.set_title ('optimization')

	#print "generation"
	#print generation
	#fig.savefig('/Users/rhett/workspace/genetic1/plots/individualM1.' + str (generation) + '.png', bbox_inches='tight')
	#fig.clf ()
	#showIt = True #self.individual.fitnessRank == 0# False
	#if showIt :
	show ()

class Simulation :

	def __init__ (self,individual) :
		l.indent()
		self.numSamples = 10
		self.currentSample = 0
		self.sampleRange = [0,20]
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

		l.dedent ()

	def postProcess (self) :
		#print "postProcess()"
		#x = [_ * .1 for _ in range (self.sampleRange[0], self.sampleRange[1])]
		self.targetUpperBound = max (self.targetFuncResults)
		self.targetLowerBound = min (self.targetFuncResults)
		self.targetDomain = self.targetUpperBound - self.targetLowerBound

		self.resultsUpperBound = max (self.results)
		self.resultsLowerBound = min (self.results)
		self.resultsDomain = self.resultsUpperBound - self.resultsLowerBound

		if self.resultsDomain != 0.0 :

			self.resultsScaleFactor = self.targetDomain / float (self.resultsDomain)

			if self.resultsScaleFactor != 0.0 :
				self.yOffset = self.targetLowerBound - self.resultsLowerBound
				self.processedResults = [(r-self.resultsLowerBound) * self.resultsScaleFactor + self.targetLowerBound for r in self.results]

			else :
				self.processedResults = [(r-self.resultsLowerBound) + self.targetLowerBound for r in self.results]

		else :
			self.processedResults = [(r-self.resultsLowerBound) + self.targetLowerBound for r in self.results]


	def computeFitness (self) :
		#print "compute fitness"
# 		self.simulation.run ()
		resultSum = 0
		for index,sample in enumerate (self.processedResults) :

			#print self.processedResults
			resultSum += abs (self.targetFuncResults[index] - sample)
			#print abs (self.targetFuncResults[index] - sample)
			#print resultSum
			#print abs (self.targetFuncResults[index] - self.processedResults[index])
		#diff = sum ([abs (self.simulation.processedResults[index] - self.simulation.targetFuncResults[index]) for index in range (len (self.simulation.results))])
		#print "diff = ", diff
		#self.fitness = parameters["fitnessInvert"] - resultSum
		#print "computeFitness resultSum = " + str (resultSum)
		self.individual.fitness = resultSum

	def run (self, generation) :
		l.indent ()
		#print "run () start simulation"

		self.results = []
		self.targetFuncResults = []
		for _sample in range (self.sampleRange[0], self.sampleRange[1]) :

			self.individual.phenotype.execute(inputs = [_sample])

			if len (self.individual.phenotype.realStack) > 0 :
				result = self.individual.phenotype.realStack[-1]

			else :
				result = 0.0

			self.results.append (result)
			self.targetFuncResults.append (func (_sample))

		self.postProcess ()

		self.computeFitness ()


		l.dedent ()



class Individual : #SYNTHESIZE is REQUIRED to populate an individual unless some other function will do it

	# POS = probability of survival
	funcPrec = program.Instruction.funcPrecision
	operTypePrec = program.Instruction.operTypePrec
	registerSetPrec = program.Instruction.registerSetPrecision

	def __init__ (self, parentA=None, parentB=None) :

		l.indent()
		global unusedID

		self.numLines = parameters["numLines"]

		self.phenotype = program.Program (parameters['realMax'], parameters['intMax'])

		if (parentA is None and
			parentB is None ) :
			self.parents = [None, None]

		elif (parentA is None and
			parentB is not None) :
			raise "Individual: ParentA is None"

		elif (parentA is not None and
			parentB is None) :
			raise "Individual: ParentB is None"

		else :
			self.parents = [parentA, parentB]

		self.ID = unusedID

		self.POS = None
		self.fitness = None
		self.fitnessRank = -1
		self.closestNeighborDistance = None
		self.closestNeighborIndex = None
		self.diversity = 0.0
		self.active = True
		self.genotype = None # binary string

		self.simulation = Simulation (self)

		unusedID += 1

		l.dedent ()



	def mutate (self, probabilityOfFeatureMutation, progress) :
		#print "mutate"

		#binaryStr = self.self.phenotype.instructionsToFlatBinary ()
		binaryStr = self.genotype
# 		print "before"
		#print self.genotype
		#lineLength = int (len (binaryStr) / float (len (self.phenotype.instructions)))
		strToList = [bool (int (_)) for _ in binaryStr]

# 		instrList = ''
# 		for line in self.phenotype.instructions :
# 			if line.register2Type is not None :
# 				instrList += str (line.setRegister2) + " "
# 
# 			else :
# 				instrList += "None "


		for _mutation in range (parameters["numMutatePoints"]) :
			bitPlace = int (random.random () * len(binaryStr))
			strToList[bitPlace] = not strToList[bitPlace] 

# 		#print "mutate before"
		#print self.phenotype.instructions
		#print binaryStr
		binaryStr = ''
		for bit in strToList :
			binaryStr += str (int (bit))

		self.genotype = binaryStr
		self.expressGenotypeToPhenotype ()
		#print self.genotype
# 		instrList = ''
# 		for line in self.phenotype.instructions :
# 			if line.register2Type is not None :
# 				#instrList += program.AtomType.strings[line.register2Type] + " "
# 				instrList += str (line.setRegister2) + " "
# 
# 			else :
# 				instrList += "None "

	def expressGenotypeToPhenotype (self) :
		self.phenotype.flatBinaryToInstructions (self.genotype)

def test () :
	#random.seed (0)
	for generation in range (1000) :
		i = Individual ()
		i.synthesize()
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


#test()
