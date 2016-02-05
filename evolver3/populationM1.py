
#import math
import random
import time
#from scipy import spatial

import parameters as p

import individualM1 as individual

import logM1

# if l doesn't exist, make it otherwise use the existing one
try :
	l

except :
	l = logM1.Log()


def inRange (a, b, rangeIn) :
	print abs (b - a),rangeIn
	return abs (b - a) < rangeIn

def linterp (controlPts, parameter) :
	if controlPts[0] == controlPts[1] :
		return controlPts[0]
	
	else :
		rangeSize = controlPts[1] - controlPts[0]
		return (parameter / float (rangeSize)) + controlPts[0]


minimize = False
maximize = True



class Population :

	# POS = probability of survival

	def __init__ (self) :
		l.pr ("Population __init__")
		l.indent()


		self.generationNum = 1
		self.maxNumGenerations = p.parameters["maxNumGenerations"]
		self.initialPopulationSize = p.parameters["initialPopulationSize"]
		self.maxPopulationSize = p.parameters["maxPopulationSize"]
		parentGeneration = [ individual.Individual () for _ in range (self.initialPopulationSize)]
		# synthesize the parents
		for individ in parentGeneration :
			individ.synthesize ()
			individ.expressGenotypeToPhenotype ()  # required if you need synthesized features converted into instructions
			#print individ.phenotype.instructions
		print parentGeneration
		#exit (0)

		self.individuals = [ individual.Individual () for _ in range (self.initialPopulationSize)]

		for individ in self.individuals :
			individ.synthesize()
			individ.expressGenotypeToPhenotype ()

			# define the parents for the first generation
			separator = int (random.random () * (len (parentGeneration) - 2)) + 1
			parentAIndex = int (random.random () * (separator - 1))
			parentBIndex = int (random.random () * (len (parentGeneration) - separator - 1)) + separator
		#print "parentAIndex = " + str (parentAIndex) + " of range 0-" + str (len(parentGeneration)-1)
			print "parentBIndex = " + str (parentBIndex) + " of range 0-" + str (len(parentGeneration)-1)

			individ.parents = [ parentGeneration[parentAIndex], parentGeneration[parentBIndex] ]

		print "Num initial individuals = " + str (len (self.individuals))
		print "Num initial parents = " + str (len (parentGeneration))

		self.probabilityOfFeatureMutation = p.parameters["probabilityOfFeatureMutation"]

		self.numConvergeLevels = 1
		self.convergeLevelCount = [0 for _ in range (self.numConvergeLevels)]
		self.convergeGenerations = [3 for _ in range (self.numConvergeLevels)]
		self.convergeRange = [100, .001, .01]

		self.runProgress = 0.0

		self.history = []

		self.stallCount = 0

		l.dedent ()


	def recordGenerationIntoHistory (self) :
		fitnesses = [_.fitness for _ in self.individuals]
		ids = [_.ID for _ in self.individuals]
		
# 		maxFitness = max (fitnesses)
# 		minFitness = min (fitnesses)
		fitnessSum = sum (fitnesses)
		#diversitySum = sum ([0 for _ in self.individuals])

		genomes = [_.genotype for _ in self.individuals]

		self.history.append ({'bestFitness' : 0.0,
							  #'diversity': diversitySum / len ( self.individuals),
							  'avgFitness': fitnessSum / len ( self.individuals),
							  'fitnesses': fitnesses,
							  'IDs': ids,
							  'genomes': genomes })

# 	def writeFitnessViz (self, generation) :
# 		filename = parameters["vizFileBasename"] + "." + str (generation) + ".obj"
# 		fileHandle = open (filename, "w")
# 		for individual in self.individuals :
# 			fileHandle.write ('v ' + str (individual.features[0].value) + ' ' + str (individual.fitness) + ' ' + str (individual.features[1].value) + '\n')
# 		fileHandle.close ()

	def convertFitnessToPOS (self, diversityWeight) :
		#diversityWeight=1
		
		#diversityMult = 40
		#l.pr ("Population convertFitnessToPOS")
		fitnesses = [_.fitness for _ in self.individuals]
		#print fitnesses
		fitnessSum = sum (fitnesses)

# 		diversities = [_.diversity for _ in self.individuals]
# 		diversitySum = sum (diversities) * diversityMult

		#combinedSum = fitnessSum + diversitySum

		for individual in self.individuals :
			if fitnessSum > 0 :
				#individual.POS = (individual.fitness + individual.diversity * diversityMult * diversityWeight) / combinedSum
				individual.POS = individual.fitness# / fitnessSum

			else :
				print "Fitness of all individuals is 0"
				exit (0)
				individual.POS = 1.0 / len (self.individuals)


# 	def distanceToClosestParameterSpaceIndividual (self) :
# 		points = [(individ.features[0].value,individ.features[1].value) for individ in self.individuals]
# 
# 		#numDimensions = len (self.individuals[0].features)
# 		kdt = spatial.KDTree(points)
# 
# 		neighbors = kdt.query_ball_point(points, r=parameters["diversityTestRadius"]) #, return_distance=False)
# 
# 		assert len (neighbors) == len (self.individuals), "k-D tree query returned incorrect number of neighbors"
# 		for index,individ in enumerate (self.individuals) :
# 			numNeighbors = len(neighbors[index])
# 			if numNeighbors > 1 :
# 				individ.diversity =  math.sqrt (numNeighbors) / numNeighbors
# 				#print numNeighbors,individ.diversity
# 
# 			else :
# 				individ.diversity = 1.0

	def runSimulation (self, generation) :
		#l.pr ("Population runSimulation")
		l.indent ()
		for indiv in self.individuals :

			#individual.simulation.initialize ()
			#if individual.active :
			indiv.simulation.run (generation)
# 			if indiv.ID == 0 :
# 				print indiv.simulation.processedResults
# 			individual.plotFunction(indiv.simulation.sampleRange,
# 								indiv.simulation.processedResults)

			#individual.computeFitness ()
			#individual_fitness = individual.fitness
			#individual_parents = [individual.parents[0].ID, individual.parents[1].ID ]

		# delete inactive individuals
# 		newList = []
# 		for individual in self.individuals :
# 			if (individual.active or
# 				random.random () < 0.5) :
# 				newList.append (individual)
# 				
# 
# 			else :
# 				print "delete inactive"
# 				
# 		self.individuals = newList

		l.dedent ()

# 	def crossover (self, top10Fitness, progress) :
# 		#l.pr ("Population crossover")
# 		for individual in self.individuals :
# 			individual.crossover (top10Fitness, progress)

	def mutateLifeEvent (self, progress) :
		#l.pr ("Population mutate")
		for index,individual in enumerate (self.individuals) :
			#if individual.active :
			if index/float (len (self.individuals)) > .05 : #elitism
				individual.mutate (linterp ([2,1], progress), progress)

	def expressGenotypeToPhenotype (self) :
		#l.pr ("Population expressGenotypeToPhenotype")
		l.indent ()
		for individual in self.individuals :
			#print "len (individual.phenotype.instructions) = "
			#print len (individual.phenotype.instructions)
			#print individual.phenotype.instructions
			
			#if individual.active :
			individual.expressGenotypeToPhenotype ()

		l.dedent ()

	def selection (self) :
		l.pr ("Population selection")
		l.indent ()
		#exclude individuals with lowest fitness where population size is greater than maxPopulationSize
		#rebuild fitness-sorted individuals until max population size is reached

		# rank in fitness order 
		self.individuals.sort (key= lambda individual: individual.fitness, reverse=(p.parameters["minimizeOrMaximize"]))
		for index,individ in enumerate (self.individuals) :
			individ.fitnessRank = index

		fitnesses = [_.fitness for _ in self.individuals]
# 		print "??itnesses
		#print [_.fitnessRank for _ in self.individuals]
		#print "POS sorted"
		#print [_.POS for _ in self.individuals]
		if p.parameters["minimizeOrMaximize"] == minimize :
			print "Fitnesses sorted smallest to largest"
			print fitnesses
			print "Minimum fitness = " + str (min (fitnesses))

		elif p.parameters["minimizeOrMaximize"] == maximize :
			print "Fitnesses sorted largest to smallest"
			print fitnesses
			print "Maximum fitness = " + str (min (fitnesses))

		else :
			print 'parameters["minimizeOrMaximize"] set to bad value'

		if (self.generationNum == p.parameters["maxNumGenerations"] - 1 or
			min (fitnesses) < p.parameters["fitnessTarget"] ):
			individual.plotFunction([0,len (self.individuals[0].simulation.processedResults)], self.individuals[0].simulation.processedResults, self.individuals[0].simulation.targetFuncResults)

		print "growth pop size = " + str (len (self.individuals))
		tempIndividuals = []
		for indiv in self.individuals :
			if len (tempIndividuals) == self.maxPopulationSize :
				break

			#if individual.active :
			tempIndividuals.append (indiv)

		self.individuals = tempIndividuals
		#individual.plotFunction([0,len (self.individuals[0].simulation.processedResults)], self.individuals[0].simulation.processedResults)
		#print self.individuals[0].genotype
		print self.individuals[0].phenotype.instructions[0].funcString
		print self.individuals[0].fitness

		print "cull pop size = " + str (len (self.individuals))
		l.dedent ()

	def couple (self, progress) :
		#l.pr ("Population couple")
		l.indent ()

		groupSize = len (self.individuals)
		for index in range (groupSize) :


			parentAIndex =  index
			parentBIndex =  int (random.random () * groupSize * .125)   # select a random mate

			#for c in self.individuals :
				#print "phenotype len = " + str (len(c.phenotype.instructions))

			#print "parentAIndex = " + str (parentAIndex)
			#print "parentBIndex = " + str (parentBIndex)
			#print "groupSize    = " + str (groupSize   )

			parentA = self.individuals[parentAIndex]
			parentB = self.individuals[parentBIndex]

			#print len (parentA.phenotype.instructions)
			#print len (parentB.phenotype.instructions)

			child1 = individual.Individual(parentA,parentB)
			child1.synthesize ()
			child2 = individual.Individual(parentA,parentB)
			child2.synthesize ()

			parentAChromosome = parentA.genotype #getFlatBinary ()

			parentBChromosome = parentB.genotype #getFlatBinary ()
			#print "parentBChromosome"
			#print "len parentBChromosome = "
			#print len (parentBChromosome)


			dividePoint = int (random.random () * (len (parentAChromosome) - 2) + 1)
			#divide point is in range of [1,bitDepth-2]  i.e. for a 16 bit value, the range is [1,14]
			# this leaves 0, and 15 outside of the divide point range
			#print "dividePoint = " + str (dividePoint) + "/" + str (len (parentAChromosome)) + " parent b len = " + str (len (parentBChromosome))
			#print "parentA"
			#print parentAChromosome
			
			#print "parentB"
			#print parentBChromosome
			child1Chromosome = parentAChromosome[:dividePoint] + parentBChromosome[dividePoint:]
			#print parentAChromosome[:dividePoint]
			#print parentBChromosome[dividePoint:]

			child2Chromosome = parentBChromosome[:dividePoint] + parentAChromosome[dividePoint:]

			#print "dividePoint                = " + str (dividePoint            )
			#print "len (parentAChromosome)    = " + str (len (parentAChromosome))
			#print "len (parentBChromosome)    = " + str (len (parentBChromosome))
			#print "len (child1Chromosome)     = " + str (len (child1Chromosome) )
			#print "len (child2Chromosome)     = " + str (len (child2Chromosome) )

			child1.genotype = child1Chromosome
			child1.phenotype.flatBinaryToInstructions (child1Chromosome)
			#print "child1"
			#print child1Chromosome
			#print child1
			#print child1.phenotype
			#print child1.phenotype.instructions

			child2.genotype = child2Chromosome
			child2.phenotype.flatBinaryToInstructions (child2Chromosome)
			#print "child2"
			#print child2
# 			print child2.phenotype
			#print child2.phenotype.instructions
			self.individuals.append (child1)
			self.individuals.append (child2)

		print "Num individuals = " + str (len (self.individuals))

		l.dedent ()

	def diversityExplosion (self) :
		print "diversity explosion"
		numIterations = 20 #int (parameters["featureBitDepth"] / 2)
		for _iter in range (numIterations) :
			self.mutateLifeEvent (0)

# 	def runHighestFitness (self) :
# 		bestFitness = 0
# 		bestGenome = None
# 		bestFitnessGeneration = -1
# 		for generationIndex,generation in enumerate (self.history) :
# # 			if ((parameters["minimizeOrMaximize"] == minimize and generation["bestFitness"] < bestFitness) or
# # 			    (parameters["minimizeOrMaximize"] == maximize and generation["bestFitness"] > bestFitness)):
# 			if generation["bestFitness"] < bestFitness :
# 				bestFitness = generation["bestFitness"]
# 				print "bestFitness"
# 				print bestFitness
# 				
# 				bestFitnessGeneration = generationIndex
# 				for genomeIndex in range (len (generation["genomes"])) :
# 					if generation["fitnesses"][genomeIndex] == bestFitness :
# 						bestGenome = generation["genomes"][genomeIndex]
# 
# 		return (bestFitnessGeneration, bestFitness, bestGenome)

	def convergence (self) : #, progress) :

		for level in range (self.numConvergeLevels) :
			if len (self.history) > 2 :
				converged = inRange (self.history[-1]["avgFitness"], self.history[-2]["avgFitness"], self.convergeRange[level])
				print "converged = " + str (converged)
			else :
				converged = False

			if converged :
				self.convergeLevelCount[level] += 1
				print "converged"

			else :
				self.convergeLevelCount[level] = 0

			if self.convergeLevelCount[level] > self.convergeGenerations[level] :
				self.runProgress= 0
				self.convergeLevelCount[level] = 0
				self.diversityExplosion ()

	def  stallReset (self) :
		self.history
		if len (self.history) > 2 :
			minFitnessesOne = min (self.history[-1]['fitnesses'])
			minFitnessesTwo = min (self.history[-2]['fitnesses'])

			print "stalled _______"
			print self.history[-1]["avgFitness"], self.history[-2]["avgFitness"]
			stalled = inRange (minFitnessesOne, minFitnessesTwo, .01)
			print stalled
			time.sleep (1)
		else :
			stalled = False

		if stalled :
			self.stallCount += 1
			print ">>>>>>> stallCount = " + str (self.stallCount)
			if self.stallCount > 5 :
				print "stallReset: stall reset!"
				self.stallCount = 0
				self.diversityExplosion ()

	def run (self) :
		#l.pr ("Population run")
		l.indent ()
		# population -> mutation -> crossover -> GtoP -> fitness -> convertFitnessToPOS -> selection -> new generation


		self.convergeLevelCount = [0 for _ in range (self.numConvergeLevels)]

		self.runProgress = 0

		#reduce stepsize as generations increase
		#select by best fitness and diversity rank
		for generation in range (parameters["maxNumGenerations"]) : #self.maxNumGenerations) :
			l.prV ("generation")
			l.indent ()
			self.generationNum = generation
# 			##### reproduction #####
# 			# selection
			self.couple (self.runProgress)
# 
# 			# mutate
			self.mutateLifeEvent (self.runProgress)
# 
# 			# genotype to phenotype
			#self.expressGenotypeToPhenotype()
# 		 	sim = Simulation (i)
# 		 	random.seed (gen)
# 		 	sim.run (gen)

			###### lifetime #####
			# simulate 
			self.runSimulation (generation)

			# similarity in parameters
			#self.distanceToClosestParameterSpaceIndividual()

# 			# probability of survival (sum to 1)
			#self.convertFitnessToPOS (max (0, 0.5 - float (generation)/self.maxNumGenerations))
# 
# 			# selection
			self.selection ()
# 
# 			# vizualization
# 			self.writeFitnessViz (generation)
# 
			self.recordGenerationIntoHistory ()
# 
			self.runProgress += 1./self.maxNumGenerations
# 
# 			# convergence
			self.convergence ()

			# stall
			self.stallReset ()

			l.dedent ()

# 		bestFitnessGeneration,bestFitness,bestGenome = self.runHighestFitness ()
# 		print "bestFitnessGeneration"
# 		print bestFitnessGeneration
# 		print "bestFitness"
# 		print bestFitness
# 		print "bestGenome"
# 		print bestGenome

	"""
self.history.append ({'bestFitness' : 0.0,
					  #'diversity': diversitySum / len ( self.individuals),
					  'avgFitness': fitnessSum / len ( self.individuals),
					  'fitnesses': fitnesses,
					  'IDs': ids,
					  'genomes': genomes })
	"""
	
	def plotFitnesses (self) :
		minFitnesses = [min (_['fitnesses']) for _ in self.history]
		avgFitnesses = [_['avgFitness'] for _ in self.history]
		#print minFitnesses
		individual.plotFunction ([0,len(minFitnesses)], minFitnesses, avgFitnesses)
#individual.Individual()
p=Population ()
p.run ()
p.plotFitnesses ()
# lowestFitness = 1e+38
# lowestFitnessIndex = -1
# for index,indiv in enumerate (p.individuals) :
# 	if indiv.fitness < lowestFitness :
# 		lowestFitness = indiv.fitness
# 		lowestFitnessIndex = index

#print "lowest fitness  " + str (lowestFitness)
# topIndiv = individual.Individual ()
# topIndiv.genotype = p.history[lowestFitnessIndex]['genomes']
# topIndiv.expressGenotypeToPhenotype ()
# topIndiv.simulation.run (0)
# print "Primordial"
# 
# print topIndiv.simulation.processedResults
print "Num final individuals = " + str (len (p.individuals))
#individual.plotFunction ([0,len(topIndiv.simulation.processedResults)], topIndiv.simulation.processedResults, targetFuncResults=topIndiv.simulation.targetFuncResults)

