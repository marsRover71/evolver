

import random
import math

from pylab import figure, show

import parameters as p
import individual

def inRange (a, b, rangeIn) :
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

	def __init__ (self) :
		self.generationNum = 1
		self.maxNumGenerations = p.parameters["maxNumGenerations"]
		self.initialPopulationSize = p.parameters["initialPopulationSize"]
		self.maxPopulationSize = p.parameters["maxPopulationSize"]
		self.history = []
		self.individuals = []
		self.apocalypse ()

	def apocalypse (self) :
		print "apocalypse!"
		parentGeneration = [individual.Individual (initGeneration=self.generationNum) for _ in xrange (self.initialPopulationSize * 2)]
		# synthesize the parents
		for individ in parentGeneration :
			individ.phenotype.synthesize ()
			individ.expressGenotypeToPhenotype ()  # required for synthesized features to be converted into instructions

		newIndividuals = [individual.Individual (initGeneration=self.generationNum) for _ in xrange (self.initialPopulationSize * 2)]

		for individ in newIndividuals :
			#random.seed (individ.ID)
			individ.phenotype.synthesize()
			individ.expressGenotypeToPhenotype ()

			# define the parents for the first generation
			separator = int (random.random () * (len (parentGeneration) - 2)) + 1
			parentAIndex = int (random.random () * (separator - 1))
			parentBIndex = int (random.random () * (len (parentGeneration) - separator - 1)) + separator
			if  parentAIndex == parentBIndex :
				raise Exception, "Population.__init__: parentAIndex != parentBIndex; made a clone!!"

			individ.parents = [ parentGeneration[parentAIndex], parentGeneration[parentBIndex] ]

		self.individuals.extend (newIndividuals)
		print "Num initial individuals = " + str (len (self.individuals))
		print "Num initial parents = " + str (len (parentGeneration))

	def runSimulation (self, generation) :
		for indiv in self.individuals :
			indiv.simulation.run (generation)

	def mutate (self, progress) :

		isNotNone = sum (1 for _ in self.individuals if _.fitness is not None)
		if isNotNone :

# 			fitnesses = [_.fitness for _ in self.individuals if _.fitness is not None]
# 			bestFitness = min (fitnesses)
# 			worstFitness = max (fitnesses)

			maxGracePeriod = p.parameters['gracePeriod']
# 			nonMutatedCopies = []
			for index,individual in enumerate (self.individuals) :
				gracePeriod = min (maxGracePeriod, maxGracePeriod - index)

				if (index > 0 and
					individual.fitness is not None and
					(self.generationNum >= individual.initGeneration + gracePeriod )) : # allow genetic lines to hang around for a few generations
# 					newCopy = individual
# 					newCopy.initGeneration = self.generationNum
# 					nonMutatedCopies.append (newCopy)
					individual.mutate (0, progress)

			#self.individuals.extend (nonMutatedCopies)

	def expressGenotypeToPhenotype (self) :
		for individual in self.individuals :
			individual.expressGenotypeToPhenotype ()

	def removeClones (self) :

		# cull out all but one of identical clones
		fitnesses = [_.fitness for _ in self.individuals]

		maxDiff = p.parameters['maxFitnessSimilarity']
		dupCount = [0 for _ in fitnesses]

		for indexX,x in enumerate (fitnesses) :
			for indexY,y in enumerate (fitnesses) :
				if indexX != indexY :
					if abs (x - y) < maxDiff :
						dupCount[indexX] += 1
						dupCount[indexY] += 1
						break

		elites = p.parameters["numElites"]
		kept = []
		for index,count in enumerate (dupCount) :
			if (count == 0 or
				index < elites) :
				kept.append (self.individuals[index])

		self.individuals = kept


	def selection (self) :
		#exclude individuals with lowest fitness where population size is greater than maxPopulationSize
		#rebuild fitness-sorted individuals until max population size is reached

		self.removeClones ()

		# rank in fitness order 
		self.individuals.sort (key= lambda individual: individual.fitness, reverse=(p.parameters["minimizeOrMaximize"]=="maximize"))
		for index,individ in enumerate (self.individuals) :
			individ.fitnessRank = index

		if self.generationNum > 1 :
			individual.plotFunction([0,len (self.individuals[0].simulation.processedResults)],
									self.individuals[0].simulation.processedResults,
									self.individuals[0].simulation.targetFuncResults,
									generation=self.generationNum)


		tempIndividuals = []
		for indiv in self.individuals :
			# when to cull

			if math.isnan (indiv.fitness) :
				print "fitness is nan"
				continue

			if len (tempIndividuals) > self.maxPopulationSize :
				continue

			tempIndividuals.append (indiv)

		self.individuals = tempIndividuals

	def driftArguments (self) :
		for indiv in self.individuals :
			indiv.phenotype.driftArguments ()

	def couple (self, progress) :

		groupSize = min (p.parameters['maxPopulationSize'], len (self.individuals))
		eliteEnd = int (groupSize * 0.25)
		for index in xrange (groupSize) :

			foundDiverseMate = False
			numIter1 = 0
			while (not foundDiverseMate and
					numIter1 < 10 ):

				numIter2 = 0
				eliteSection = index
				while (eliteSection == index and
						numIter2 < 10) :
					eliteSection = int (random.random () * eliteEnd)
					numIter2 += 1

				# randomly select which parent will be the first half and which will be the second half
				if random.random () < 0.5 :
					parentAIndex = index
					parentBIndex = eliteSection

				else :
					parentAIndex = eliteSection
					parentBIndex = index

				parentA = self.individuals[parentAIndex]
				parentB = self.individuals[parentBIndex]

				if progress > 0 :
					foundDiverseMate = parentA.fitness != parentB.fitness

				else :
					foundDiverseMate = True
				numIter1 += 1

			child1 = individual.Individual(parentA,parentB, initGeneration=self.generationNum)
			child2 = individual.Individual(parentA,parentB, initGeneration=self.generationNum)

			parentAChromosome = parentA.phenotype.genotype
			parentBChromosome = parentB.phenotype.genotype

			if len (parentAChromosome) != len (parentBChromosome) :
				raise Exception, "couple: parentA and parentB chromosomes should be the same length but are not."

			dividePoint = int (random.random () * (len (parentAChromosome) - 2) + 1)

			child1Chromosome = parentAChromosome[:dividePoint] + parentBChromosome[dividePoint:]

			child2Chromosome = parentBChromosome[:dividePoint] + parentAChromosome[dividePoint:]

			child1.phenotype.genotype = child1Chromosome
			child1.phenotype.flatBinaryToInstructions (child1Chromosome)

			child2.phenotype.genotype = child2Chromosome
			child2.phenotype.flatBinaryToInstructions (child2Chromosome)

			self.individuals.append (child1)
			self.individuals.append (child2)

		print "Num individuals = " + str (len (self.individuals))

	def run (self) :

		self.runProgress = 0
		for generation in xrange (p.parameters["maxNumGenerations"]) :
			self.generationNum = generation

			if generation % p.parameters["apocalypseCycle"] == p.parameters["apocalypseCycle"] - 1 :
				self.apocalypse ()

# 			##### reproduction #####
			self.couple (self.runProgress)
			self.mutate (1)

			###### lifetime #####
			self.runSimulation (generation)
			self.selection ()

			print "Population size = " + str (len (self.individuals))
			f = [_.fitness for _ in self.individuals]
			print "min fitness = " + str (min(f))
			bestFitness = self.individuals[0].fitness
			print "bestFitness = " + str (bestFitness)
			worstFitness = max (_.fitness for _ in self.individuals if _.fitness is not None)
			print "worstFitness = " + str (worstFitness)
			print "abs (bestFitness - worstFitness) = " + str (abs (bestFitness - worstFitness))
			print "Mean fitness = " + str (sum (_.fitness for _ in self.individuals)/len (self.individuals))

			self.recordGenerationIntoHistory ()
			print "generation = " + str (generation)

			self.runProgress += 1./self.maxNumGenerations

			self.plotFitnesses()

			self.driftArguments()

		individual.plotFunction([0,len (self.individuals[0].simulation.processedResults)],
								self.individuals[0].simulation.processedResults,
								self.individuals[0].simulation.targetFuncResults)

		print self.individuals[0].phenotype #individuals are sorted by fitness, so 0th index is best

	"""
self.history.append ({'bestFitness' : 0.0,
					  #'diversity': diversitySum / len ( self.individuals),
					  'avgFitness': fitnessSum / len ( self.individuals),
					  'fitnesses': fitnesses,
					  'IDs': ids,
					  'genomes': genomes })
	"""

	def recordGenerationIntoHistory (self) :
		fitnesses = [_.fitness for _ in self.individuals]
		ids = [_.ID for _ in self.individuals]

		fitnessSum = sum (fitnesses)

		genomes = [_.genotype for _ in self.individuals]

		self.history.append ({'bestFitness' : 0.0,
							  #'diversity': diversitySum / len ( self.individuals),
							  'avgFitness': fitnessSum / len ( self.individuals),
							  'fitnesses': fitnesses,
							  'IDs': ids,
							  'genomes': genomes,
							  'generation':self.generationNum })

	def plotFitnesses (self) :
		minFitnesses = [min (_['fitnesses']) for _ in self.history]

		fig = figure (1)
		ax1 = fig.add_subplot (1,1,1)
		allFitnesses = []
		allTimes = []
		for time,timeStep in enumerate (self.history) :
			for individ in timeStep['fitnesses'] :
				allFitnesses.append (individ)
				allTimes.append (time)
		ax1.scatter (allTimes, allFitnesses, s=4)

		ax1.grid (True)

		fig.savefig('/Users/rhett/workspace/evolver/plots/individualScatter.' + str (len(self.history)) + '.png', bbox_inches='tight')



population=Population ()
population.run ()
population.plotFitnesses ()
