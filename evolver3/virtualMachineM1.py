
import random
import math
import types
import time
import numpy as np
import parameters as p

from pylab import figure, show

import atomM1 as atom

import sys

sys.path.append ('/Users/rhett/workspace/evolver3')

import logM1


#random.seed (0)
# if l doesn't exist, make it otherwise use the existing one
try :
	l

except :
	l = logM1.Log()


class Random :
	random.seed (0.77777777)
	#randValues =[(random.randint(),random.random ()) for _ in xrange (numRandValues)]

	def __init__ (self, numValues) :
		self.randomIndex = 0
		self.numRandValues = numValues
		self.randValues = [random.random () for _ in xrange (self.numRandValues)]
		#print self.randValues

	def getRandomValue (self) :
		self.randomIndex += 1
		return self.randValues[self.randomIndex % self.numRandValues]

	def setRandomSeed (self, randomIndexIn) :
		self.randomIndex = int (randomIndexIn) % self.numRandValues

r = Random (numValues=1000000)
# for _ in xrange (20) :
# 	print r.getRandomValue ()
# 
# exit(0)

def func (x) :
	return .5 +   (.2 * x) +     0.4 *   (x ** 3.0)
	#return 0.4 * (x ** 3.0)

def instructionExecuteFragment (self, instr) :
	opCode = instr.opCode
	fragment = []
	if Instruction.specs[opCode]['arity'] == InstructionOperandType.nullary :
		if opCode == Instruction.popToOperand :
			print "instructionExecuteFragment: note: no fragment generated for pop2Oper0 or pop2Oper1"

	elif Instruction.specs[opCode]['arity'] == InstructionOperandType.unary :
		fragment.append (Instruction (Instruction.popToOperand,
												argument=0))

	elif Instruction.specs[opCode]['arity'] == InstructionOperandType.binary :
		fragment.append (Instruction (Instruction.popToOperand,
												argument=0))
		fragment.append (Instruction (Instruction.popToOperand,
												argument=1))

	else :
		print "instructionExecuteFragment: invalid InstructionOperandType"

	fragment.append (instr)

	return fragment

class InstructionOperandType :
	nullary = 0
	unary = 1
	binary = 2

	strings = [ "nullary",
				"unary",
				"binary"]

	precision = atom.BinaryString().rangeToPrecision ([0,len (strings)])

	def toStr (self, InstructionArgumentIn) :
		assert InstructionArgumentIn < len (self.strings), "toStr: InstructionArgumentIn not found in InstructionOperandTypes.strings."
		return self.strings[InstructionArgumentIn]

	def toInstructionOperandType (self, stringIn) :
		assert stringIn in self.strings, "toInstructionOperandType: stringIn not found in InstructionOperandType.strings."
		return self.strings.index (stringIn)


class Instruction :
	# instructions that take an int argument (not to be confused with operands)
	assign = 0
	pushFromSym = 1
	pushFromInput = 2
	popToOperand = 3

	# instructions that take an real or int argument (not to be confused with operands)
	pushConst = 4

	# instructions that take no argument (not to be confused with operands)
	pushAllSym = 5
	popToReturn = 6
	noOp = 7
	rand = 8
	add = 9
	sub = 10
	mult = 11
	div = 12
	pow = 13
	sin = 14
	cos = 15

	expressionInstructions = [	
								pushFromSym,
								pushFromInput,
								rand,
								add,
								sub,
								mult,
								div,
								pow,
								sin,
								cos
							]
	# other instructions: tan, max, min, clamp, abs, sqrt, rand, log10, logE, log2, mod, 
	strings =  ["assign",
				"pushFromSym", 
				"pushFromInput",
				"popToOperand",
				"pushConst",
				"pushAllSym",
				"popToReturn",
				"noOp",
				"rand",
				"add", 
				"sub", 
				"mult", 
				"div", 
				"pow", 
				"sin", 
				"cos"]


	opCodeTypeRange = [0,len (strings)]
	opCodePrecision = atom.BinaryString().rangeToPrecision (opCodeTypeRange)
	#print "opCodeTypeRange = " + str (opCodeTypeRange)
	#print "opCodePrec = " + str (opCodePrecision)

	argumentPrecision = 32

	def toStr (self, intValueIn) :
		return self.Instr[intValueIn]

	def toInstruction (self, stringIn) :
		assert stringIn in self.strings, "toInstruction: stringIn not found in Instruction.strings."
		return self.strings.index (stringIn)

	def isValid (self, instr) :
		return (type (instr) == types.IntType and
				instr < len (self.strings))

	specs =	{assign:		{'arity': InstructionOperandType.unary,
							'oper0Type': atom.AtomType.any, #value to set
							'oper1Type': None, 
							'argType': atom.AtomType.int,
							'resultType': None }, #symbol index to set

			pushFromSym:	{'arity': InstructionOperandType.unary,
							'oper0Type': atom.AtomType.int,  # why are there 2 int inputs?
							'oper1Type': None,
							'argType': atom.AtomType.int,
							'resultType': atom.AtomType.any  },#symbol index to push

			pushFromInput:	{'arity': InstructionOperandType.nullary,
							'oper0Type': None,
							'oper1Type': None,
							'argType': atom.AtomType.int,
							'resultType': atom.AtomType.any },  #input index to push

			popToOperand:	{'arity': InstructionOperandType.nullary,
							'oper0Type': None,
							'oper1Type': None,
							'argType': atom.AtomType.int,
							'resultType': None  }, # index of operand to set (0 or 1) from the pop'ed value

			pushConst:		{'arity': InstructionOperandType.nullary,
							'oper0Type': None,
							'oper1Type': None,
							'argType': atom.AtomType.any,
							'resultType': atom.AtomType.any  }, #const value

			pushAllSym:		{'arity': InstructionOperandType.nullary,
							'oper0Type': None,
							'oper1Type': None,
							'argType': None,
							'resultType': None },

			popToReturn:	{'arity': InstructionOperandType.nullary,
							'oper0Type': None,
							'oper1Type': None,
							'argType': None,
							'resultType': None },

			noOp:			{'arity': InstructionOperandType.nullary ,
							'oper0Type': None,
							'oper1Type': None,
							'argType': None,
							'resultType': None},

			rand:			{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.real, # range min
							'oper1Type': atom.AtomType.real, # range max
							'argType': None,
							'resultType': atom.AtomType.real },

			add:			{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any },

			sub:			{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any },

			mult:			{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any},

			div:			{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any },

			pow:			{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any },

			sin:			{'arity': InstructionOperandType.unary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': None,
							'argType': None,
							'resultType': atom.AtomType.any },

			cos:       	{'arity': InstructionOperandType.unary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': None,
							'argType': None,
							'resultType': atom.AtomType.any }}


	#instrTypeRange = [0, len (strings)]
	operTypeRange = [0,len (atom.AtomType.strings)]
	operTypePrec = atom.BinaryString().rangeToPrecision (operTypeRange)

	def __init__ (self, opCodeIn=None, argument=None) :

		if opCodeIn is None :

			return

		else :
			if opCodeIn not in self.specs.keys () :
				print ("Instruction.__init__() unknown opCode '" +
						str (opCodeIn) +
						"'.")
				exit(0)


			self.opCode = opCodeIn # expects an int

			###### validate argument value against spec type #########
			if self.specs[self.opCode]['argType'] == None :
				if argument is not None :
					print ("Instruction.__init__() argument should be None, but was '" +
							str (argument) +
							"' for opCode " +
							self.strings[opCodeIn] +
							".")
					exit(0)

				else :
					self.argument = None

			elif self.specs[self.opCode]['argType'] == atom.AtomType.any :
				self.argument = argument

			elif self.specs[self.opCode]['argType'] == atom.AtomType.int :
				if type (argument) != types.IntType :
					print ("Instruction.__init__() argument should be int, but was '" +
							str (argument) +
							"' for opCode " +
							self.strings[opCodeIn] +
							".")
					exit(0)

				else :
					self.argument = argument

			elif self.specs[self.opCode]['argType'] == atom.AtomType.real :
				if type (argument) != types.FloatType :
					print ("Instruction.__init__() argument should be real, but was '" +
							str (argument) +
							"' for opCode " +
							self.strings[opCodeIn] +
							".")
					exit(0)

				else :
					self.argument = argument

			else :
				print ("Instruction.__init__() unknown argType in specs for opCode '" +
						str (self.specs[self.opCode]['name']) +
						"'.")
				exit(0)


	def repairArgument (self, opCode, argumentValue, argumentType) :


	
# 		print "repair opCode = " + str (self.strings[opCode.toInt ()])
# 		print "opCode to int" + str (opCode.toInt ())
# 		print "repair argVal in = " + str (argumentValue)
		if opCode.toInt () == self.assign :

			#print "assign"
			if type (argumentValue) != types.IntType :
				#print "argVal"
				argumentValue = int (random.random () * p.parameters['maxSymbols'])

		elif opCode.toInt () == self.pushFromSym :
			if type (argumentValue) != types.IntType :
				argumentValue = int (random.random () * p.parameters['maxSymbols'])

		elif opCode.toInt () == self.pushFromInput :
			if type (argumentValue) != types.IntType :
				argumentValue = int (random.random () * p.parameters['maxSymbols'])

		elif opCode.toInt () == self.popToOperand :
			if type (argumentValue) != types.IntType :
				numOperands = 2
				argumentValue = int (random.random () * numOperands)

		elif opCode.toInt () in [	self.noOp,
									self.popToReturn,
									self.rand,
									self.add, 
									self.sub, 
									self.mult,
									self.div, 
									self.pow, 
									self.sin, 
									self.cos] :
			argumentValue = None

		elif opCode.toInt () == self.pushConst :
			if argumentValue is None :
				if argumentType == atom.AtomType.real :
					argumentValue = random.random () * p.parameters['realMax']

				elif argumentType == atom.AtomType.int :
					argumentValue = int (random.random () * p.parameters['intMax'])

		#print "repair argVal out = " + str (argumentValue)

		return argumentValue

class Stack :
	def __init__ (self, allocatedSize) :
		self.pointer = -1
		self.allocatedSize = allocatedSize
		self.valueArray = np.array(np.zeros (self.allocatedSize), dtype=np.float_)
		self.typeArray = np.array(np.zeros (self.allocatedSize), dtype=np.int_)

	def length (self) :
		return self.pointer + 1

	def pop (self) :
		if self.pointer >= 0 :
			resultType = self.typeArray[self.pointer]
			resultValue = atom.AtomType ().coerce (self.valueArray[self.pointer], resultType)

			self.pointer -= 1
			return (resultValue, resultType)

		else :
			print "Stack:pop () trying to pop from an empty stack."

	def push (self, item) :
		if (self.pointer < self.allocatedSize or
		    self.pointer == -1) :
			#print "Stack.push " + str (item)
			self.pointer += 1
			#print "new pointer = " + str (self.pointer)
			self.valueArray[self.pointer] = float (item)
			self.typeArray[self.pointer] = atom.AtomType ().getValueType (item)

		else :
			print "Stack:push () Overflow. Trying to push beyond the end of a stack."


	def printValues (self) :
		#print "Pointer = " + str (self.pointer)
		if self.pointer > -1 :
			print 'Stack ['
			for index in range (self.pointer+2) :
				print self.valueArray[index], atom.AtomType ().strings[self.typeArray[index]]
			print ']'

		else :
			print 'Stack []'

# print "before"
# s = Stack (1000000)
# print "allocated"
# for _ in range (9900) :
# 	s.push (float (_))
# print "pushed"
# for _ in range (9800) :
# 	s.pop ()
# print "done"
# s.printValues ()
# exit(0)
overflowMessagesOn = True

class Program :
	def __init__ (self, randomSeed, expressionSize) :

		self.instructions =   []
		self.symbolTable = []
		self.symbolTopIndex = 0

		self.expressionSize = expressionSize
		self.inputs = []

		# set up parameters as symbols
# 		self.symbolTable.append (p.parameters["numInstructions"]) # number of instructions in the program
#
		self.machineStack = Stack (10000)
		self.programCounter = 0

		self.operand0 = None
		self.operand1 = None
		self.returnValue = None

		self.realMax = p.parameters['realMax']
		self.intMax = p.parameters['intMax']

		self.randomSeed = randomSeed
		self.binaryBreakdown = []

	def randomInstruction (self) :
		numSpecs = len (Instruction.specs)
		setVarSpecs = 3

		choice = int (random.random () * (numSpecs - setVarSpecs)) + setVarSpecs
		key = Instruction.specs.keys ()[choice]
		return key

	def synthesize (self) :
		#print "synthesize "

		random.seed (time.time ())
		#print "Random seed " + str (self.randomSeed)
		#time.sleep (1)
		self.instructions = []

		numConstants = 3
		numWorkingSymbols = 4
		numInputs = p.parameters["numInputs"]
		numSymbols = numInputs + numWorkingSymbols + numConstants

		###### instructions to populate the symbol table with input variables #####
		index = 0
		for count in range (numInputs) :

			# push value from inputs[index] onto the stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushFromInput,
														argument=index))
			# assign the stackTop value to symbol[index]
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index))
			#print "assign input " + str (index) + " to symbol #" + str (index)

			index += 1



		###### instructions to populate the symbol table with constants ######
		pythagoras=math.sqrt (2.0)
		theodorus=math.sqrt (3.0)
		sqrt5=math.sqrt (5.0)
		mascheroni=0.5772156649
		goldenRatio=1.5180339887
		bernstein=0.2801694990
		gauss=0.3036630028
		landau=0.5
		omega=0.5671432904
		sierpinski=2.5849817595
		recipFib=3.3598856662

		consts = [0.0,1.0,math.pi,math.e,pythagoras,theodorus,sqrt5,mascheroni,goldenRatio,bernstein,gauss,landau,omega,sierpinski,recipFib]

		for const in consts :
			# push const onto stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushConst,
														argument=const))
			# assign TOS to a symbol at index
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index))

			#print "assign const " + str (const) + " to symbol #" + str (index)
			index += 1


		###### instructions to populate the symbol table with working variables (init to random) #####
		for count in range (numInputs,numWorkingSymbols) :

			# push value onto stack
			randValue = random.random () * p.parameters["realMax"]
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushConst,
														argument=randValue))
			# assign TOS to a symbol at index
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index))
			#print "assign rand " + str (randValue) + " to symbol #" + str (index)
			index += 1

		###### instruction to push all symbols onto the stack #####
		self.instructions.append (Instruction (   opCodeIn=Instruction.pushAllSym))


		###### instruction to push the input onto the stack ######
		self.instructions.append (Instruction (   opCodeIn=Instruction.pushFromInput, 
														argument=0))

		self.numSetupLines = len (self.instructions)
		sizeOfSymTable = index

		self.numLines = p.parameters["numInstructions"]
		# SYNTHESIZE random instructions
		while len (self.instructions) - self.numSetupLines < self.numLines :

			#print "____________________"
			# instruction
			opCode = int (random.random () * len (Instruction.strings))
# 			
# # 								Instruction.pushFromInput,
# # 								Instruction.pushFromInput,
# # 								Instruction.div,
# 			opCodeTestList = (
# 								Instruction.assign,
# 								Instruction.pushFromSym, 
# 								Instruction.pushFromInput,
# 								Instruction.pushFromInput,
# 								Instruction.pushFromInput,
# 								Instruction.pushFromInput,
# 								Instruction.pushFromInput,
# 								Instruction.pushFromInput,
# 								#Instruction.popToOperand,
# 								#Instruction.pushConst,
# 								#Instruction.pushAllSym,
# 								Instruction.popToReturn,
# 								#Instruction.pushFromInput,
# 								Instruction.add,
# 								Instruction.sub,
# 								Instruction.mult,
# 								Instruction.pow,
# 								Instruction.sin,
# 								Instruction.cos)
# 			opCode = opCodeTestList[int (random.random () * len (opCodeTestList))]
			print "opCode = " + str (Instruction.strings[opCode])
			preFragment = None

			##### set the argument ####
			argument = None
			if Instruction.specs[opCode]['arity'] == InstructionOperandType.nullary :
				#print 'a'
				if opCode == Instruction.popToOperand :
					#print 'b'
					numOperands = 2
					index = int (random.random () * numOperands)
					argument = index
					#print "synthesize: popToOperand: argument = " + str (argument)
					#time.sleep (2)
					preFragment = []
					#print "synthesize: no fragment generated for popToOperand"

				elif opCode == Instruction.pushConst :
					#print 'c'
					# randomly choose whether the type should be int or real
					rhsType = [atom.AtomType.real, atom.AtomType.int][int (random.random () * 2)]
					if rhsType == atom.AtomType.real :
						argument = random.random () * p.parameters["realMax"]

					elif rhsType == atom.AtomType.int :
						argument = int (random.random () * p.parameters["intMax"])

					else :
						print "synthesize: pushConst unknown constant type"

					preFragment = []
# 					# push the rhs of the assignment onto the stack to prepare for the assign instructions
# 					preFragment.append (Instruction (Instruction.pushConst,
# 													argument=rhs))
# 
# 
# 					preFragment.append (Instruction (Instruction.popToOperand,
# 													argument=0))

				elif opCode == Instruction.pushFromInput :
					#print 'd'
					argument = int (random.random () * p.parameters["numInputs"])
					#print "pushFromInput: argument"
					#print argument
					preFragment = []

				elif opCode == Instruction.noOp :
					preFragment = []
					#print 'e'
					pass

				elif opCode == Instruction.pushAllSym :
					preFragment = []
					#print 'f'
					pass

				elif opCode == Instruction.popToReturn :
					preFragment = []
					#print 'k'
					pass

				else :
					print "unsupported nullary instruction " +str ( Instruction.strings[opCode])
					exit(0)

			elif Instruction.specs[opCode]['arity'] == InstructionOperandType.unary :
				#print 'g'
				preFragment = []#[Instruction (Instruction.popToOperand,
											#	argument=0)]

				if opCode == Instruction.pushFromSym :
					#print 'h'
					# set the argument to the index of the symbolTable to which the value will be assigned
					index = int (random.random () * sizeOfSymTable)
					argument = index

				elif opCode == Instruction.assign :
					#print 'i'
					#print "Instruction.assign"
					# randomly choose whether the type should be int or real
					rhsType = [atom.AtomType.real, atom.AtomType.int][int (random.random () * 2)]
					if rhsType == atom.AtomType.real :
						rhs = random.random () * p.parameters["realMax"]

					elif rhsType == atom.AtomType.int :
						rhs = int (random.random () * p.parameters["intMax"])

					# push the rhs of the assignment onto the stack to prepare for the assign instructions
					# this value will get popped off the stack by the assign instruction
					preFragment.append (Instruction (Instruction.pushConst,
													argument=rhs))

					# set the argument to the index of the symbolTable to which the value will be assigned
					lhs = int (random.random () * sizeOfSymTable)
					argument = lhs

					#print InstructionOperandType.strings[rhsType]
					#print "rhs"
					#print rhs

					#print "lhs"
					#print lhs

					#print "___________"

# 				else :
# 					print "unsupported unary instruction " +str ( Instruction.strings[opCode])
# 					exit(0)


			elif Instruction.specs[opCode]['arity'] == InstructionOperandType.binary :
				#print 'j'
				preFragment = []
				pass
# 				preFragment.append (Instruction (Instruction.popToOperand,
# 												argument=0))
# 				preFragment.append (Instruction (Instruction.popToOperand,
# 					 							argument=1))

			else :
				print "synthesize: invalid InstructionOperandType"

			instr = Instruction (opCode,
								argument=argument)

			self.instructions.extend (preFragment)
			self.instructions.append (instr)

		print "synth numLines = " + str (self.numLines)
		print "numInstructions = " + str ( len (self.instructions) - self.numSetupLines)
		print "total num instructions = " + str ( len (self.instructions))

		if len (self.instructions) - self.numSetupLines > self.numLines :
			self.instructions = self.instructions[:self.numLines + self.numSetupLines]
		#newprint "synthesize..."
		#print self
		#exit(0)

		self.genotype = self.instructionsToFlatBinary ()
		#self.instructions = []

	def __repr__ (self) :
		result = ''
		for line,instr in enumerate (self.instructions) :

			oper0Type = Instruction.specs[instr.opCode]['oper0Type'] 
			oper1Type = Instruction.specs[instr.opCode]['oper1Type']
			resultType = Instruction.specs[instr.opCode]['resultType']

			if oper0Type is not None :
				oper0TypeString = atom.AtomType.strings[oper0Type]

			else :
				oper0TypeString = 'None'

			if oper1Type is not None :
				oper1TypeString = atom.AtomType.strings[oper1Type]

			else :
				oper1TypeString = 'None'

			if resultType is not None :
				resultTypeString = atom.AtomType.strings[resultType]

			else :
				resultTypeString = 'None'

			result += (("%04i" % line) +
					("   %-7s" % oper0TypeString) + 
					" " +
					("%-7s" % oper1TypeString) +
					" -> " +
					("%-15s" % Instruction.strings [instr.opCode]) +
					" (" +
					("%-15s" % str (instr.argument)) +
					") -> " +
					("%-7s" % resultTypeString) +
					'\n')
		return result

	def popOperValue (self, operandNumber, instruction) :
		#take the operand type and pop off the appropriate stack

		global overflowMessagesOn

		if operandNumber == 0 :
			oper0Type = Instruction ().specs[instruction.opCode]['oper0Type']
			#print "oper 0 popOperValue type = " + str (instruction.operand0Type)
			if oper0Type == atom.AtomType.real :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (float (pop[0]), oper0Type)

			elif oper0Type == atom.AtomType.int :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (int (pop[0]), oper0Type)

			elif oper0Type == atom.AtomType.bool :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (bool (pop[0]), oper0Type)

			elif oper0Type == atom.AtomType.any :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (pop[0], atom.AtomType ().getValueType (pop[0]))

			else :
				if overflowMessagesOn :
					print "Program: popOperValue: instruction.operand0Type is unknown atom.AtomType " + str(oper0Type)

		elif operandNumber == 1 :
			#print "pop oper 1"
			oper1Type = Instruction ().specs[instruction.opCode]['oper1Type']

			if oper1Type == atom.AtomType.real :
				#print "pop d"
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (float (pop[0]), oper1Type)

			elif oper1Type == atom.AtomType.int :
				#print "pop a"
				if self.machineStack.length () > 0 :
					#print "pop b"
					pop = self.machineStack.pop ()
					#print "pop c"
					#print pop
					return (int (pop[0]), oper1Type)

			elif oper1Type == atom.AtomType.bool :
				#print "pop e"
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (bool (pop[0]), oper1Type)

			elif oper1Type == atom.AtomType.any :
				#print "pop f"
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					#print 
					return (pop[0], atom.AtomType ().getValueType (pop[0]))

			else :
				if overflowMessagesOn :
					print "Program: popOperValue: instruction.operand1Type is invalid atom.AtomType " + str (oper1Type)
				return (None, None)

		else :
			if overflowMessagesOn :
				print "Program: popOperValue: invalid operand number " + str (operandNumber)
			return (None, None)



	def makeValueTypeTuple (self, value) :
			return (value,atom.AtomType ().getValueType (value))


	def execute (self, inputs=None) :

		#print self
		self.symbolTable = [10000]
		self.symbolTopIndex = 0
		self.inputs = []

		self.machineStack = Stack (10000)

		self.operand0 = 0.0 # set from operand1 type stack pop
		self.operand1 = 0.0 # set from operand1 type stack pop

		self.returnValue = 0.0 # set from result of operator

		if inputs is not None :
			self.inputs = inputs

		global overflowMessagesOn

		for lineNum,instr in enumerate (self.instructions) :
			#random.seed (self.randomSeed)
			r.setRandomSeed(self.randomSeed)

			#print "\n\n______ " +  ("%04i" % lineNum) + " " + Instruction.strings [instr.opCode] + " _______"

			if Instruction.specs[instr.opCode]['arity']  == InstructionOperandType.binary :
				self.operand0 = self.popOperValue (0, instr)
				self.operand1 = self.popOperValue (1, instr)

			elif Instruction.specs[instr.opCode]['arity'] == InstructionOperandType.unary :
				self.operand0 = self.popOperValue (0, instr)

			else :
				self.operand0 = None
				self.operand1 = None

			if (Instruction.specs[instr.opCode]['oper0Type'] is not None and
			 	self.operand0 is None) :
				if overflowMessagesOn :
					print ("Program: execute: instr: " +
							str (lineNum) +
							" " +
							Instruction.strings[instr.opCode] +
							" instruction found operand0 set to None. Nothing for pushing.")
				continue

			if (Instruction.specs[instr.opCode]['oper1Type'] is not None and
			 	self.operand1 is None) :
				if overflowMessagesOn :
					print ("Program: execute: instr: " +
							str (lineNum) +
							" " +
							Instruction.strings[instr.opCode] +
							" instruction found operand1 set to None. Nothing for pushing.")
				continue

			printArgType = False
			if printArgType :
				if Instruction.specs[instr.opCode]['argType'] is not None :
					print "\targ type = " + str (atom.AtomType.strings[Instruction.specs[instr.opCode]['argType']])
	
				else :
					print "\targ type = None"

			if instr.argument is not None :
				(constValue, constType) = self.makeValueTypeTuple (instr.argument)
# 				print "\t(" + str (constValue) + " isA " + atom.AtomType.strings[constType] + ")"
# 
# 			else :
# 				print "\t(None isA None)"

			#print "Operand 0 "
			printOperand = False
			if printOperand :
				if self.operand0 is not None :
					print "\t(" + str (self.operand0[0]) + " isA " + atom.AtomType.strings[self.operand0[1]] + ")"

				else :
					print "\t(None isA None)"

				#print "Operand 1 "

				#print self.operand1
				if self.operand1 is not None :
					print "\t(" + str (self.operand1[0]) + " isA " + atom.AtomType.strings[self.operand1[1]] + ")"

				else :
					print "\t(None isA None)"

			#print "stack pointer " + str (self.machineStack.pointer)
			#print "stack size " + str (self.machineStack.length ())
			#print "symbol top index " + str (self.symbolTopIndex)
			############################
			if instr.opCode == Instruction.assign :   ######## ASSIGN (BINARY) ######

				if self.symbolTopIndex > 0 :
# 					destIndex = int (self.operand0[0]) % self.symbolTopIndex
# 					print "assign destIndex = " + str (destIndex)
#
# 					if destIndex > self.symbolTopIndex :
# 						self.symbolTopIndex = destIndex
					destIndex = instr.argument

				else :
					#print "destIndex = 0"
					destIndex = 0
					self.symbolTopIndex = 0

				#print "assign value/type = " + str ((self.operand0[0],self.operand0[1]))

				self.symbolTable[destIndex] = (self.operand0[0],self.operand0[1])

				self.machineStack.push (self.operand0[0])

				#print "new stack size = " + str (self.machineStack.length ())

			elif instr.opCode == Instruction.pushFromSym :     ####### PUSH SYMBOL  (NULLARY) ######

				# push contents of symbol table at index argument or operand0 to specified stack
				if instr.argument is not None :
					sourceIndex = instr.argument

				elif self.operand0 is not None :
					sourceIndex = int (self.operand0)

				else:
					print "Program: execute: pushFromSym: neither argument nor operand0 are set to source type."
					continue

				if self.symbolTopIndex > 0 :
					(symbolValue, symbolType) = self.symbolTable[sourceIndex % self.symbolTopIndex]
	
					if symbolType == atom.AtomType.real :
						self.machineStack.push (float (symbolValue))
	
					elif symbolType == atom.AtomType.int :
						self.machineStack.push (int (symbolValue))
	
					elif symbolType == atom.AtomType.bool :
						self.machineStack.push (bool (symbolValue))
	
					else :
						if overflowMessagesOn :
							print "Program: execute: pushFromSym: instr: " + str (lineNum) + " ???? symbolType = " + str (symbolType) 
	
					#print "pushFromSym: new stack size = " + str (self.machineStack.length ())

# 				else :
# 					print "pushFromSym can't push from 0-length symbol table."

			elif instr.opCode == Instruction.pushFromInput :     ####### PUSH FROM INPUTS LIST  (UNARY) ######

# 				print "Program: execute: PUSH INPUT"
				if instr.argument is None :
					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushFromInput instruction found operand0 set to None. Unknown what index to push."
					continue

				if self.inputs is None :
					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushFromInput instruction found inputs set to None. Nothing for pushing."
					continue

				if len (self.inputs) == 0 :
					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushFromInput instruction found zero-length inputs. Nothing for pushing."
					continue

				#print "instr.argument = " + str (instr.argument)
				##print "len inputs = " + str (len (self.inputs))
				inputValue = self.inputs[int (instr.argument) % len (self.inputs)]
				##print "input value " + str (inputValue)
				self.machineStack.push (float (inputValue))

				##print "new stack size = " + str (self.machineStack.length ())

			elif instr.opCode == Instruction.popToOperand  :     ####### POP to OPERAND ######
				# pop contents of specified stack to self.operand0 or operand1
				if instr.argument % 2 == 0 :
					self.operand0 = self.machineStack.pop ()
					#print "execute: popToOperand: operand0 chosen"
					#time.sleep(3)
				elif instr.argument % 2 == 1 :
					self.operand1 = self.machineStack.pop ()
					#print "execute: popToOperand: operand0 chosen"
					#time.sleep(3)

				else :
					#print "execute: popToOperand: operand " + str (instr.argument)
					print "Program: execute: popToOperand: argument should be 0 or 1 (operand number) Instead got " + str (instr.argument)

				#print "new stack size = " + str (self.machineStack.length ())

			elif instr.opCode == Instruction.pushConst :     ####### PUSH CONST  (UNARY) ######

				(constValue, constType) = self.makeValueTypeTuple (instr.argument)
				if constType == atom.AtomType.real :
					#print "pushConst a"
					self.machineStack.push (float (constValue))

				elif constType == atom.AtomType.int :
					#print "pushConst b"
					self.machineStack.push (int (constValue))

				elif constType == atom.AtomType.bool :
					#print "pushConst c"
					self.machineStack.push (bool (constValue))

				else :
					#print "pushConst d"

					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " ???? argumentType = " + str (constType) 

				#print "new stack size = " + str (self.machineStack.length ())

			elif instr.opCode == Instruction.pushAllSym :     ####### PUSH ALL SYMBOLS  (NULLARY) ######

# 				print "Program: execute: PUSH ALL SYMBOLS"
				for (symbolValue, symbolType) in self.symbolTable :
					if symbolType == atom.AtomType.real :
						self.machineStack.push (float (symbolValue))

					elif symbolType == atom.AtomType.int :
						self.machineStack.push (int (symbolValue))

					elif symbolType == atom.AtomType.bool :
						self.machineStack.push (bool (symbolValue))

					else :
						if overflowMessagesOn :
							print "Program: execute: instr: " + str (lineNum) + " ???? symbolType = " + str (symbolType) 


			elif instr.opCode == Instruction.popToReturn  :    ####### POP TO RETURN ######
				self.returnValue = self.machineStack.pop ()

			elif instr.opCode == Instruction.noOp  :    ####### NOOP ######
				pass

			elif instr.opCode == Instruction.rand  :    ####### RAND ######
				self.returnValue = r.getRandomValue()   #random.random ()
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.add  :    ####### ADD ######
				self.returnValue = self.operand0[0] + self.operand1[0]
				self.machineStack.push (self.returnValue)


			elif instr.opCode == Instruction.sub  :	####### SUB ######
				self.returnValue = self.operand0[0] - self.operand1[0]
				self.machineStack.push (self.returnValue)


			elif instr.opCode == Instruction.mult :	####### MULT #####
				self.returnValue = self.operand0[0] * self.operand1[0]
				self.machineStack.push (self.returnValue)


			elif instr.opCode == Instruction.div  :	####### DIV ######
				if self.operand1[0] != 0.0 :
					self.returnValue = self.operand0[0] / self.operand1[0]

				else :
					self.returnValue = 0
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.pow :	####### POW ######
				if (self.operand0[0] == 0.0 and
				   self.operand1[0] == 0.0 ) :
					self.returnValue = 0.0
					self.machineStack.push (self.returnValue)

				maxAllowed = 10.0
				if self.operand0[0] + self.operand1[0] > maxAllowed :
					self.operand0 = ((self.operand0[0]/float(self.operand0[0]+self.operand1[0])) * maxAllowed, self.operand0[1])
					self.operand1 = ((self.operand1[0]/float(self.operand0[0]+self.operand1[0])) * maxAllowed, self.operand1[1])

				if abs (self.operand0[0]) > 200 :
					sign = self.operand0[0] / abs (self.operand0[0])
					self.operand0 = (200.0 * sign, self.operand0[1])

				if abs (self.operand1[0]) > 100 :
					sign = self.operand1[0] / abs (self.operand1[0])
					self.operand1 = (100.0 * sign, self.operand1[1])

				if self.operand0[0] < 0.0 :
					self.operand0 = (abs (self.operand0[0]), self.operand0[1])

				if (self.operand0[0] == 0.0 and
					self.operand1[0] < 0.0) :
					self.returnValue = 0.0

				else :
					#print "a^b a = " + str(a) + " " + str(b)
					self.returnValue = self.operand0[0]**self.operand1[0]

				#print self.operand0
				#print self.operand1
				#print "push pow result = " + str (self.returnValue)
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.sin :	####### SIN ######

				self.returnValue = math.sin (self.operand0[0])
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.cos :	####### COS ######

				self.returnValue = math.cos (self.operand0[0])
				self.machineStack.push (self.returnValue)

			else :
				if overflowMessagesOn :
					assert False, "Program: execute: instr: " + str (lineNum) + " CRASH: execute2: instruction " + str (instr.opCode) + " not defined."

		#print "Program: execute: symbolTable = " + str (self.symbolTable)
		if self.machineStack.length () > 0 :
			return self.machineStack.pop ()

		else :
			return None

	def makeBreakdown (self) :
		startBit = 0
		endBit = 0
		self.binaryBreakdown = []
		for thisInstruction in self.instructions :
			# opCode
			startBit = endBit
			endBit = startBit + Instruction ().opCodePrecision
			self.binaryBreakdown.append ([startBit,endBit])

			# argument type code
			startBit = endBit
			endBit = startBit + Instruction ().operTypePrec
			self.binaryBreakdown.append ([startBit,endBit])

			# argument value
			startBit = endBit
			endBit = startBit + Instruction ().argumentPrecision
			self.binaryBreakdown.append ([startBit,endBit])

	def instructionsToFlatBinary (self) :

		#print "itfb program..."
		#self.printProgram ()

		#print "itfb len(self.instructions) = " + str (len(self.instructions))
# 		instructionSize =  (Instruction ().opCodePrecision +
# 							Instruction ().operTypePrec +
# 							Instruction ().operTypePrec +
# 							Instruction ().argumentPrecision +
# 							Instruction ().operTypePrec)
		flatBinary = ''
		for thisInstruction in self.instructions :
			string = ''
			breakdown = []
			opCodeBinStr = atom.BinaryString (precision=Instruction ().opCodePrecision, rangeIn=Instruction ().opCodeTypeRange)
			opCodeBinStr.setFromInt(thisInstruction.opCode)
			opCodeBinary = opCodeBinStr.binaryStr
			string += opCodeBinary
			flatBinary += opCodeBinary
			breakdown.append (len (opCodeBinary))
			assert len (opCodeBinary) == Instruction ().opCodePrecision, ("instructionsToFlatBinary: opCode should be " +
														str (self.opCodePrec) +
														" bits in length, but is " +
														str (len (opCodeBinary)))

			argType = Instruction.specs[thisInstruction.opCode]['argType']

			##### argument type
			argumentType = atom.BinaryString (precision=Instruction ().operTypePrec, rangeIn=Instruction ().operTypeRange)
			argumentType.setFromInt (argType)
			argumentTypeBinary = argumentType.binaryStr
# 			if Instruction.specs[thisInstruction.opCode]['argType'] is None :
# 				print "itfb " + str (Instruction.strings[thisInstruction.opCode]) + " argType = None"
# 
# 			else :
# 				print "itfb " + str (Instruction.strings[thisInstruction.opCode]) + " argType = " + str (atom.AtomType.strings[Instruction.specs[thisInstruction.opCode]['argType']])
# 			#time.sleep(1)

			flatBinary += argumentTypeBinary
			string += argumentTypeBinary

			breakdown.append (len (argumentTypeBinary))
			assert len (argumentTypeBinary) == Instruction ().operTypePrec, ("instructionsToFlatBinary: argument type should be " +
																str (self.operTypePrec) +
																" bits in length, but is " +
																str (len (argumentTypeBinary)))

			##### set argument
# 				real = 0
# 				int = 1
# 				bool = 2
# 				nil = 3
# 				any = 4
			if argType == atom.AtomType.any :
				if   type (thisInstruction.argument) == types.IntType :
					argType = atom.AtomType.int

				elif type (thisInstruction.argument) == types.FloatType :
					argType = atom.AtomType.real

				elif type (thisInstruction.argument) == types.BooleanType :
					argType = atom.AtomType.bool

			if argType == atom.AtomType.real :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.realMax])
				argument.setFromReal (thisInstruction.argument)

			elif argType == atom.AtomType.int :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
				argument.setFromInt (thisInstruction.argument)

			elif argType == atom.AtomType.bool :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
				argument.setFromInt (thisInstruction.argument)

			elif argType == None :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
				argument.setToNil ()

			else :
				if overflowMessagesOn :
					print "instructionsToFlatBinary: unsupported argumentType (should be real, int, or None"
				exit (-1)
			argumentBinary = argument.binaryStr

			flatBinary += argumentBinary
			string += argumentBinary
			breakdown.append (len (argumentBinary))
			assert len (argumentBinary) == Instruction ().argumentPrecision, ("instructionsToFlatBinary: argument type should be " +
																str (Instruction ().argumentPrecision) +
																" bits in length, but is " +
																str (len (argumentBinary)))


			#print "itfb len(string) = " + str (len (string))
		#print "itfb len(flatBinary)/len(self.instructions) = " + str (len(flatBinary)/float(len(string)))
		return flatBinary
# 
# 	def instructionsToBinary (self) :
# 
# 		#print "itb program..."
# 		#self.printProgram ()
# 
# 		#print "itb len(self.instructions) = " + str (len(self.instructions))
# 		instructionSize =  (Instruction ().opCodePrecision +
# 							Instruction ().operTypePrec + # argument type
# 							Instruction ().argumentPrecision)
# 		binary = []
# 		for thisInstruction in self.instructions :
# 			binString = ''
# 			debug = ''
# 			###### op code ####
# 			opCodeBinStr = atom.BinaryString (precision=Instruction ().opCodePrecision, rangeIn=Instruction ().opCodeTypeRange)
# 			opCodeBinStr.setFromInt(thisInstruction.opCode)
# 			opCodeBinary = opCodeBinStr.binaryStr
# 
# 			#print thisInstruction.opCode
# 			binString += opCodeBinary
# 			debug += opCodeBinary + ' '
# 
# 			assert len (opCodeBinary) == Instruction ().opCodePrecision, ("instructionsToFlatBinary: opCode should be " +
# 														str (self.opCodePrec) +
# 														" bits in length, but is " +
# 														str (len (opCodeBinary)))
# 
# 			####### arg type #####
# 			argType = Instruction.specs[thisInstruction.opCode]['argType']
# 			argumentType = atom.BinaryString (precision=Instruction ().operTypePrec, rangeIn=Instruction ().operTypeRange)
# 			argumentType.setFromInt (argType)
# 			argumentTypeBinary = argumentType.binaryStr
# 			binString += argumentTypeBinary
# 			debug += argumentTypeBinary + ' '
# 
# 			assert len (argumentTypeBinary) == Instruction ().operTypePrec, ("instructionsToFlatBinary: argument type should be " +
# 																str (self.operTypePrec) +
# 																" bits in length, but is " +
# 																str (len (argumentTypeBinary)))
# 
# 			##### set argument
# # 				real = 0
# # 				int = 1
# # 				bool = 2
# # 				nil = 3
# # 				any = 4
# 			if argType == atom.AtomType.any :
# 				if   type (thisInstruction.argument) == types.IntType :
# 					argType = atom.AtomType.int
# 
# 				elif type (thisInstruction.argument) == types.FloatType :
# 					argType = atom.AtomType.real
# 
# 				elif type (thisInstruction.argument) == types.BooleanType :
# 					argType = atom.AtomType.bool
# 
# 			if argType == atom.AtomType.real :
# 				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.realMax])
# 				argument.setFromReal (thisInstruction.argument)
# 
# 			elif argType == atom.AtomType.int :
# 				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
# 				argument.setFromInt (thisInstruction.argument)
# 
# 			elif argType == atom.AtomType.bool :
# 				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
# 				argument.setFromInt (thisInstruction.argument)
# 
# 			elif argType == None :
# 				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
# 				argument.setToNil ()
# 
# 			else :
# 				if overflowMessagesOn :
# 					print "instructionsToFlatBinary: unsupported argumentType (should be real, int, or None"
# 				exit (-1)
# 			argumentBinary = argument.binaryStr
# 
# 			binString += argumentBinary
# 			debug += argumentBinary + ' '
# 
# 			assert len (argumentBinary) == Instruction ().argumentPrecision, ("instructionsToFlatBinary: argument type should be " +
# 																str (Instruction ().argumentPrecision) +
# 																" bits in length, but is " +
# 																str (len (argumentBinary)))
# 
# 
# 			#print debug
# 			#print "itb len(binString) = " + str (len (binString)) + " instr calculated length = " + str (instructionSize)
# 			binary.append (binString)
# 
# 		#print "len binary = " + str (len (binary))
# 		#exit(0)
# 		return binary

	def flatBinaryToInstructions (self, binaryIn) :
		#print "flatBinaryToInstructions"
		#print "binaryIn"
		#print binaryIn

		instructions = []
		instructionSize =  (Instruction ().opCodePrecision +
							Instruction ().argumentPrecision +
							Instruction ().operTypePrec)
		numInstructions = len (binaryIn) / instructionSize

		global overflowMessagesOn

		startBit = 0
		endBit = 0
		for instruction in range (numInstructions) :

			string = ''
			######## OPCODE ######
			breakdown = []
			startBit = endBit
			endBit = startBit + Instruction ().opCodePrecision
			opCode = atom.BinaryString (rangeIn=[0,len (Instruction.strings)], binaryStrIn=binaryIn[startBit:endBit])
			#opCodeInt = opCode.toInt ()

			###### change opCode if in the expression section and not an expression opcode ##########
			print " opCode.toInt () = " + str (opCode.toInt ())
			if (not opCode.toInt () in Instruction.expressionInstructions and
				instruction >= numInstructions - self.expressionSize):
				print instruction
				print "changing opCode " + Instruction.strings[opCode.toInt ()]
				opCodeInt = Instruction.expressionInstructions[int (random.random () * len (Instruction.expressionInstructions))] 
				opCode = atom.BinaryString (rangeIn=[0,len (Instruction.strings)], binaryStrIn='0' * Instruction ().opCodePrecision)
				opCode.setFromInt (opCodeInt)

			string += binaryIn[startBit:endBit]
			if opCode.toInt () is None :
				##### case in which a valid op code has been mutated to a None (11111) and needs to be made valid (switched to 0000)
				opCode = atom.BinaryString (rangeIn=[0,len (Instruction.strings)], binaryStrIn='0' * Instruction ().opCodePrecision)
				#opCodeInt = opCode.toInt ()
			#breakdown.append (len (binaryIn[startBit:endBit]))


			####### ARGUMENT TYPE ######
			startBit = endBit
			endBit = startBit + Instruction ().operTypePrec
			argumentTypeBString = atom.BinaryString (rangeIn=[0,len (atom.AtomType.strings)], binaryStrIn=binaryIn[startBit:endBit])
			#breakdown.append (len (binaryIn[startBit:endBit]))
			argumentType = argumentTypeBString.toInt ()
			string += binaryIn[startBit:endBit]


			####### ARGUMENT VALUE ######
			startBit = endBit
			endBit = startBit + Instruction ().argumentPrecision
			if argumentType == atom.AtomType.real :
				argument = atom.BinaryString (rangeIn=[0,self.realMax], binaryStrIn=binaryIn[startBit:endBit])
				#breakdown.append (len (binaryIn[startBit:endBit]))
				argumentValue = argument.toReal ()

			elif argumentType == atom.AtomType.int :
				argument = atom.BinaryString (rangeIn=[0,self.intMax], binaryStrIn=binaryIn[startBit:endBit])
				#breakdown.append (len (binaryIn[startBit:endBit]))
				argumentValue = argument.toInt ()

			elif argumentType == None :
				argumentValue = None

			else :
				argumentType = 0
				argument = atom.BinaryString (rangeIn=[0,self.realMax], binaryStrIn=binaryIn[startBit:endBit])
				#breakdown.append (len (binaryIn[startBit:endBit]))
				argumentValue = argument.toReal ()

			if (not opCode.toInt () in Instruction.expressionInstructions and
				instruction >= numInstructions - self.expressionSize):
				Instruction.repairArgument (opCode, argumentValue, argumentType)


			argumentValue = Instruction ().repairArgument (opCode, argumentValue, argumentType)
			string += binaryIn[startBit:endBit]

# 			print "fbti len(string) = " + str (len (string))
# 			print Instruction ().strings[opCode.toInt ()]
# 			print "argumentType = " + atom.AtomType.strings[argumentType]
# 			print "argumentValue = " + str (argumentValue)

			instructions.append (Instruction (opCode.toInt (),argumentValue))

		self.instructions = instructions

		#print "fbti program..."
		#self.printProgram ()
		#exit(0)

	def test (self) :

		#opCodePrec = Instruction.opCodePrecision # opCode/operator precision = 4
		#operPrec = Instruction.operandPrecision # operand precision = 32

		#self.symbolTable = [(0,0),(0,0),(0,0),(0,0),(0,0)] #inputValue, 0.5, 0.2, 0.3, 3.0]
		#elf.symbolTable = [inputValue, 3.0]
		
		
		
		#.5 +   (.2 * x) +     0.4 *   (x ** 3.0)
		
		self.instructions = []
		
		
		expression = False
		if expression :
			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(3.0))) #(3.0)
			
			self.instructions.append (Instruction ( Instruction.pushFromInput,
													argument=0)) # push input 0 to stack (x)
			
			self.instructions.append (Instruction ( Instruction.pow)) # (x**3.0)
	
	
			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(0.4))) #(0.4)
	
			self.instructions.append (Instruction ( Instruction.mult)) # (0.4 *   (x ** 3.0))
	
	
	
			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(0.2))) #(0.2)
			
			self.instructions.append (Instruction ( Instruction.pushFromInput,
													argument=0)) # push input 0 to stack (x)
			
			self.instructions.append (Instruction ( Instruction.mult)) # (0.2*x)
			
			
			self.instructions.append (Instruction ( Instruction.add)) # ( (.2 * x) +     0.4 *   (x ** 3.0))
			
			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(0.5))) #(0.5)
	
			self.instructions.append (Instruction ( Instruction.add)) # (.5 +   (.2 * x) +     0.4 *   (x ** 3.0))

		else :
			self.synthesize ()
# 		self.machineStack.printValues ()
# 		self.execute([2.0])
# 		print self.machineStack.valueArray[0]
# 		self.machineStack.printValues ()
# 		
# 		
# 		
# 		print func (2.0)
# 		exit (0)
# 		
# 		
# 		print "test: define instr 2..."
# 		#push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushFromInput,
# 												argument=0)) # push input 0 to stack
# 
# 		print "test: define instr 3..."
# 		#pop stack to symbol at index from stack
# 		self.instructions.append (Instruction ( Instruction.assign,
# 												argument=self.symbolTopIndex + 1))
# 
# 
# 
# # 			#var1 = 3.0
# 		print "test: define instr 4..."
# 		#set self.operand0 to 4 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=int(1)))
# 
# 		print "test: define instr 5..."
# 		#set self.operand0 to 4 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=float(3.0)))
# 
# 		print "test: define instr 6..."
# 		#pop stack to symbol at index from stack
# 		self.instructions.append (Instruction (	Instruction.assign,
# 												argument=self.symbolTopIndex + 1))
# 
# # 			#var2 = 0.4
# 		print "test: define instr 7..."
# 		#set self.operand0 to 4 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=int(2)))
# 
# 		print "test: define instr 8..."
# 		#set self.operand0 to 4 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=float(0.4)))
# 
# 		print "test: define instr 9..."
# 		#pop stack to symbol at index from stack
# 		self.instructions.append (Instruction (	Instruction.assign,
# 												argument=self.symbolTopIndex + 1))
# 
# # # 			#var3 = 0.2
# 		print "test: define instr 7..."
# 		#set self.operand0 to 3 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=int(3)))
# 
# 		print "test: define instr 8..."
# 		#set self.operand0 to 4 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=float(0.2)))
# 
# 		print "test: define instr 9..."
# 		#pop stack to symbol at index from stack
# 		self.instructions.append (Instruction (	Instruction.assign,
# 												argument=self.symbolTopIndex + 1))
# 
# # 			#var4 = 0.5
# 		print "test: define instr 7..."
# 		#set self.operand0 to 3 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=int(4)))
# 
# 		print "test: define instr 8..."
# 		#set self.operand0 to 4 and push self.operand0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushConst,
# 												argument=float(0.5)))
# 
# 		print "test: define instr 9..."
# 		#pop stack to symbol at index from stack
# 		self.instructions.append (Instruction (	Instruction.assign,
# 												argument=self.symbolTopIndex + 1))
# 
# 
# 		#push symbols onto stack
# 		#push symbol 2 to stack
# 		print "test: define instr 12..."
# 		self.instructions.append (Instruction ( Instruction.pushFromSym,
# 												argument=int(2)))
# 
# 		print "test: define instr 10..."
# 		#push symbol 0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushFromSym,
# 												argument=int(1)))
# 
# 		#push symbol 1 to stack
# 		print "test: define instr 11..."
# 		self.instructions.append (Instruction ( Instruction.pushFromSym,
# 												argument=int(0)))
# 
# 
# 		#pow
# 		print "test: define instr 13..."
# 		self.instructions.append (Instruction ( Instruction.pow))
# 
# 		# mult
# 		print "test: define instr 14..."
# 		self.instructions.append (Instruction ( Instruction.mult))
# 
# 
# 		print "test: define instr 10..."
# 		#push symbol 0 to stack
# 		self.instructions.append (Instruction ( Instruction.pushFromSym,
# 												argument=int(3)))
# 
# 
# 		#push symbol 1 to stack
# 		print "test: define instr 11..."
# 		self.instructions.append (Instruction ( Instruction.pushFromSym,
# 												argument=int(0)))
# 		# mult
# 		print "test: define instr 15..."
# 		self.instructions.append (Instruction ( Instruction.mult)) 
# 
# 		# add
# 		print "test: define instr 16..."
# 		self.instructions.append (Instruction ( Instruction.add)) 
# 
# 		#push symbol 4 to stack
# 		print "test: define instr 11..."
# 		self.instructions.append (Instruction ( Instruction.pushFromSym,
# 												argument=int(4)))
# 
# 		# add
# 		print "test: define instr 16..."
# 		self.instructions.append (Instruction ( Instruction.add)) 


		sampleRange = [0,10]
		samples = []

		for test in range (sampleRange[0], sampleRange[1]) :
			print "___New Sample___"

			self.machineStack = []

# 			self.operand0 = 0.0
# 			self.operand1 = 0.0
# 			self.value = 0.0

			#.5 +   .2 * x +   .3 * (x ** 3)

			test2 = test * .1
			print "input = " + str (test2)
			#self.inputs = [test2]

			#  0.4 * (x ** 3.0)
			#.5 +   (.2 * x) +     0.4 *   (x ** 3.0)

			for i in self.instructions :
				print i
			#print "test: _______about to call execute________"

			result = self.execute ([test2])
			#print "test: result"
			#print result
			if result is None :
				result = 0

			else :
				result = result[0]

			samples.append (result + .01)

		def plotFunction () :
		
			#t = arange (0.0, 1.0, 0.01)
		
			fig = figure (1)
		
			ax1 = fig.add_subplot (1,1,1)
		
			#ax1.text(40, 0, expression)
		
			#ax1.plot (t, sin (2*pi*t))
		
			x = [_ * .1 for _ in range (sampleRange[0], sampleRange[1])]
			hf = [func (_) for _ in x]
			ax1.plot (x,hf)
			#print len(x)
			#print len(samples)
			ax1.plot (x,samples)
		
			ax1.grid (True)
			
			yBoxMax = max (max (samples), max(hf)) * 1.1
			yBoxMin = min (min (samples), min(hf)) - 0.1
			
			ax1.set_ylim ((yBoxMin,  yBoxMax))
			ax1.set_xlim ((-.1,  max (x) * 1.1))
			#ax1.set_ylabel('fitness')
			#ax1.set_xlabel('generation')
			#ax1.set_title ('optimization')
		
			show ()
		
		#print samples

# 		plotFunction ()
# 
# prog = Program(0)
# prog.test ()
