
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

# if l doesn't exist, make it otherwise use the existing one
try :
	l

except :
	l = logM1.Log()



def func (x) :
	return .5 +   (.2 * x) +     0.4 *   (x ** 3.0)
	#return 0.4 * (x ** 3.0)

def instructionExecuteFragment (self, instr) :
	opCode = instr.opCode
	fragment = []
	if Instruction.specs[opCode]['arity'] == InstructionArgumentType.nullary :
		if opCode == Instruction.popToOperand :
			print "instructionExecuteFragment: note: no fragment generated for pop2Oper0 or pop2Oper1"

	elif Instruction.specs[opCode]['arity'] == InstructionArgumentType.unary :
		fragment.append (Instruction (Instruction.popToOperand,
												argument=0))

	elif Instruction.specs[opCode]['arity'] == InstructionArgumentType.binary :
		fragment.append (Instruction (Instruction.popToOperand,
												argument=0))
		fragment.append (Instruction (Instruction.popToOperand,
												argument=1))

	else :
		print "instructionExecuteFragment: invalid InstructionArgumentType"

	fragment.append (instr)

	return fragment

class InstructionArgumentType :
	nullary = 0
	unary = 1
	binary = 2

	strings = [ "nullary",
				"unary",
				"binary"]

	precision = atom.BinaryString().rangeToPrecision ([0,len (strings)])

	def toStr (self, InstructionArgumentIn) :
		assert InstructionArgumentIn < len (self.strings), "toStr: InstructionArgumentIn not found in InstructionArgumentTypes.strings."
		return self.strings[InstructionArgumentIn]

	def toInstructionArgumentType (self, stringIn) :
		assert stringIn in self.strings, "toInstructionArgumentType: stringIn not found in InstructionArgumentType.strings."
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

	precision = atom.BinaryString().rangeToPrecision ([0,len (strings)])

	literalPrecision = 32

	def toStr (self, intValueIn) :
		return self.Instr[intValueIn]

	def toInstruction (self, stringIn) :
		assert stringIn in self.strings, "toInstruction: stringIn not found in Instruction.strings."
		return self.strings.index (stringIn)

	def isValid (self, instr) :
		return (type (instr) == types.IntType and
				instr < len (self.strings))

	specs =	{assign:		{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.int,
							'oper1Type': atom.AtomType.any,
							'argType': atom.AtomType.int },

			pushFromSym:	{'arity': InstructionArgumentType.unary,
							'oper0Type': atom.AtomType.int,
							'argType': atom.AtomType.int  },

			pushFromInput:	{'arity': InstructionArgumentType.unary,
							'oper0Type': atom.AtomType.int,
							'argType': atom.AtomType.int }, 

			popToOperand:	{'arity': InstructionArgumentType.nullary,
							'argType': atom.AtomType.int  },

			pushConst:		{'arity': InstructionArgumentType.nullary,
							'argType': atom.AtomType.any  },

			pushAllSym:		{'arity': InstructionArgumentType.nullary,
							'argType': None },

			popToReturn:	{'arity': InstructionArgumentType.nullary,
							'argType': None },

			noOp:			{'arity': InstructionArgumentType.nullary ,
							'argType': None},

			rand:			{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.real, # range min
							'oper1Type': atom.AtomType.real, # range max
							'argType': None },

			add:			{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None },

			sub:			{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None },

			mult:			{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None},

			div:			{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None },

			pow:			{'arity': InstructionArgumentType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None },

			sin:			{'arity': InstructionArgumentType.unary,
							'oper0Type': atom.AtomType.any,
							'argType': None },

			cos:       	{'arity': InstructionArgumentType.unary,
							'oper0Type': atom.AtomType.any,
							'argType': None }}


	instrTypeRange = [0, len (strings)]
	operTypeRange = [0,len (atom.AtomType.strings)]
	operTypePrec = atom.BinaryString().rangeToPrecision (operTypeRange)

	def __init__ (self, opCodeIn=None, argument=None) :
#
		if opCodeIn is None :

			return
#
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
							"'.")
					exit(0)

				else :
					self.argument = None

			elif self.specs[self.opCode]['argType'] == atom.AtomType.any :
				self.argument = argument

			elif self.specs[self.opCode]['argType'] == atom.AtomType.int :
				if type (argument) != types.IntType :
					print ("Instruction.__init__() argument should be int, but was '" +
							str (argument) +
							"'.")
					exit(0)

				else :
					self.argument = argument

			elif self.specs[self.opCode]['argType'] == atom.AtomType.real :
				if type (argument) != types.FloatType :
					print ("Instruction.__init__() argument should be real, but was '" +
							str (argument) +
							"'.")
					exit(0)

				else :
					self.argument = argument

			else :
				print ("Instruction.__init__() unknown argType in specs for opCode '" +
						str (self.specs[self.opCode]['name']) +
						"'.")
				exit(0)

			



# 	def __repr__ (self) :
# 		opCode = self.opCodeString
# 
# 		if self.operand0Type == atom.AtomType.real :
# 			operand0 = "real:"  #### print value of operand
# 
# 		elif self.operand0Type == atom.AtomType.int :
# 			operand0 = "int:"
# 
# 		elif self.operand0Type == atom.AtomType.bool :
# 			operand0 = "bool:"
# 
# 		elif self.operand0Type == atom.AtomType.int :
# 			operand0 = "int:"
# 
# 		elif self.operand0Type == None :
# 			operand0 = "None"
# 
# 		elif self.operand0Type == atom.AtomType.stackOverflow :
# 			operand0 = "stackOverflow:"
# 
# 		else :
# 			operand0 = "__repr__: operand0 unknown type"
# 
# 		if self.operand1Type == atom.AtomType.real :
# 			operand1 = "real:"  #### print value of operand
# 
# 		elif self.operand1Type == atom.AtomType.int :
# 			operand1 = "int:"
# 
# 		elif self.operand1Type == atom.AtomType.bool :
# 			operand1 = "bool:"
# 
# 		elif self.operand1Type == atom.AtomType.int :
# 			operand1 = "int:"
# 
# 		elif self.operand1Type == None :
# 			operand1 = "None"
# 
# 		elif self.operand1Type == atom.AtomType.stackOverflow :
# 			operand1 = "stackOverflow:"
# 
# 		else :
# 			operand1 = "__repr__: operand1 unknown type"
# 
# 		helpStr = "\t(" + self.specs[opCode]['format'] + ")"
# 		if self.register2Type is None :
# 			register2TypeStr = '<Nil>'
# 
# 		else :
# 			print "self.register2Type = " + str (self.register2Type)
# 			if self.register2Type < len (atom.AtomType.strings) :
# 				register2TypeStr = atom.AtomType.strings [self.register2Type]
# 
# 			else :
# 				register2TypeStr = 'InvalidType'
# 
# 		result = "{:<15}{:<15}{:<15}{:<13}{:<10}{}".format (opCode, 
# 														operand0, 
# 														operand1, 
# 														str (self.setRegister2), 
# 														register2TypeStr,
# 														helpStr)
# 		return result

class Stack :
	def __init__ (self, allocatedSize) :
		self.pointer = -1
		self.allocatedSize = allocatedSize
		self.valueArray = np.array(np.zeros (self.allocatedSize), dtype=np.float_)
		self.typeArray = np.array(np.zeros (self.allocatedSize), dtype=np.int_)

	def pop (self) :
		if self.pointer >= 0 :
			resultType = self.typeArray[self.pointer]
			resultValue = atom.AtomType ().coerce (self.valueArray[self.pointer], resultType)

			self.pointer -= 1
			return (resultValue, resultType)

		else :
			print "Stack:pop () trying to pop from an empty stack."

	def push (self, item) :
		print "push " + str (self.pointer)
		if (self.pointer < self.allocatedSize or
		    self.pointer == -1) :
			print "push"
			self.pointer += 1
			self.valueArray[self.pointer] = float (item)
			self.typeArray[self.pointer] = atom.AtomType ().getValueType (item)

		else :
			print "Stack:pop () Overflow. Trying to push beyond the end of a stack."

	def printValues (self) :
		print "Pointer = " + str (self.pointer)
		if self.pointer > 0 :
			print 'Stack ['
			for index in range (self.pointer, 0, -1) :
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

class Program :
	def __init__ (self) :

		self.instructions =   []
		self.symbolTable = []
		self.symbolTopIndex = 0

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

	def randomInstruction (self) :
		numSpecs = len (Instruction.specs)
		setVarSpecs = 3

		choice = int (random.random () * (numSpecs - setVarSpecs)) + setVarSpecs
		key = Instruction.specs.keys ()[choice]
		return key

# 	def printState (self) :
# 		print "symbolTable: " + str (self.symbolTable)
# 		print ""
# 		print "stack: " + str (self.stack)


	def synthesize (self) :
		#print "synthesize "

		self.instructions = []

		numConstants = 3
		numWorkingSymbols = 4
		numInputs = 1
		numSymbols = numInputs + numWorkingSymbols + numConstants

		#inputValue = 0
			#  0.4 * (x ** 3.0)
			#.5 +   (.2 * x) +     0.4 *   (x ** 3.0)


		###### instructions to populate the symbol table with input argument variables #####
		for index in range (numInputs) :

			# push value from inputs[index] onto the stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushFromInput,
														argument=index))
			# assign the stackTop value to symbol[index]
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index))


		###### instructions to populate the symbol table with constants ######
		pythagoras=math.sqrt (2.0)
		theodorus=math.sqrt (3.0)
		sqrt5=math.sqrt (5.0)
		mascheroni=0.5772156649
		goldenRatio=1.5180339887
		berstein=0.2801694990
		gauss=0.3036630028
		landau=0.5
		omega=0.5671432904
		sierpinski=2.5849817595
		recipFib=3.3598856662

		consts = [0.0,1.0,math.pi,math.e,pythagoras,theodorus,sqrt5,mascheroni,goldenRatio,berstein,gauss,landau,omega,sierpinski,recipFib]
		index = numInputs

		for const in consts :
			# push const onto stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushConst,
														argument=const))
			# assign TOS to a symbol at index
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index))
			index += 1


		###### instructions to populate the symbol table with working variables (init to random) #####
		for index in range (numInputs,numWorkingSymbols) :

			# push value onto stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushConst,
														argument=random.random () * p.parameters["realMax"]))
			# assign TOS to a symbol at index
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index))
			index += 1

		###### instruction to push all symbols onto the stacks #####
		self.instructions.append (Instruction (   opCodeIn=Instruction.pushAllSym))

		sizeOfSymTable = index

		# SYNTHESIZE random instructions
		for _ in range (self.numLines) :

			# instruction
			opCode = int (random.random () * len (Instruction.specs))

			preFragment = []

			if Instruction.specs[opCode]['arity'] == InstructionArgumentType.nullary :
				if opCode == Instruction.popToOperand :
					# set the argument to the index of the symbolTable to which the value will be assigned
					index = int (random.random * sizeOfSymTable)
					argument = index
					print "synthesize: no fragment generated for popToOperand"

				elif opCode == Instruction.pushConst :
					# randomly choose whether the type should be int or real
					rhsType = [atom.AtomType.real, atom.AtomType.int][int (random.random () * 2)]
					if rhsType == atom.AtomType.real :
						rhs = random.random * p.parameters["realMax"]

					elif rhsType == atom.AtomType.int :
						rhs = int (random.random * p.parameters["intMax"])

					# push the rhs of the assignment onto the stack to prepare for the assign instructions
					preFragment.append (Instruction (Instruction.pushConst,
													argument=rhs))


					preFragment.append (Instruction (Instruction.popToOperand,
													argument=0))


			elif Instruction.specs[opCode]['arity'] == InstructionArgumentType.unary :
				preFragment.append (Instruction (Instruction.popToOperand,
												argument=0))
				if (opCode == Instruction.pushFromSym or
				   opCode == Instruction.pushFromSym) :
					# set the argument to the index of the symbolTable to which the value will be assigned
					index = int (random.random * sizeOfSymTable)
					argument = index


			elif Instruction.specs[opCode]['arity'] == InstructionArgumentType.binary :
				if opCode == Instruction.assign :
					# randomly choose whether the type should be int or real
					rhsType = [atom.AtomType.real, atom.AtomType.int][int (random.random () * 2)]
					if rhsType == atom.AtomType.real :
						rhs = random.random * p.parameters["realMax"]

					elif rhsType == atom.AtomType.int :
						rhs = int (random.random * p.parameters["intMax"])

					# push the rhs of the assignment onto the stack to prepare for the assign instructions
					preFragment.append (Instruction (Instruction.pushConst,
													argument=rhs))

					# set the argument to the index of the symbolTable to which the value will be assigned
					lhs = int (random.random * sizeOfSymTable)
					argument = lhs


				preFragment.append (Instruction (Instruction.popToOperand,
												argument=0))
				preFragment.append (Instruction (Instruction.popToOperand,
												argument=1))

			else :
				print "synthesize: invalid InstructionArgumentType"

			instr = Instruction (Instruction.opCode,
								argument=argument)

			self.instructions.extend (preFragment)
			self.instructions.append (instr)

		self.genotype = self.instructionsToFlatBinary ()
		self.instructions = []

	def printProgram (self) :
		for instr in self.instructions :
			print instr


# 	def getOperValue (self, operandNumber, instruction) :
# 		#take the operand type and pop off the appropriate stack
# 
# 		global overflowMessagesOn
# 
# 		if operandNumber == 0 :
# 			#print "oper 0 getOperValue type = " + str (instruction.operand0Type)
# 			if instruction.operand0Type == atom.AtomType.real :
# 				if len (self.machineStack) > 0 :
# 					return (self.machineStack.pop (), instruction.operand0Type)
# 
# 			elif instruction.operand0Type == atom.AtomType.int :
# 				if len (self.machineStack) > 0 :
# 					return (self.machineStack.pop (), instruction.operand0Type)
# 
# 			elif instruction.operand0Type == atom.AtomType.bool :
# 				if len (self.machineStack) > 0 :
# 					return (self.machineStack.pop (), instruction.operand0Type)
# 
# 			else :
# 				if overflowMessagesOn :
# 					print "Program: getOperValue: instruction.operand0Type is unknown atom.AtomType " + str(instruction.operand0Type)
# 
# 		elif operandNumber == 1 :
# 			if instruction.operand1Type == atom.AtomType.real :
# 				if len (self.machineStack) > 0 :
# 					return (self.machineStack.pop (), instruction.operand1Type)
# 
# 			elif instruction.operand1Type == atom.AtomType.int :
# 				if len (self.machineStack) > 0 :
# 					return (self.machineStack.pop (), instruction.operand1Type)
# 
# 			elif instruction.operand1Type == atom.AtomType.bool :
# 				if len (self.machineStack) > 0 :
# 					return (self.machineStack.pop (), instruction.operand1Type)
# 
# 			else :
# 				if overflowMessagesOn :
# 					print "Program: getOperValue: instruction.operand1Type is invalid atom.AtomType " + str (instruction.operand1Type)
# 				return (None, None)
# 
# 		else :
# 			if overflowMessagesOn :
# 				print "Program: getOperValue: invalid operand number " + str (operandNumber)
# 			return (None, None)

	def getOperValue (self, operandNumber, instruction) :
		#take the operand type and pop off the appropriate stack

		global overflowMessagesOn

		if operandNumber == 0 :
			#print "oper 0 getOperValue type = " + str (instruction.operand0Type)
			if instruction.operand0Type == atom.AtomType.real :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return (float (pop[0]), instruction.operand0Type)

			elif instruction.operand0Type == atom.AtomType.int :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return (int (pop[0]), instruction.operand0Type)

			elif instruction.operand0Type == atom.AtomType.bool :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return (bool (pop[0]), instruction.operand0Type)

			elif instruction.operand0Type == atom.AtomType.any :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return pop

			else :
				if overflowMessagesOn :
					print "Program: getOperValue: instruction.operand0Type is unknown atom.AtomType " + str(instruction.operand0Type)

		elif operandNumber == 1 :
			if instruction.operand1Type == atom.AtomType.real :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return (float (pop[0]), instruction.operand1Type)

			elif instruction.operand1Type == atom.AtomType.int :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return (int (pop[0]), instruction.operand1Type)

			elif instruction.operand1Type == atom.AtomType.bool :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return (bool (pop[0]), instruction.operand1Type)

			elif instruction.operand1Type == atom.AtomType.any :
				if len (self.machineStack) > 0 :
					pop = self.machineStack.pop ()
					return pop

			else :
				if overflowMessagesOn :
					print "Program: getOperValue: instruction.operand1Type is invalid atom.AtomType " + str (instruction.operand1Type)
				return (None, None)

		else :
			if overflowMessagesOn :
				print "Program: getOperValue: invalid operand number " + str (operandNumber)
			return (None, None)



	def makeValueTypeTuple (self, value) :
			return (value,atom.AtomType ().getValueType (value))


	def execute (self, inputs=None) :

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
		
		def pushToCoercedStack (result, aType, bType) :
			coerced = atom.AtomType().coerce (aType, bType)

			if coerced == 'real' :
				self.machineStack.push (float (result))

			elif coerced == 'int' :
				self.machineStack.push (int (result))

			elif coerced == 'bool' :
				self.machineStack.push (bool (result))

			elif coerced == 'int' :
				if len (self.symbolTable) == 0 :
					if overflowMessagesOn :
						print "pushToCoercedStack: line: " + str (lineNum) + " zero-length symbolTable"
					return
				self.machineStack.push (int (result) % len (self.symbolTable))



		for lineNum,instr in enumerate (self.instructions) :

			#print "\n"
# 			print "len stack =" + str (len (self.stack))
# 			print "len stack =" + str (len (self.stack))
# 			print "len stack =" + str (len (self.stack))
# 			print "len stack =" + str (len (self.stack))
# 
# 			print "Program: execute: >>> " + Instruction.strings [instr.opCode]

# 			if instr.opCode in [Instruction.pushConst, Instruction.pushFromSym, Instruction.pushFromInput] :   #instr.setRegister2 is not None :
# 				# for instructions setting register2, no operands will be evaluated
# 				self.register2 = instr.setRegister2
# 				self.register2Type = instr.register2Type
# 
# 
# 			else :
# 			self.register2 = None
# 			self.register2Type = None

			if Instruction.specs[instr.opCode]['arity']  == InstructionArgumentType.binary :
				self.operand0 = self.machineStack.pop ()
				self.operand1 = self.machineStack.pop ()

			elif Instruction.specs[instr.opCode]['arity'] == InstructionArgumentType.unary :
				self.operand0 = self.machineStack.pop ()

			if instr.opCode == Instruction.assign :   ######## ASSIGN (BINARY) ######
# 
# 				if self.register2 is not None :
# 					if overflowMessagesOn :
# 						print ("Program: execute: instr: " + str (lineNum) + " assign: expected register2 to be None but found " +
# 														str (self.register2))
# 
# 
# 				if self.operand0Type != atom.AtomType.int :
# 					if overflowMessagesOn :
# 						print ("Program: execute: instr: " + str (lineNum) + " assign instruction should have int for first operand.  Instead found '" +
# 							atom.AtomType.strings[self.operand0Type] +
# 							"'.")
# 					print instr
# 					exit (-1)

				if self.symbolTopIndex > 0 :
					destIndex = int (self.operand0) % self.symbolTopIndex

					if destIndex > self.symbolTopIndex :
						self.symbolTopIndex = destIndex

				else :
					destIndex = 0
					self.symbolTopIndex = 0


# 				# if index is > length of the symbol table, extend the symbol table
# 				if destIndex >= len (self.symbolTable) :
# 					extendElements = (destIndex + 1) - len (self.symbolTable)
# 					#print "Program: execute: extending symbol table"
# 					for _ in range (extendElements) :
# 						self.symbolTable.append ((0.0,0))


#################### define the operand types globally

				self.symbolTable[destIndex] = (self.operand1,self.operand1Type)


			elif instr.opCode == Instruction.pushConst :     ####### PUSH CONST  (UNARY) ######

# 				print "Program: execute: instr: " + str (lineNum) + " PUSH LIT"

				if self.operand0 is None :
					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushConst instruction found operand0 set to None. Nothing for pushing."
					continue

				print instr.argument
				(constValue, constType) = self.makeValueTypeTuple (instr.argument)
				print (constValue, constType)

				if constType == atom.AtomType.real :
					self.machineStack.push (float (constValue))

				elif constType == atom.AtomType.int :
					print "pushConst"
					self.machineStack.push (int (constValue))

				elif constType == atom.AtomType.bool :
					self.machineStack.push (bool (constValue))

				else :
					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " ???? operand0Type = " + str (instr.operand0Type) 

			elif instr.opCode == Instruction.pushFromSym :     ####### PUSH SYMBOL  (NULLARY) ######

				#elf.operand0Type = instr.operand0Type
				# push contents of symbol table at index operand0 to specified stack

				(symbolValue, symbolType) = self.symbolTable[int (instr.argument) % self.symbolTopIndex]
				#self.operand0 = symbolValue

				if symbolType == atom.AtomType.real :
					self.machineStack.push (float (symbolValue))

				elif symbolType == atom.AtomType.int :
					self.machineStack.push (int (symbolValue))

				elif symbolType == atom.AtomType.bool :
					self.machineStack.push (bool (symbolValue))

				else :
					if overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " ???? symbolType = " + str (symbolType) 

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

				inputValue = self.inputs[int (instr.argument) % len (self.inputs)]
				self.machineStack.push (float (inputValue))
# 
# 				inputType = instr.operand0Type
# 
# 
# 				if inputType == atom.AtomType.real :
# 					self.stack.push (float (inputValue))
# 
# 				elif inputType == atom.AtomType.int :
# 					self.stack.push (int (inputValue))
# 
# 				elif inputType == atom.AtomType.bool :
# 					self.stack.push (bool (inputValue))
# 
# 				else :
# 					if overflowMessagesOn :
# 						print "Program: execute: instr: " + str (lineNum) + " ???? symbolType = " + str (symbolType)

			elif instr.opCode == Instruction.popToOperand  :     ####### POP to OPERAND ######
				# pop contents of specified stack to self.operand0 or operand1
				if instr.argument % 2 == 0 :
					self.operand0 = self.machineStack.pop ()

				elif instr.argument % 2 == 1 :
					self.operand1 = self.machineStack.pop ()


# 				if instr.operand0Type == atom.AtomType.real :
# 					if len (self.stack) > 0 :
# 						self.operand0 = self.stack.pop ()
# 
# 					else :
# 						self.operand0 = 0.0
# 
# 				elif instr.operand0Type == atom.AtomType.int :
# 					if len (self.stack) > 0 :
# 						self.operand0 = self.stack.pop ()
# 
# 					else :
# 						self.operand0 = 0.0
# 
# 				elif instr.operand0Type == atom.AtomType.bool :
# 					if len (self.stack) > 0 :
# 						self.operand0 = self.stack.pop ()
# 
# 					else :
# 						self.operand0 = 0.0

			elif instr.opCode == Instruction.add  :    ####### ADD ######
				self.returnValue = self.operand0 + self.operand1
				pushToCoercedStack (self.returnValue, self.operand0Type, self.operand1Type)


			elif instr.opCode == Instruction.sub  :	####### SUB ######
				self.returnValue = self.operand0 - self.operand1
				pushToCoercedStack (self.returnValue, self.operand0Type, self.operand1Type)


			elif instr.opCode == Instruction.mult :	####### MULT #####
				self.returnValue = self.operand0 * self.operand1
				pushToCoercedStack (self.returnValue, self.operand0Type, self.operand1Type)


			elif instr.opCode == Instruction.div  :	####### DIV ######
				if self.operand1 != 0.0 :
					self.returnValue = self.operand0 / self.operand1

				else :
					self.returnValue = 0
				pushToCoercedStack (self.returnValue, self.operand0Type, self.operand1Type)

			elif instr.opCode == Instruction.pow :	####### POW ######
				if (self.operand0 == 0.0 and
				   self.operand1 == 0.0 ) :
					self.returnValue = 0.0
					pushToCoercedStack (self.returnValue, self.operand0Type, -1)

				maxAllowed = 10.0
				if self.operand0 + self.operand1 > maxAllowed :
					self.operand0 = (self.operand0/float(self.operand0+self.operand1)) * maxAllowed
					self.operand1 = (self.operand1/float(self.operand0+self.operand1)) * maxAllowed

				if abs (self.operand0) > 200 :
					sign = self.operand0 / abs (self.operand0)
					self.operand0 = 200.0 * sign

				if abs (self.operand1) > 100 :
					sign = self.operand1 / abs (self.operand1)
					self.operand1 = 100.0 * sign

				if self.operand0 < 0.0 :
					self.operand0 = abs (self.operand0)

				if (self.operand0 == 0.0 and
					self.operand1 < 0.0) :
					self.returnValue = 0.0

				else :
					#print "a^b a = " + str(a) + " " + str(b)
					self.returnValue = self.operand0**self.operand1

				pushToCoercedStack (self.returnValue, self.operand0Type, -1)

			elif instr.opCode == Instruction.sin :	####### SIN ######

				self.returnValue = math.sin (self.operand0)
				pushToCoercedStack (self.returnValue, self.operand0Type, -1)

			elif instr.opCode == Instruction.cos :	####### COS ######

				self.returnValue = math.cos (self.operand0)
				pushToCoercedStack (self.returnValue, self.operand0Type, -1)

			else :
				if overflowMessagesOn :
					assert False, "Program: execute: instr: " + str (lineNum) + " CRASH: execute2: instruction " + str (instr.opCode) + " not defined."

			#self.operand0 = None
		#print "Program: execute: stack =" + str (self.stack)
# 		print "Program: execute: stack =" + str (self.stack)
# 		print "Program: execute: stack =" + str (self.stack)
# 		print "Program: execute: stack =" + str (self.stack)

# 		print "______out of instruction loop________"
# 		print "len stack =" + str (len (self.stack))
# 		print "len stack =" + str (len (self.stack))
# 		print "len stack =" + str (len (self.stack))
# 		print "len stack =" + str (len (self.stack))

		#print "Program: execute: symbolTable = " + str (self.symbolTable)
		return self.machineStack

	def instructionsToFlatBinary (self) :

		flatBinary = ''
		for thisInstruction in self.instructions :
			breakdown = []
			opCodeBinStr = atom.BinaryString (precision=Instruction ().opCodePrecision, rangeIn=Instruction ().opCodeTypeRange)
			opCodeBinStr.setFromInt(thisInstruction.opCode)
			opCodeBinary = opCodeBinStr.binaryStr
			flatBinary += opCodeBinary
			breakdown.append (len (opCodeBinary))
			assert len (opCodeBinary) == Instruction ().opCodePrecision, ("instructionsToFlatBinary: opCode should be " +
														str (self.opCodePrec) +
														" bits in length, but is " +
														str (len (opCodeBinary)))

# 			##### OPERAND 0 type 
# 			oper0Type = atom.BinaryString (precision=Instruction ().operTypePrec, rangeIn=Instruction ().operTypeRange)
# 			oper0Type.setFromInt (thisInstruction.operand0Type)
# 			oper0TypeBinary = oper0Type.binaryStr
# 
# 			flatBinary += oper0TypeBinary
# 			breakdown.append (len (oper0TypeBinary))
# 			assert len (oper0TypeBinary) == Instruction ().operTypePrec, ("instructionsToFlatBinary: operand0 type should be " +
# 																str (self.operTypePrec) +
# 																" bits in length, but is " +
# 																str (len (oper0TypeBinary)))
# 
# 			##### OPERAND 1 type
# 			oper1Type = atom.BinaryString (precision=Instruction ().operTypePrec, rangeIn=Instruction ().operTypeRange)
# 			oper1Type.setFromInt (thisInstruction.operand1Type)
# 			oper1TypeBinary = oper1Type.binaryStr
# 
# 			flatBinary += oper1TypeBinary
# 			breakdown.append (len (oper1TypeBinary))
# 			assert len (oper1TypeBinary) == Instruction ().operTypePrec, ("instructionsToFlatBinary: operand1 type should be " +
# 																str (self.operTypePrec) +
# 																" bits in length, but is " +
# 																str (len (oper1TypeBinary)))
###### change resiger to argument below
			##### argument type
			argumentType = atom.BinaryString (precision=Instruction ().operTypePrec, rangeIn=Instruction ().operTypeRange)
			argumentType.setFromInt (thisInstruction.argumentType)
			argumentTypeBinary = argumentType.binaryStr

			flatBinary += argumentTypeBinary
			breakdown.append (len (argumentTypeBinary))
			assert len (argumentTypeBinary) == Instruction ().operTypePrec, ("instructionsToFlatBinary: argument type should be " +
																str (self.operTypePrec) +
																" bits in length, but is " +
																str (len (argumentTypeBinary)))

			##### set argument

			if thisInstruction.argumentType == atom.AtomType.real :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.realMax])
				argument.setFromReal (thisInstruction.argument)

			elif thisInstruction.argumentType == atom.AtomType.int :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
				argument.setFromInt (thisInstruction.argument)

			elif thisInstruction.argumentType == None :
				argument = atom.BinaryString (precision=Instruction ().argumentPrecision, rangeIn=[0,self.intMax])
				argument.setToNil ()

			else :
				if overflowMessagesOn :
					print "instructionsToFlatBinary: unsupported argumentType (should be real, int, or None"
				exit (-1)

			argumentBinary = argument.binaryStr
			flatBinary += argumentBinary
			breakdown.append (len (argumentBinary))
			assert len (argumentBinary) == Instruction ().argumentPrecision, ("instructionsToFlatBinary: argument type should be " +
																str (Instruction ().argumentPrecision) +
																" bits in length, but is " +
																str (len (argumentBinary)))

		return flatBinary


	def flatBinaryToInstructions (self, binaryIn) :
		#print "flatBinaryToInstructions"
		instructions = []
		instructionSize = Instruction ().opCodePrecision + Instruction ().operTypePrec + Instruction ().operTypePrec + Instruction ().registerSetPrecision + Instruction ().operTypePrec
		numInstructions = len (binaryIn) / instructionSize

		global overflowMessagesOn

		startBit = 0
		endBit = 0
		for instruction in range (numInstructions) :
			breakdown = []
			startBit = endBit
			endBit = startBit + Instruction ().opCodePrecision
			opCode = atom.BinaryString (rangeIn=[0,len (Instruction.strings)], binaryStrIn=binaryIn[startBit:endBit])
			if opCode.toInt () is None :
				opCode = atom.BinaryString (rangeIn=[0,len (Instruction.strings)], binaryStrIn='0' * Instruction ().opCodePrecision)
			breakdown.append (len (binaryIn[startBit:endBit]))
# 			
# 			
# 			startBit = endBit
# 			endBit = startBit + Instruction ().operTypePrec
# 			operand0Type = atom.BinaryString (rangeIn=[0,len (atom.AtomType.strings)], binaryStrIn=binaryIn[startBit:endBit])
# 			breakdown.append (len (binaryIn[startBit:endBit]))
# 
# 			startBit = endBit
# 			endBit = startBit + Instruction ().operTypePrec
# 			operand1Type = atom.BinaryString (rangeIn=[0,len (atom.AtomType.strings)], binaryStrIn=binaryIn[startBit:endBit])
# 			breakdown.append (len (binaryIn[startBit:endBit]))
# 
# 			startBit = endBit
# 			endBit = startBit + Instruction ().operTypePrec

			argumentTypeBString = atom.BinaryString (rangeIn=[0,len (atom.AtomType.strings)], binaryStrIn=binaryIn[startBit:endBit])
			breakdown.append (len (binaryIn[startBit:endBit]))
			argumentType = argumentTypeBString.toInt ()

			startBit = endBit
			endBit = startBit + Instruction ().argumentPrecision

			if argumentType == atom.AtomType.real :
				argument = atom.BinaryString (rangeIn=[0,self.realMax], binaryStrIn=binaryIn[startBit:endBit])
				breakdown.append (len (binaryIn[startBit:endBit]))
				argumentValue = argument.toReal ()

			elif argumentType == atom.AtomType.int :
				argument = atom.BinaryString (rangeIn=[0,self.intMax], binaryStrIn=binaryIn[startBit:endBit])
				breakdown.append (len (binaryIn[startBit:endBit]))
				argumentValue = argument.toInt ()

			elif argumentType == None :
				argumentValue = None

			else :
				argumentType = 0
				argument = atom.BinaryString (rangeIn=[0,self.realMax], binaryStrIn=binaryIn[startBit:endBit])
				breakdown.append (len (binaryIn[startBit:endBit]))
				argumentValue = argument.toReal ()
# 
# 			#  fix them up so they work
# 			operand0TypeValue = operand0Type.toInt ()
# 			operand1TypeValue = operand1Type.toInt ()
# 			if opCode.toInt () >= len (Instruction ().specs.keys ()) :
# 				opCode = atom.BinaryString (rangeIn=[0,len (Instruction.strings)], binaryStrIn='0' * len (Instruction ().specs.keys ()))
# 
			opCodeString = Instruction ().strings[opCode.toInt ()]
# 			if Instruction.specs[opCodeString]['arity'] == InstructionArgumentType.nullary :
# 				if operand0TypeValue is not None :
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper0 to None"
# 					operand0TypeValue = None
# 
# 				if operand1TypeValue is not None :
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper1 to None"
# 					operand1TypeValue = None
# 
# 			elif Instruction.specs[opCodeString]['arity'] == InstructionArgumentType.unary :
# 
# 				if operand1TypeValue is not None :
# 					operand1TypeValue = None
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper1 to None"
# 
# 				if operand0TypeValue > 3 :
# 					operand0TypeValue = [atom.AtomType.real,atom.AtomType.int][int (random.random () * 2)]
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper0 to " + str (atom.AtomType.strings[operand0TypeValue])
# 
# 				if operand0TypeValue is None :
# 					operand0TypeValue = [atom.AtomType.real,atom.AtomType.int][int (random.random () * 2)]
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper0 to " + str (atom.AtomType.strings[operand0TypeValue])
# 
			if opCodeString == 'pushConst' :
				if argumentValue is None :
					if argumentType == atom.AtomType.real :
						argumentValue = random.random () * self.realMax

					elif argumentType == atom.AtomType.int :
						argumentValue = int (random.random () * self.intMax)
# 
			if opCodeString == 'pushFromSym' :
				if argumentValue is None :
					argumentValue = int (random.random () * len (self.symbolTable))
# 
# 			elif Instruction.specs[opCodeString]['arity'] == InstructionArgumentType.binary :
# 				if operand0TypeValue is None :
# 					operand0TypeValue = [atom.AtomType.real,atom.AtomType.int][int (random.random () * 2)]
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper0 to " + str (atom.AtomType.strings[operand0TypeValue])
# 
# 				if operand1TypeValue is None :
# 					operand1TypeValue = [atom.AtomType.real,atom.AtomType.int][int (random.random () * 2)]
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper1 to " + str (atom.AtomType.strings[operand1TypeValue])
# 
# 				if operand0TypeValue > 3 :
# 					operand0TypeValue = [atom.AtomType.real,atom.AtomType.int][int (random.random () * 2)]
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper0 to " + str (atom.AtomType.strings[operand0TypeValue])
# 
# 				if operand1TypeValue > 3 :
# 					operand1TypeValue = [atom.AtomType.real,atom.AtomType.int][int (random.random () * 2)]
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper1 to " + str (atom.AtomType.strings[operand1TypeValue])
# 
# 			if opCodeString == 'assign' :
# 				if operand0TypeValue != atom.AtomType.int :
# 					operand0TypeValue = atom.AtomType.int
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing oper0 to " + str (atom.AtomType.strings[operand0TypeValue])

# 				if argumentValue is not None :
# 					argumentValue = None
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing argument to None"
# 
# 				if argumentType is not None :
# 					argumentType = None
# 					argumentType = None
# 					if overflowMessagesOn :
# 						print "flatBinaryToInstructions: changing argumentType to None"


# 			if (overflowMessagesOn and
# 				opCodeString == 'assign' ) :
# 				print "fix opCodeIn=          ASSIGN"
# 				print "fix operand0TypeIn= " + str (atom.AtomType.strings[operand0TypeValue]    )
# 				print "fix operand1TypeIn= " + str (atom.AtomType.strings[operand1TypeValue]    )
# 				print "fix setoperand0=   " + str (registerSetValue )
# 				if registerType is not None :
# 					print registerType
# 					print "fix operand0Type=  " + str (atom.AtomType.strings[registerType])
# 
# 				else :
# 					print "fix operand0Type=  None"

			instructions.append (Instruction (opCode.toInt (),argumentValue,argumentType))
		self.instructions = instructions


	def test (self) :

		#opCodePrec = Instruction.opCodePrecision # opCode/operator precision = 4
		#operPrec = Instruction.operandPrecision # operand precision = 32

		#self.symbolTable = [(0,0),(0,0),(0,0),(0,0),(0,0)] #inputValue, 0.5, 0.2, 0.3, 3.0]
		#self.symbolTable = [inputValue, 3.0]

		
		self.instructions = []
		#var0 = test2
		print "test: define instr 1..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=int(0)))
		print "test: define instr 2..."
		#push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushFromInput,
												argument=0)) # push input 0 to real stack

		print "test: define instr 3..."
		#pop stack to symbol at index from stack
		self.instructions.append (Instruction ( Instruction.assign,
												argument=self.symbolTopIndex + 1))


		self.machineStack.printValues ()
		self.execute([0])
		print self.machineStack.array[0]
		self.machineStack.printValues ()
		exit (0)

# 			#var1 = 3.0
		print "test: define instr 4..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=int(1)))

		print "test: define instr 5..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=float(3.0)))

		print "test: define instr 6..."
		#pop stack to symbol at index from stack
		self.instructions.append (Instruction (	Instruction.assign,
												argument=self.symbolTopIndex + 1))

# 			#var2 = 0.4
		print "test: define instr 7..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=int(2)))

		print "test: define instr 8..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=float(0.4)))

		print "test: define instr 9..."
		#pop stack to symbol at index from stack
		self.instructions.append (Instruction (	Instruction.assign,
												argument=self.symbolTopIndex + 1))

# # 			#var3 = 0.2
		print "test: define instr 7..."
		#set self.operand0 to 3 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=int(3)))

		print "test: define instr 8..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=float(0.2)))

		print "test: define instr 9..."
		#pop stack to symbol at index from stack
		self.instructions.append (Instruction (	Instruction.assign,
												argument=self.symbolTopIndex + 1))

# 			#var4 = 0.5
		print "test: define instr 7..."
		#set self.operand0 to 3 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=int(4)))

		print "test: define instr 8..."
		#set self.operand0 to 4 and push self.operand0 to stack
		self.instructions.append (Instruction ( Instruction.pushConst,
												argument=float(0.5)))

		print "test: define instr 9..."
		#pop stack to symbol at index from stack
		self.instructions.append (Instruction (	Instruction.assign,
												argument=self.symbolTopIndex + 1))


		#push symbols onto stack
		#push symbol 2 to stack
		print "test: define instr 12..."
		self.instructions.append (Instruction ( Instruction.pushFromSym,
												argument=int(2)))

		print "test: define instr 10..."
		#push symbol 0 to stack
		self.instructions.append (Instruction ( Instruction.pushFromSym,
												argument=int(1)))

		#push symbol 1 to stack
		print "test: define instr 11..."
		self.instructions.append (Instruction ( Instruction.pushFromSym,
												argument=int(0)))


		#pow
		print "test: define instr 13..."
		self.instructions.append (Instruction ( Instruction.pow))

		# mult
		print "test: define instr 14..."
		self.instructions.append (Instruction ( Instruction.mult))


		print "test: define instr 10..."
		#push symbol 0 to stack
		self.instructions.append (Instruction ( Instruction.pushFromSym,
												argument=int(3)))

		#push symbol 1 to stack
		print "test: define instr 11..."
		self.instructions.append (Instruction ( Instruction.pushFromSym,
												argument=int(0)))
		# mult
		print "test: define instr 15..."
		self.instructions.append (Instruction ( Instruction.mult)) 

		# add
		print "test: define instr 16..."
		self.instructions.append (Instruction ( Instruction.add)) 

		#push symbol 4 to stack
		print "test: define instr 11..."
		self.instructions.append (Instruction ( Instruction.pushFromSym,
												argument=int(4)))

		# add
		print "test: define instr 16..."
		self.instructions.append (Instruction ( Instruction.add)) 


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
			print "test: _______about to call execute________"

			result = self.execute ([test2])
			print "test: result"
			print result
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
		
		print samples

		plotFunction ()

prog = Program()
prog.test ()
