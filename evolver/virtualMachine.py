
import random
import math
import types
import numpy as np
from pylab import figure, show

import parameters as p
import atom


# import sys
#sys.path.append ('/Users/rhett/workspace/evolver')

class RandomCache :
	#random.seed (0.77777777)
	random.seed (p.parameters["randomSeed"])

	def __init__ (self, numValues) :
		self.randomIndex = 0
		self.numRandValues = numValues
		self.randValues = [random.random () for _ in xrange (self.numRandValues)]

	def getRandomValue (self) :
		self.randomIndex += 1
		return self.randValues[self.randomIndex % self.numRandValues]

	def setRandomSeed (self, randomIndexIn) :
		self.randomIndex = int (randomIndexIn) % self.numRandValues

r = RandomCache (numValues=1000000)


def rangeRandReal () :
	rnge = p.parameters['realMax'] - p.parameters['realMin']
	return random.random () * rnge + p.parameters['realMin']

def rangeRandInt () :
	rnge = p.parameters['intMax'] - p.parameters['intMin']
	return int (random.random () * rnge + p.parameters['intMin'])

def randInvert () :
	return [-1,1][int (random.random () * 2)]

class InstructionOperandType :
	nullary = 0
	unary = 1
	binary = 2

	strings = [ "nullary",
				"unary",
				"binary"]

	precision = atom.BinaryString().rangeToPrecision ([0,len (strings) - 1])

	def toStr (self, InstructionArgumentIn) :
		if InstructionArgumentIn >= len (self.strings) :
			raise Exception, "toStr: InstructionArgumentIn not found in InstructionOperandTypes.strings."
		return self.strings[InstructionArgumentIn]

	def toInstructionOperandType (self, stringIn) :
		if not stringIn in self.strings :
			raise Exception, "toInstructionOperandType: stringIn not found in InstructionOperandType.strings."
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
	log10 = 16
	mod = 17
	abs = 18
	min = 19
	max = 20

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
								cos,
								log10,
								mod,
								abs,
								min,
								max
							]
	# other instructions: tan, max, min, clamp, logE, log2, 
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
				"cos",
				"log10",
				"mod",
				"abs",
				"min",
				"max"]


	opCodeTypeRange = [0,len (strings) - 1]
	opCodePrecision = atom.BinaryString().rangeToPrecision (opCodeTypeRange)
	argTypeRange = [0,len (atom.AtomType.strings) - 1]
	argTypePrecision = atom.BinaryString().rangeToPrecision (argTypeRange)

	argumentPrecision = 32

	instructionSize =  (opCodePrecision +
						argTypePrecision +
						argumentPrecision +
						argumentPrecision)

	def toStr (self, intValueIn) :
		return self.Instr[intValueIn]

	def toInstruction (self, stringIn) :
		if not stringIn in self.strings :
			raise Exception, "toInstruction: stringIn not found in Instruction.strings."
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
							'resultType': atom.AtomType.any },

			log10:     	{'arity': InstructionOperandType.unary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': None,
							'argType': None,
							'resultType': atom.AtomType.any },

			mod:     	{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any },

			abs:     	{'arity': InstructionOperandType.unary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': None,
							'argType': None,
							'resultType': atom.AtomType.any },

			min:     	{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any },

			max:     	{'arity': InstructionOperandType.binary,
							'oper0Type': atom.AtomType.any,
							'oper1Type': atom.AtomType.any,
							'argType': None,
							'resultType': atom.AtomType.any }}



	def __init__ (self, opCodeIn=None, argument=None, argumentDelta=None) :

		if isinstance (argument, atom.BinaryString) :
			raise Exception, "binary string!!!"

		self.realMax = p.parameters['realMax']
		self.realMin = p.parameters['realMin']
		self.intMax = p.parameters['intMax']
		self.intMin = p.parameters['intMin']

		if opCodeIn is None :
			# opcode will be None if no args are passed to the __init__
			self.opCode = self.noOp
			self.argument = None
			self.argumentDelta = None
			return

		else :
			if opCodeIn not in self.specs.keys () :
				raise Exception, (	"Instruction.__init__() unknown opCode '" +
									str (opCodeIn) +
									"'.")

			self.opCode = opCodeIn # expects an int

			###### validate argument value against spec type #########
			if self.specs[self.opCode]['argType'] == None :
				if argument is not None :
					raise Exception, (	"Instruction.__init__() argument should be None, but was '" +
										str (argument) +
										"' for opCode " +
										self.strings[opCodeIn] +
										".")

				else :
					self.argument = None
					self.argumentDelta = None

			elif self.specs[self.opCode]['argType'] == atom.AtomType.any :
				self.argument = argument

			elif self.specs[self.opCode]['argType'] == atom.AtomType.int :
				if type (argument) != types.IntType :
					raise Exception, (	"Instruction.__init__() argument should be int, but was '" +
										str (argument) +
										"' for opCode " +
										self.strings[opCodeIn] +
										".")

				else :
					self.argument = argument
					self.argumentDelta = int (argument * p.parameters["argumentDeltaInit"])

			elif self.specs[self.opCode]['argType'] == atom.AtomType.real :
				if type (argument) != types.FloatType :
					raise Exception, (	"Instruction.__init__() argument should be real, but was '" +
										str (argument) +
										"' for opCode " +
										self.strings[opCodeIn] +
										".")

				else :
					self.argument = argument

			else :
				raise Exception, (	"Instruction.__init__() unknown argType in specs for opCode '" +
									str (self.specs[self.opCode]['name']) +
									"'.")

			# check the argumentDelta type against te argumentType
			if type (argumentDelta) != type (argument) :
				raise Exception, (	"Instruction.__init__() argumentDelta should be '" +
									str (type (argument)) +
									"', but was '" +
									str (type (argumentDelta)) +
									"' for opCode " +
									self.strings[opCodeIn] +
									".")

			self.argumentDelta = argumentDelta

	def repairArgument (self, opCode, argumentValue, argumentDelta, argumentType) :
		if opCode.toInt () == self.assign :
			if type (argumentValue) != types.IntType :
				argumentValue = int (random.random () * p.parameters['maxSymbols'])
				argumentDelta = 0

		elif opCode.toInt () == self.pushFromSym :
			if type (argumentValue) != types.IntType :
				argumentValue = int (random.random () * p.parameters['maxSymbols'])
				argumentDelta = 0

		elif opCode.toInt () == self.pushFromInput :
			if type (argumentValue) != types.IntType :
				argumentValue = int (random.random () * p.parameters['maxSymbols'])
				argumentDelta = 0

		elif opCode.toInt () == self.popToOperand :
			if type (argumentValue) != types.IntType :
				numOperands = 2
				argumentValue = int (random.random () * numOperands)
				argumentDelta = 0

		elif opCode.toInt () in [	self.noOp,
									self.popToReturn,
									self.pushAllSym,
									self.rand,
									self.add, 
									self.sub, 
									self.mult,
									self.div, 
									self.pow, 
									self.sin, 
									self.cos,
									self.log10,
									self.mod,
									self.abs,
									self.min,
									self.max] :

			argumentValue = None
			argumentDelta = None

		elif opCode.toInt () == self.pushConst :
			if argumentValue is None :
				if argumentType == atom.AtomType.real :
					argumentValue = rangeRandReal ()
					argumentDelta = 0.0#5

				elif argumentType == atom.AtomType.int :
					argumentValue = rangeRandInt ()
					argumentDelta = 0

		else :
			print "repair argument: Stop! unknown opcode " + str (opCode.toInt ())

		return argumentValue, argumentDelta

	def driftArgument (self) :
		if (self.argument is not None and
			self.argumentDelta is not None) :

			self.argument += self.argumentDelta

	def toBinary (self) :
		overflowMessagesOn = p.parameters["overflowMessagesOn"]

		string = ''

		opCodeBinStr = atom.BinaryString (precision=self.opCodePrecision, rangeIn=self.opCodeTypeRange)
		opCodeBinStr.setFromInt(self.opCode)
		opCodeBinary = opCodeBinStr.binaryStr
		string += opCodeBinary
		if len (opCodeBinary) != self.opCodePrecision :
			raise Exception, (	"instructionsToFlatBinary: opCode should be " +
								str (self.opCodePrec) +
								" bits in length, but is " +
								str (len (opCodeBinary)))

		argType = Instruction.specs[self.opCode]['argType']

		##### argument type
		argumentType = atom.BinaryString (precision=self.argTypePrecision, rangeIn=self.argTypeRange)
		argumentType.setFromInt (argType)
		argumentTypeBinary = argumentType.binaryStr

		string += argumentTypeBinary

		if len (argumentTypeBinary) != self.argTypePrecision :
			raise Exception, (	"instructionsToFlatBinary: argument type should be " +
								str (self.argTypePrecision) +
								" bits in length, but is " +
								str (len (argumentTypeBinary)))

		##### set argument
# 				real = 0
# 				int = 1
# 				bool = 2
# 				nil = 3
# 				any = 4
		if argType == atom.AtomType.any :
			if   type (self.argument) == types.IntType :
				argType = atom.AtomType.int

			elif type (self.argument) == types.FloatType :
				argType = atom.AtomType.real

			elif type (self.argument) == types.BooleanType :
				argType = atom.AtomType.bool

			elif type (self.argument) == types.NoneType :
				argType = None

			else :
				raise Exception, "argument not recognized"

		realRange = [self.realMin,self.realMax]
		intRange = [self.intMin,self.intMax]

		if argType == atom.AtomType.real :
			argument = atom.BinaryString (precision=self.argumentPrecision, rangeIn=realRange)
			argument.setFromReal (self.argument)
			argumentDelta = atom.BinaryString (precision=self.argumentPrecision, rangeIn=realRange)
			argumentDelta.setFromReal( self.argumentDelta)

		elif argType == atom.AtomType.int :
			argument = atom.BinaryString (precision=self.argumentPrecision, rangeIn=intRange)
			argument.setFromInt (self.argument)
			argumentDelta = atom.BinaryString (precision=self.argumentPrecision, rangeIn=intRange)
			argumentDelta.setFromInt (self.argumentDelta)

		elif argType == atom.AtomType.bool :
			argument = atom.BinaryString (precision=self.argumentPrecision, rangeIn=intRange)
			argument.setFromInt (self.argument)
			argumentDelta = atom.BinaryString (precision=self.argumentPrecision, rangeIn=intRange)
			argumentDelta.setFromInt (self.argumentDelta)

		elif argType is None :
			argument = atom.BinaryString (precision=self.argumentPrecision, rangeIn=intRange)
			argument.setToNil ()
			argumentDelta = atom.BinaryString (precision=self.argumentPrecision, rangeIn=intRange)
			argumentDelta.setToNil ()

		else :
			if overflowMessagesOn :
				raise Exception, "instructionsToFlatBinary: unsupported argumentType (should be real, int, or None"

		self.argument = argument
		argumentBinary = argument.binaryStr
		string += argumentBinary

		self.argumentDelta = argumentDelta
		argumentDeltaBinary = argumentDelta.binaryStr
		string += argumentDeltaBinary

		if len (argumentBinary) != self.argumentPrecision :
			raise Exception, (	"instructionsToFlatBinary: argument type should be " +
								str (self.argumentPrecision) +
								" bits in length, but is " +
								str (len (argumentBinary)))
		if len (argumentDeltaBinary) != self.argumentPrecision :
			raise Exception, (	"instructionsToFlatBinary: argument delta type should be " +
								str (self.argumentPrecision) +
								" bits in length, but is " +
								str (len (argumentDeltaBinary)))

		return string

	def fromBinary (self, binaryIn) :
# 		overflowMessagesOn = p.parameters["overflowMessagesOn"]

		opCodeStart = 0
		######## OPCODE ######
		opCodeEnd = self.opCodePrecision

		opCode = atom.BinaryString (rangeIn=self.opCodeTypeRange, binaryStrIn=binaryIn[opCodeStart:opCodeEnd])

		if opCode.toInt () is None :
			##### case in which a valid op code has been mutated to a None (11111) and needs to be made valid (switched to 0000)
			opCode = atom.BinaryString (rangeIn=self.opCodeTypeRange, binaryStrIn='0' * self.opCodePrecision)

		elif opCode.toInt () > len (Instruction.strings) - 1 :
			raise Exception, "invalid opcode " + str (opCode.toInt ())
# 			exit(0)
# 			opCode = atom.BinaryString (rangeIn=self.opCodeTypeRange, binaryStrIn='0' * self.opCodePrecision)

		self.opCode = opCode.toInt ()

		####### ARGUMENT TYPE ######
		argTypeStart = opCodeEnd
		argTypeEnd = argTypeStart + self.argTypePrecision
		typeListRange = [0,len (atom.AtomType.strings) - 1]
		argumentTypeBString = atom.BinaryString (rangeIn=typeListRange, binaryStrIn=binaryIn[argTypeStart:argTypeEnd])
		argumentType = argumentTypeBString.toInt ()

		realRange = [self.realMin,self.realMax]
		intRange = [self.intMin,self.intMax]

		####### ARGUMENT VALUE ######
		argValueStart = argTypeEnd
		argValueEnd = argValueStart + self.argumentPrecision

		argDeltaStart = argValueEnd
		argDeltaEnd = argDeltaStart + self.argumentPrecision

		argSlice = binaryIn[argValueStart:argValueEnd]
		argDeltaSlice = binaryIn[argDeltaStart:argDeltaEnd]

		if argumentType == atom.AtomType.real :
			argument = atom.BinaryString (rangeIn=realRange, binaryStrIn=argSlice)
			argumentValue = argument.toReal ()
			argumentDelta = atom.BinaryString (rangeIn=realRange, binaryStrIn=argDeltaSlice)
			argumentDeltaValue = argumentDelta.toReal ()

		elif argumentType == atom.AtomType.int :
			argument = atom.BinaryString (rangeIn=intRange, binaryStrIn=argSlice)
			argumentValue = argument.toInt ()
			argumentDelta = atom.BinaryString (rangeIn=intRange, binaryStrIn=argDeltaSlice)
			argumentDeltaValue = argumentDelta.toInt ()

		elif argumentType == atom.AtomType.bool :
			argument = atom.BinaryString (rangeIn=intRange, binaryStrIn=argSlice)
			argumentValue = argument.toInt ()
			argumentDelta = atom.BinaryString (rangeIn=intRange, binaryStrIn=argDeltaSlice)
			argumentDeltaValue = argumentDelta.toInt ()

		elif argumentType == None :
			argumentValue = None
			argumentDeltaValue = None

		elif argumentType == atom.AtomType.nil :
			argumentValue = None
			argumentDeltaValue = None

		elif argumentType == atom.AtomType.any :
			argument = atom.BinaryString (rangeIn=realRange, binaryStrIn=argSlice)
			argumentValue = argument.toReal ()
			argumentDelta = atom.BinaryString (rangeIn=realRange, binaryStrIn=argDeltaSlice)
			argumentDeltaValue = argumentDelta.toReal ()

		else :
			raise Exception, "fromBinary: arg type unrecognized = " + str (argumentType)

		self.argument, self.argumentDelta = self.repairArgument (opCode, argumentValue, argumentDeltaValue, argumentType)
		if isinstance (self.argument, atom.BinaryString) :
			raise Exception, "fromBin stop!!!"

	def __repr__ (self) :
		oper0Type = Instruction.specs[self.opCode]['oper0Type']
		oper1Type = Instruction.specs[self.opCode]['oper1Type']
		resultType = Instruction.specs[self.opCode]['resultType']

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

		result = (("   %-7s" % oper0TypeString) + 
				" " +
				("%-7s" % oper1TypeString) +
				" -> " +
				("%-15s" % Instruction.strings [self.opCode]) +
				" (" +
				("%-15s" % str (self.argument)) +
				(" +/- %-15s" % str (self.argumentDelta)) +
				") -> " +
				("%-7s" % resultTypeString) +
				'\n')

		return result

class Stack :
	def __init__ (self, allocatedSize) :
		self.pointer = -1
		self.allocatedSize = allocatedSize
		self.valueArray = np.array(np.zeros (self.allocatedSize), dtype=np.float_)
		self.typeArray = np.array(np.zeros (self.allocatedSize), dtype=np.int_)
		self.overflowMessagesOn = p.parameters["overflowMessagesOn"]

	def length (self) :
		return self.pointer + 1

	def top (self) :
		resultType = self.typeArray[self.pointer]
		resultValue = atom.AtomType ().coerce (self.valueArray[self.pointer], resultType)
		return resultValue, resultType

	def pop (self) :
		if self.pointer >= 0 :
			resultType = self.typeArray[self.pointer]
			resultValue = atom.AtomType ().coerce (self.valueArray[self.pointer], resultType)

			self.pointer -= 1
			return (resultValue, resultType)

		else :
			if self.overflowMessagesOn :
				print "Stack:pop () trying to pop from an empty stack."
			return None, None

	def push (self, item) :
		if (self.pointer < self.allocatedSize or
		    self.pointer == -1) :
			self.pointer += 1
			self.valueArray[self.pointer] = float (item)
			self.typeArray[self.pointer] = atom.AtomType ().getValueType (item)

		else :
			if self.overflowMessagesOn :
				print "Stack:push () Overflow. Trying to push beyond the end of a stack."

	def printValues (self) :
		if self.pointer > -1 :
			print 'Stack ['
			for index in range (self.pointer+2) :
				print self.valueArray[index], atom.AtomType ().strings[self.typeArray[index]]
			print ']'

		else :
			print 'Stack []'

class Program :
	def __init__ (self, randomSeed, expressionSize) :

		self.instructions =   []
		self.symbolTable = []
		self.symbolTopIndex = 0

		self.expressionSize = expressionSize
		self.inputs = []

		self.machineStack = Stack (10000)
		self.programCounter = 0

		self.operand0 = None
		self.operand1 = None
		self.returnValue = None

		self.realMax = p.parameters['realMax']
		self.realMin = p.parameters['realMin']
		self.intMax = p.parameters['intMax']
		self.intMin = p.parameters['intMin']

		self.randomSeed = randomSeed
		self.binaryBreakdown = []
		self.overflowMessagesOn = p.parameters["overflowMessagesOn"]

	def randomInstruction (self) :
		numSpecs = len (Instruction.specs)
		setVarSpecs = 3

		choice = int (random.random () * (numSpecs - setVarSpecs)) + setVarSpecs
		key = Instruction.specs.keys ()[choice]
		return key

	def synthesize (self, permute=False) :
		random.seed (0)
		self.instructions = []

		numConstants = 3
		numWorkingSymbols = 4
		numInputs = p.parameters["numInputs"]
# 		numSymbols = numInputs + numWorkingSymbols + numConstants

		###### instructions to populate the symbol table with input variables #####
		index = 0
		for _ in range (numInputs) :

			# push value from inputs[index] onto the stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushFromInput,
														argument=index,
														argumentDelta=0))
			# assign the stackTop value to symbol[index]
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index,
														argumentDelta=0))

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
														argument=const,
														argumentDelta=const*0))
			# assign TOS to a symbol at index
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index,
														argumentDelta=0))

			index += 1


		###### instructions to populate the symbol table with working variables (init to random) #####
		for _ in range (numInputs,numWorkingSymbols) :

			# push value onto stack
			self.instructions.append (Instruction (   opCodeIn=Instruction.pushConst,
														argument = rangeRandReal (),
														argumentDelta=random.random()*2-1))
			# assign TOS to a symbol at index
			self.instructions.append (Instruction (   opCodeIn=Instruction.assign, 
														argument=index,
														argumentDelta=0))
			index += 1

		###### instruction to push all symbols onto the stack #####
		self.instructions.append (Instruction (   opCodeIn=Instruction.pushAllSym))


		###### instruction to push the input onto the stack ######
		self.instructions.append (Instruction (   opCodeIn=Instruction.pushFromInput, 
														argument=0,
														argumentDelta=0))

		self.numSetupLines = len (self.instructions)
		sizeOfSymTable = index

		self.numLines = p.parameters["numInstructions"]
		permuteCycle = int (random.random () * (len (self.instructions) - 3)) + 3
		instrIndex = 0
		# SYNTHESIZE random instructions
		while len (self.instructions) - self.numSetupLines < self.numLines :
			# instruction
			if permute :
				opCode = int ((permuteCycle + instrIndex % permuteCycle) % len (Instruction.strings))
				instrIndex += 1

			else :
				opCode = int (random.random () * len (Instruction.strings))

			preFragment = None

			##### set the argument ####
			argument = None
			argumentDelta=None
			if Instruction.specs[opCode]['arity'] == InstructionOperandType.nullary :
				if opCode == Instruction.popToOperand :
					numOperands = 2
					index = int (random.random () * numOperands)
					argument = index
					argumentDelta = 0
					preFragment = []

				elif opCode == Instruction.pushConst :
					# randomly choose whether the type should be int or real
					rhsType = [atom.AtomType.real, atom.AtomType.int][int (random.random () * 2)]
					if rhsType == atom.AtomType.real :
						argument = rangeRandReal ()
						argumentDelta = argument * 0.0

					elif rhsType == atom.AtomType.int :
						argument = rangeRandInt ()
						argumentDelta = argument * 0

					else :
						print "synthesize: pushConst unknown constant type"

					preFragment = []

				elif opCode == Instruction.pushFromInput :
					argument = int (random.random () * p.parameters["numInputs"])
					argumentDelta = 0
					preFragment = []

				elif opCode == Instruction.noOp :
					preFragment = []

				elif opCode == Instruction.pushAllSym :
					preFragment = []

				elif opCode == Instruction.popToReturn :
					preFragment = []

				else :
					raise Exception, "unsupported nullary instruction " +str ( Instruction.strings[opCode])

			elif Instruction.specs[opCode]['arity'] == InstructionOperandType.unary :
				preFragment = []

				if opCode == Instruction.pushFromSym :
					# set the argument to the index of the symbolTable to which the value will be assigned
					index = int (random.random () * sizeOfSymTable)
					argument = index
					argumentDelta = 0

				elif opCode == Instruction.assign :
					# randomly choose whether the type should be int or real
					numPossibleTypes = 2
					rhsType = [atom.AtomType.real, atom.AtomType.int][int (random.random () * numPossibleTypes)]
					if rhsType == atom.AtomType.real :
						rhs = rangeRandReal ()

					elif rhsType == atom.AtomType.int :
						rhs = rangeRandInt ()

					# push the rhs of the assignment onto the stack to prepare for the assign instructions
					# this value will get popped off the stack by the assign instruction
					preFragment.append (Instruction (Instruction.pushConst,
													argument=rhs,
													argumentDelta=rhs*0))

					# set the argument to the index of the symbolTable to which the value will be assigned
					lhs = int (random.random () * sizeOfSymTable)
					argument = lhs
					argumentDelta=0

			elif Instruction.specs[opCode]['arity'] == InstructionOperandType.binary :
				preFragment = []

			else :
				print "synthesize: invalid InstructionOperandType"

			instr = Instruction (opCode,
								argument=argument,
								argumentDelta=argumentDelta)

			self.instructions.extend (preFragment)
			self.instructions.append (instr)

		if len (self.instructions) - self.numSetupLines > self.numLines :
			self.instructions = self.instructions[:self.numLines + self.numSetupLines]

		self.genotype = self.instructionsToFlatBinary ()

	def driftArguments (self) :
		for line in self.instructions :
			line.driftArgument()

	def __repr__ (self) :
		result = ''
		for line,instr in enumerate (self.instructions) :

			result += ("%04i" % line) + str (instr)

		return result

	def popOperValue (self, operandNumber, instruction) :
		#take the operand type and pop off the stack

# 		global overflowMessagesOn

		if operandNumber == 0 :
			oper0Type = Instruction ().specs[instruction.opCode]['oper0Type']
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
				if self.overflowMessagesOn :
					print "Program: popOperValue: instruction.operand0Type is unknown atom.AtomType " + str(oper0Type)

		elif operandNumber == 1 :
			oper1Type = Instruction ().specs[instruction.opCode]['oper1Type']

			if oper1Type == atom.AtomType.real :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (float (pop[0]), oper1Type)

			elif oper1Type == atom.AtomType.int :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (int (pop[0]), oper1Type)

			elif oper1Type == atom.AtomType.bool :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (bool (pop[0]), oper1Type)

			elif oper1Type == atom.AtomType.any :
				if self.machineStack.length () > 0 :
					pop = self.machineStack.pop ()
					return (pop[0], atom.AtomType ().getValueType (pop[0]))

			else :
				if self.overflowMessagesOn :
					print "Program: popOperValue: instruction.operand1Type is invalid atom.AtomType " + str (oper1Type)
				return (None, None)

		else :
			if self.overflowMessagesOn :
				print "Program: popOperValue: invalid operand number " + str (operandNumber)
			return (None, None)


	def makeValueTypeTuple (self, value) :
			return (value,atom.AtomType ().getValueType (value))


	def execute (self, inputs=None) :
		#overflowMessagesOn = p.parameters["overflowMessagesOn"]
		self.symbolTable = [(0,0.0) for _ in xrange (1000)]
		self.symbolTopIndex = 0
		self.inputs = []

		self.machineStack = Stack (10000)

		self.operand0 = 0.0 # set from operand1 type stack pop
		self.operand1 = 0.0 # set from operand1 type stack pop

		self.returnValue = 0.0 # set from result of operator

		if inputs is not None :
			self.inputs = inputs

		#global overflowMessagesOn

		for lineNum,instr in enumerate (self.instructions) :
			r.setRandomSeed(self.randomSeed)

# 			stackTopType,stackTopValue = self.machineStack.top ()

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
				if self.overflowMessagesOn :
					print ("Program: execute: instr: " +
							str (lineNum) +
							" " +
							Instruction.strings[instr.opCode] +
							" instruction found operand0 set to None. Nothing for pushing.")
				continue

			if (Instruction.specs[instr.opCode]['oper1Type'] is not None and
			 	self.operand1 is None) :
				if self.overflowMessagesOn :
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

			printOperand = False
			if printOperand :
				if self.operand0 is not None :
					print "\t(" + str (self.operand0[0]) + " isA " + atom.AtomType.strings[self.operand0[1]] + ")"

				else :
					print "\t(None isA None)"

				if self.operand1 is not None :
					print "\t(" + str (self.operand1[0]) + " isA " + atom.AtomType.strings[self.operand1[1]] + ")"

				else :
					print "\t(None isA None)"

			if instr.opCode == Instruction.assign :   ######## ASSIGN (BINARY) ######
				if self.symbolTopIndex > 0 :
					destIndex = instr.argument

				else :
					destIndex = 0
					self.symbolTopIndex = 0

				self.symbolTable[destIndex] = (self.operand0[0],self.operand0[1])

				self.machineStack.push (self.operand0[0])

			elif instr.opCode == Instruction.pushFromSym :     ####### PUSH SYMBOL  (NULLARY) ######

				# push contents of symbol table at index argument or operand0 to stack
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

					elif symbolType == atom.AtomType.nil :
						self.machineStack.push (float(0.0))	

					elif symbolType == atom.AtomType.any :
						self.machineStack.push (float (0.0))

					else :
						if self.overflowMessagesOn :
							print "Program: execute: pushFromSym: pushFromSym symbolType = " + str (symbolType) 

			elif instr.opCode == Instruction.pushFromInput :     ####### PUSH FROM INPUTS LIST  (UNARY) ######
				if instr.argument is None :
					if self.overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushFromInput instruction found operand0 set to None. Unknown what index to push."
					continue

				if self.inputs is None :
					if self.overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushFromInput instruction found inputs set to None. Nothing for pushing."
					continue

				if len (self.inputs) == 0 :
					if self.overflowMessagesOn :
						print "Program: execute: instr: " + str (lineNum) + " pushFromInput instruction found zero-length inputs. Nothing for pushing."
					continue

				inputValue = self.inputs[int (instr.argument) % len (self.inputs)]
				self.machineStack.push (float (inputValue))

			elif instr.opCode == Instruction.popToOperand  :     ####### POP to OPERAND ######
				# pop contents of specified stack to self.operand0 or operand1
				if instr.argument % 2 == 0 :
					self.operand0 = self.machineStack.pop ()

				elif instr.argument % 2 == 1 :
					self.operand1 = self.machineStack.pop ()

				else :
					print "Program: execute: popToOperand: argument should be 0 or 1 (operand number) Instead got " + str (instr.argument)

			elif instr.opCode == Instruction.pushConst :     ####### PUSH CONST  (UNARY) ######

				(constValue, constType) = self.makeValueTypeTuple (instr.argument)

				if constType == atom.AtomType.real :
					self.machineStack.push (float (constValue))

				elif constType == atom.AtomType.int :
					self.machineStack.push (int (constValue))

				elif constType == atom.AtomType.bool :
					self.machineStack.push (bool (constValue))

				elif constType == atom.AtomType.nil :
					self.machineStack.push (float(0.0))	

				elif constType == atom.AtomType.any :
					self.machineStack.push (float (0.0))

				else :
					if self.overflowMessagesOn :
						print "Program: execute: pushConst constType = " + str (constType) 

			elif instr.opCode == Instruction.pushAllSym :     ####### PUSH ALL SYMBOLS  (NULLARY) ######
				index = 0
				for (symbolValue, symbolType) in self.symbolTable :
					if index >= self.symbolTopIndex :
						break

					if symbolType == atom.AtomType.real :
						self.machineStack.push (float (symbolValue))

					elif symbolType == atom.AtomType.int :
						self.machineStack.push (int (symbolValue))

					elif symbolType == atom.AtomType.bool :
						self.machineStack.push (bool (symbolValue))

					elif symbolType == atom.AtomType.nil :
						self.machineStack.push (float(0.0))	

					elif symbolType == atom.AtomType.any :
						self.machineStack.push (float (0.0))

					else :
						if self.overflowMessagesOn :
							print "Program: execute: pushAllSym: symbolType = " + str (symbolType) 


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
				try :
					self.returnValue = self.operand0[0] * self.operand1[0]
				except :
					self.returnValue = 0.0

				self.machineStack.push (self.returnValue)


			elif instr.opCode == Instruction.div  :	####### DIV ######
				if self.operand1[0] != 0.0 :
					try :
						self.returnValue = self.operand0[0] / self.operand1[0]
					except :
						self.returnValue = 0.0

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
					try :
						self.operand0 = ((self.operand0[0]/float(self.operand0[0]+self.operand1[0])) * maxAllowed, self.operand0[1])
					except :
						self.operand0 = (0.0,self.operand0[1])

					try :
						self.operand1 = ((self.operand1[0]/float(self.operand0[0]+self.operand1[0])) * maxAllowed, self.operand1[1])
					except :
						self.operand1 = (0.0,self.operand1[1])

				if abs (self.operand0[0]) > 200 :
					try :
						sign = self.operand0[0] / abs (self.operand0[0])
					except :
						sign = +1

					self.operand0 = (200.0 * sign, self.operand0[1])

				if abs (self.operand1[0]) > 100 :
					try :
						sign = self.operand1[0] / abs (self.operand1[0])
					except :
						sign = +1

					self.operand1 = (100.0 * sign, self.operand1[1])

				if self.operand0[0] < 0.0 :
					self.operand0 = (abs (self.operand0[0]), self.operand0[1])

				if (self.operand0[0] == 0.0 and
					self.operand1[0] < 0.0) :
					self.returnValue = 0.0

				else :
					if (self.operand0[0] < 1e-6 and
						self.operand1[0] < 0) :
						self.returnValue = 0.0

					else :
						try :
							self.returnValue = self.operand0[0]**self.operand1[0]
						except :
							self.returnValue = 0.0

				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.sin :	####### SIN ######
				try :
					self.returnValue = math.sin (self.operand0[0])
				except :
					self.returnValue = 0.0

				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.cos :	####### COS ######
				try :
					self.returnValue = math.cos (self.operand0[0])
				except :
					self.returnValue = 0.0
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.log10 :	####### LOG10 ######
				try :
					self.returnValue = math.log10(self.operand0[0])
				except :
					self.returnValue = 0.0
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.mod :	####### MOD ######
				try :
					self.returnValue = self.operand0[0] % self.operand1[0]
				except :
					self.returnValue = 0.0
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.abs :	####### ABS ######
				try :
					self.returnValue = abs (self.operand0[0])
				except :
					self.returnValue = 0.0
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.min :	####### MIN ######
				try :
					self.returnValue = min (self.operand0[0], self.operand0[0])
				except :
					self.returnValue = 0.0
				self.machineStack.push (self.returnValue)

			elif instr.opCode == Instruction.max :	####### MAX ######
				try :
					self.returnValue = max (self.operand0[0], self.operand1[0])
				except :
					self.returnValue = 0.0
				self.machineStack.push (self.returnValue)

			else :
				if self.overflowMessagesOn :
					raise Exception, "Program: execute: instr: " + str (lineNum) + " CRASH: execute2: instruction " + str (instr.opCode) + " not defined."

		if self.machineStack.length () > 0 :
			return self.machineStack.pop ()

		else :
			return None


	def instructionsToFlatBinary (self) :

		flatBinary = ''
		for thisInstruction in self.instructions :
			string = thisInstruction.toBinary ()
			flatBinary += string

		return flatBinary


	def flatBinaryToInstructions (self, binaryIn) :

		self.instructions = []

		numInstructions = len (binaryIn) / Instruction ().instructionSize

		if (len (binaryIn) / float (Instruction ().instructionSize)) != (len (binaryIn) / Instruction ().instructionSize) :
			raise Exception, "flatBinaryToInstructions: binaryIn size does not divide evenly by size of instruction"

		startBit = 0
		endBit = 0
		for _ in range (numInstructions) :
			startBit = endBit
			endBit = startBit + Instruction ().instructionSize
			binary = binaryIn[startBit:endBit]
			instr = Instruction ()
			instr.fromBinary (binary)
			self.instructions.append (instr)

	def test (self) :
		self.instructions = []

		expression = True
		if expression :
			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(3.0),
													argumentDelta=0.0)) #(3.0)

			self.instructions.append (Instruction ( Instruction.pushFromInput,
													argument=0,
													argumentDelta=0)) # push input 0 to stack (x)

			self.instructions.append (Instruction ( Instruction.pow)) # (x**3.0)

			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(0.4),
														argumentDelta=0.0)) #(0.4)

			self.instructions.append (Instruction ( Instruction.mult)) # (0.4 *   (x ** 3.0))


			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(0.2),
														argumentDelta=0.0)) #(0.2)

			self.instructions.append (Instruction ( Instruction.pushFromInput,
													argument=0,
														argumentDelta=0)) # push input 0 to stack (x)

			self.instructions.append (Instruction ( Instruction.mult)) # (0.2*x)

			self.instructions.append (Instruction ( Instruction.add)) # ( (.2 * x) +     0.4 *   (x ** 3.0))

			self.instructions.append (Instruction ( Instruction.pushConst,
													argument=float(0.5),
														argumentDelta=0.0)) #(0.5)

			self.instructions.append (Instruction ( Instruction.add)) # (.5 +   (.2 * x) +     0.4 *   (x ** 3.0))

			self.genotype = self.instructionsToFlatBinary ()

		else :
			self.synthesize ()


		self.flatBinaryToInstructions(self.genotype)

		sampleRange = [0,10]
		samples = []

		for test in range (sampleRange[0], sampleRange[1]) :
			self.machineStack = []

			test2 = test * .1

			for i in self.instructions :
				print i

			result = self.execute ([test2])

			if result is None :
				result = 0

			else :
				result = result[0]

			samples.append (result + .01)

		def plotFunction () :
			func = p.parameters["targetFunction"]
			fig = figure (1)
			ax1 = fig.add_subplot (1,1,1)
			x = [_ * .1 for _ in range (sampleRange[0], sampleRange[1])]
			hf = [func (_) for _ in x]
			ax1.plot (x,hf)
			ax1.plot (x,samples)
			ax1.grid (True)
			yBoxMax = max (max (samples), max(hf)) * 1.1
			yBoxMin = min (min (samples), min(hf)) - 0.1
			ax1.set_ylim ((yBoxMin,  yBoxMax))
			ax1.set_xlim ((-.1,  max (x) * 1.1))

			show ()

		plotFunction ()
#
# prog = Program(0, p.parameters["numInstructions"])
# prog.test ()

