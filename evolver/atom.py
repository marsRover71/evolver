
import random
import math
import types
import sys

from pylab import figure, show

sys.path.append ('/Users/rhett/workspace/evolver')


overflowMessagesOn = False

class BinaryString :

	extraBitForNilValue = 1
	inclusive = 1
	base2 = 2

	# precision must be large enough to account for extra "Nil" value which is always binary all ones
	# binary all-ones means Nil
	def __init__ (self, precision=None, rangeIn=None, binaryStrIn=None) :

		if (precision is None and
		   rangeIn is None and
		   binaryStrIn is None) :

			return

		else :
			if rangeIn is None :
				self.range = None

			else :
				if rangeIn[1] >= rangeIn[0] :
					self.range = rangeIn

				else :
					raise Exception, "BinaryString: Error: range passed in with out of order low and high values."

			if binaryStrIn is not None :
				self.binaryStr = binaryStrIn
				precision = len (binaryStrIn)

			if precision is not None :
				self.precision = precision

				if binaryStrIn is None :
					self.binaryStr = '0' * self.precision

			else :
				raise Exception, "BinaryString: Error: instance should have precision supplied for real value."

			self.rangeSize = self.range[1] - self.range[0]# + 1
			self.numValues = self.rangeSize + self.extraBitForNilValue

	def setToNil (self) :
		self.binaryStr = '1' * self.precision

	def toggleBit (self, bitPlace) :
		newBit = str (int (not bool (int (self.binaryStr[bitPlace]))))
		self.binaryStr = self.binaryStr[:bitPlace] + newBit + self.binaryStr[(bitPlace+1):]

	def rangeToPrecision (self, rangeIn) : # range is inclusive
		return int (math.ceil (math.log (rangeIn[1] - rangeIn[0] + 1, self.base2)))

	def flipRandomBit (self) :
		index = int (random.random () * (len (self.binaryStr) - 1))
		newBit = str (int (not bool (int (self.binaryStr[index]))))
		return self.binaryStr[:index] + newBit + self.binaryStr[(index + 1):]

	def toInt (self) :
		# binary all-ones means None
		#
		# bin --+-> int
		#       |
		#       +-> None
		#
		assert (type (self.binaryStr) == types.StringType or
				self.binaryStr == None), ("toInt: binaryIn should be an string or None")

		assert len (self.binaryStr) == self.precision, ("toInt: precision of binaryIn " +
												str (len (self.binaryStr)) + 
												" should equal self.precision " + 
												str (self.precision))

		if self.binaryStr == '1' * self.precision :
			return None

		else :

			rangeSize = self.range[1] - self.range[0] + 1

			decInt = int (self.binaryStr,2) % (rangeSize)
			value = decInt + self.range[0]
			assert (value is None or 
				   (value >= self.range[0] and value <= self.range[1])), ("toInt: value " +
																				str (value) +
																				" outside of range " +
																				str (self.range))
			return value

	def setFromInt (self, valueIn) :
		# binary all-ones means None
		#
		# int ---+-> bin
		#        |
		# None --+
		#

		if valueIn is None :
			self.setToNil()
			return

		assert (type (valueIn) == types.IntType or
				valueIn == None), ("setFromInt: valueIn should be an int or None")

		valueMod = (valueIn - self.range[0]) % self.rangeSize

		binaryStr = format(valueMod, '#0' + str (self.precision) + 'b')[2:]

		# prepend leading zeros
		binaryStr = '0' * (self.precision - len (binaryStr)) + binaryStr
		self.binaryStr = binaryStr

	def setFromReal (self, valueIn) :
		# binary all-ones means None
		#
		# real --+-> int -> bin
		#        |
		# None --+
		if valueIn is None :
			self.setToNil()
			return

		assert (type (valueIn) == types.FloatType or
				valueIn == None), ("setFromReal: value should be a float or None")

		assert (valueIn >= self.range[0] and valueIn <= self.range[1]), ("setFromReal: value " +
																			str (valueIn) +
																			" outside of range " +
																			str (self.range))


		if self.rangeSize > 0 :
			valueMod = (valueIn - self.range[0]) % self.rangeSize + self.range[0]
			normValue = (valueMod-self.range[0]) / self.rangeSize

		else :
			valueMod = 0
			normValue = 0


		self.numValues = (2**self.precision)-1
		interInt = int (normValue * self.numValues)
		binaryStr = format (interInt, '#0' + str (self.precision) + 'b')[2:]
		# prepend leading zeros
		binaryStr = '0' * (self.precision - len (binaryStr)) + binaryStr
		self.binaryStr = binaryStr

	def toReal (self) :
		# binary all-ones means None
		#
		# bin --+-> int -> real
		#       |
		#       +-> None
		#
		assert (type (self.binaryStr) == types.StringType or
				self.binaryStr == None), ("toReal: self.binaryStr should be a string or None")

		assert len (self.binaryStr) == self.precision, ("toReal: precision of self.binaryStr " +
												str (len (self.binaryStr)) + 
												" should equal self.precision " + 
												str (self.precision))

		if self.binaryStr == '1' * self.precision :
			return None

		else :
			self.numValues = (2**self.precision)-1

			decInt = int (self.binaryStr,self.base2)
			value = (decInt / float (self.numValues)) * self.rangeSize + self.range[0]

			assert value >= self.range[0] and value <= self.range[1], ("toReal: value " +
																		str (value) +
																		" outside of range " +
																		str (self.range))

			return value


	def setFromBool (self, valueIn) :
		# binary all-ones means None
		#
		# bool ---+-> bin
		#        |
		# None --+
		#
		assert (type (valueIn) == types.BooleanType or
				valueIn == None), ("fromBool: valueIn should be a bool or None")

		assert (valueIn is None or 
				(valueIn >= self.range[0] and valueIn <= self.range[1])), ("fromBool: value " +
																			str (valueIn) +
																			" outside of range " +
																			str (self.range))

		if valueIn is None :
			binaryStr = '11'

		else :
			binaryStr = '0' + str (int (valueIn))

		self.binaryStr = binaryStr

	def toBool (self) :
		# binary all-ones means None
		#
		# bin --+-> bool
		#       |
		#       +-> None
		#
		assert (type (self.binaryStr) == types.StringType or
				self.binaryStr == None), ("toBool: self.binaryStr should be an string or None")

		assert len (self.binaryStr) == self.precision, ("toBool: precision of self.binaryStr " +
												str (len (self.binaryStr)) + 
												" should equal self.precision " + 
												str (self.precision))

		if self.binaryStr == '11' :
			return None

		else :
			value = bool (self.binaryStr[-1])
			assert (value is None or 
				   (value >= self.range[0] and value <= self.range[1])), ("toBool: value " +
																		str (value) +
																		" outside of range " +
																		str (self.range))
			return value


class AtomType :
	real = 0
	int = 1
	bool = 2
	nil = 3
	any = 4


	strings = [ "real",
				"int",
				"bool",
				"nil",
				"any"]

	precision = BinaryString().rangeToPrecision ([0,len (strings)])

	def toStr (self, atomIn) :
		assert atomIn < len (self.strings), "toStr: atomIn not found in AtomTypes.strings."
		return self.strings[atomIn]

	def toAtomType (self, stringIn) :
		assert stringIn in self.strings, "toAtomType: stringIn not found in AtomType.strings."
		return self.strings.index (stringIn)

	def toBinary (self, typeIn) :
		binary = BinaryString (precision=self.precision, rangeIn = [0,len (self.strings)])
		binary.setFromInt (typeIn)
		return binary.toInt ()

	def getValueType (self, value) :
		if type (value) == types.IntType :
			return AtomType.int

		elif type (value) == types.FloatType :
			return AtomType.real

		elif type (value) == types.BooleanType :
			return AtomType.bool

		elif type (value) == types.NoneType :
			return AtomType.nil

		else :
			return None

	def coerce (self, value, typeIn) :
		if typeIn == AtomType.int :
			return int (typeIn)

		elif typeIn == AtomType.real :
			return float (value)

		elif typeIn == AtomType.bool :
			return bool (value)

		elif typeIn == AtomType.nil :
			return None

		else :
			return None


atomTypeBinaryList = [AtomType ().toBinary (_) for _ in range (len (AtomType.strings))]

class Atom :
	def __init__ (self, typeIn, rangeIn, valueIn, precisionIn) :
		self.type = typeIn
		self.range = rangeIn
		self.value = valueIn
		self.precision = precisionIn


if False :
	b = BinaryString(32,rangeIn=[-100.0,100.0])
	print b.binaryStr
	print b.toReal ()

	a = Atom (typeIn=AtomType.int, rangeIn=[-10,10], valueIn=0, precisionIn=32)
	print a.value

	operTypeRange = [-7,7]
	operTypePrec =BinaryString().rangeToPrecision (operTypeRange)
	print math.log(8,2)
	argumentType = BinaryString (precision=operTypePrec, rangeIn=operTypeRange)
	argumentType.setFromInt (7)
	print bin(7+7)
	argumentTypeBinary = argumentType.binaryStr
	print argumentTypeBinary


