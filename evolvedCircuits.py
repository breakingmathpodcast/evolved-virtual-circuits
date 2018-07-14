# Evolves a graph.
# Copyright 2017 Gabriel Hesch and Jonathan Baca

import itertools
import signal
import ahkab
import math
import random
import pickle
import base64
from threading import Thread, Lock
import time
import queue
from enum import Enum
from multiprocessing import Process, Manager
import copy

nThreads = 5
callTimeLimit = 1

nSignificand = 53 # Precision of floating point numbers
nExponent    = 11 # Range of floating point numbers

chanceOfToggleActivation = 1/50 # Ask an expert
chanceOfRealSignFlip = 1/40 # Ask an expert
realMutationSigma    = .1   # Ask an expert
numSeedAddNodes = 3

generationNames = [['Adam','Eve'], ['Cain', 'Abel', 'Seth'], ['Enoch', 'Enos'], ['Irad','Cainan'],\
                   ['Mehujael', 'Mahalaleel'], ['Methushael', 'Jared'], ['Adah', 'Lamech', 'Zillah', 'Enoch'], ['Jabal', 'Jubal', 'Tubal', 'Cain', 'Naamah', 'Methuselah'], ['Lamech'], ['Noah'], ['Shem', 'Ham', 'Japheth'], ['Gomer', 'Magog', 'Madai', 'Javan', 'Tubal', 'Meshech', 'Tiras', 'Cush', 'Mizraim', 'Phut', 'Canaan', 'Elam', 'Ashur', 'Arpachshad', 'Lud', 'Aram' ], ['Ashkenaz', 'Riphath', 'Togarmah', 'Elishah', 'Tarshish', 'Kittim', 'Dodanim', 'Seba', 'Havilah', 'Sabtah', 'Raamah', 'Sabtechah', 'Nimrod', 'Cainan', 'Shelah', 'Uz', 'Hul', 'Gether', 'Mash'], ['Sheba', 'Dedan', 'Eber'], ['Ludim', 'Anamim', 'Lehabim', 'Naphtuhim', 'Pathrusim', 'Casluhim', 'Caphtorim', 'Peleg', 'Joktan'], ['Sidon', 'Heth', 'Jebusites', 'Amorites', 'Girgashites', 'Hivites', 'Arkites', 'Sinites', 'Arvadites', 'Zemarites', 'Hamathites', 'Reu'], ['Almodad'], ['Sheleph', 'Hazarmaveth', 'Jerah', 'Hadoram', 'Uzal', 'Diklah', 'Obal', 'Abimael', 'Sheba', 'Ophir', 'Havilah', 'Jobab' ], ['Serug', 'Nahor', 'Terah'], ['Abraham', 'Sarah', 'Nahor', 'Haran', 'Hagar', 'Nahor', 'Haran' ], ['Milcah', 'Lot', 'Iscah', 'Ishmael' ], ['Bethuel', 'Isaac', 'Rebecca', 'Laban', 'Moabites', 'Ammonites' ], ['Jacob', 'Nebaioth', 'Kedar', 'Adbeel', 'Mibsam', 'Mishma', 'Dumah', 'Massa', 'Hadar', 'Tema', 'Jetur', 'Naphish', 'Kedemah']]

# Replaces "ab <c> d<<e>> <f><<f>>gh", {'c': 'x', 'f': 23} with
# "ab x d<e> 23<f>gh"
# BUGGY! FIX BUGS!
def replaceTagged(string, d):
  for k,x in d.items():
    kesc = re.escape(k)
    string = re.sub('((?!<).<'+kesc+'>(?!>))|^<'+kesc+'>', \
                    re.escape(str(x)), string)
    string = re.sub('<'+kesc+'>', kesc, string)
  return string

# All this is is a linked list node that points to another
# node, and a piece of data
class StackNode():
  def __init__(self, data=None, next=None):
    self.data = data
    self.next = next

# [1, 5, 3]      .push(7)
# [7, 1, 5, 3]   .pop() --> 7
# [1, 5, 3]
# Also iterable, so we go through it
class Stack():
  def __init__(self):
    self.head = StackNode()
    self.n = 0
  def peek(self):
    return self.head.data
  def pop(self):
    if self.n <= 0:
      return
    self.n -= 1
    x = self.head.data
    self.head = self.head.next
    return x
  def push(self, x):
    self.n += 1
    newHead = StackNode(x, self.head)
    self.head = newHead
  def __iter__(self):
    self.cursor = self.head
    return self
  def __next__(self):
    if self.cursor.next:
      value = self.cursor.data
      self.cursor = self.cursor.next
      return value
    else:
      raise StopIteration()
  def __len__(self):
    return self.n

# It gives the client a job to do.
class ElectricJob():
  def __init__(self, **kwargs):
    self.__docket__ = Stack()
    self.infoID = kwargs['infoID']
    self.returnInfo = kwargs['returnInfo']
    if 'electricWorld' in kwargs:
      self.electricWorld = kwargs['electricWorld']
    if 'mother' in kwargs:
      self.mother = kwargs['mother']
      self.father = kwargs['father']
      self.__docket__.push('breed')
    if 'code' in kwargs:
      self.code = kwargs['code']
      if 'vars' in kwargs:
        self.vars = kwargs['vars']
      else:
        self.vars = None
      if 'globals' in kwargs:
        self.globals = kwargs['globals']
      else:
        self.globals = None
      if 'locals' in kwargs:
        self.locals = kwargs['locals']
      else:
        self.locals = None
      self.__docket__.push('exec')
  def __call__(self):
    message = Message(infoID=self.infoID, returnInfo=self.returnInfo)
    for job in self.__docket__:
      if job == 'breed':
        message += self.breed()
      if job == 'exec':
        message += self.exec()
  def breed(self):
    mother = loadThing64(self.mother)
    father = loadThing64(self.father)
    while True:
      offspring = mother % father
      ~offspring
      fitness = offspring()
      if fitness != 0:
        break
    return Message(offspring=offspring, fitness=fitness)
  def exec(self):
    code = loadThing64(self.code)
    if self.vars is not None:
      code = replaceTagged(code, self.vars)
    else:
      code = code
    eval(code, self.globals, self.locals)

# Transforms data into hex (BUGGY)
def dataToHex(data, encoding=None):
  if encoding is not None:
    data = bytearray(data, encoding)
    encoding = bytearray(encoding, 'utf-8')+b':'
  else:
    if isinstance(data, str):
      data = bytearray(data, 'utf-8')
      encoding = bytearray('utf-8:','utf-8')
    else:
      encoding = b''
  return encoding+b''.join([bytearray(hex(c),'utf-8')[2:] for c in data])

# Transforms hex into data (BUGGY)
def hexToData(code):
  code = code.split(b':')
  if len(code) == 2:
    encoding = code[0]
    code = code[1]
  else:
    code = code[0]
    encoding = None
  code = bytearray([(int(chr(c[0]),base=16)<<4)+int(chr(c[1]),base=16) for c in chunkIntoTuples(code,2)])
  if encoding:
    return code.decode(str(encoding,'utf-8'))
  else:
    return code

# A fake file. Takes and gives data like a real file without bugging the file system
class Pseudofile():
  def __init__(self):
    self.data = b''
  def write(self, data):
    self.data += data
  def read(self, *args):
    return self.data
  def readline(self, *args):
    return self.data.split(b'\n')[0]

# Stores an object as code
def saveThing(thing):
  ps = Pseudofile()
  pickle.dump(thing, ps)
  return ps.data

# Saves a pickle in base-64
def saveThing64(thing):
  return base64.b64encode(saveThing(thing))

# Loads an encoded object.
def loadThing(code):
  ps = Pseudofile()
  ps.data = code
  return pickle.load(ps)

# Loads a base-64 encoded pickle string
def loadThing64(thing):
  return loadThing(base64.b64decode(thing))

# 'abc', 10, 'x' --> 'abcxxxxxxx'
def padTo(string, width, using=' '):
  numLeft = width - len(string)
  if numLeft <= 0:
    return string
  padding = (using * math.ceil(numLeft / len(using)))[0:numLeft]
  return string + padding

# Makes things into tables
def tablify(xss, nSpaces = 1, **kwargs):
  pss = [[str(x) for x in xs] for xs in xss]
  lens = [max([len(p) for p in ps]) for ps in pss]
  maxlen = max([len(ps) for ps in pss])
  if 'stripesEvery' in kwargs:
    padchars = ' '*(kwargs['stripesEvery'] - 1) + kwargs['stripeChar'][0]
  else:
    padchars = ' '
  padlen = len(padchars)
  r = ''
  for i in range(0, maxlen):
    padchar = padchars[i % padlen]
    for j in range(0, len(pss)):
      if i < len(pss[j]):
        r += padTo(pss[j][i], lens[j] + (0 if j == len(pss)-1 else nSpaces), padchar)
      else:
        r += padTo('', lens[j] + (0 if j == len(pss)-1 else nSpaces), padchar)
    r += '\n'
  return r

# def packStrings(multiStrings, width=80, spacing=1):
#   multiStrings = [multiString.split('\n') for multiString in multiStrings]
#   widths = [max([len(string) for string in multiString]) for multiString in multiStrings]
#   r = ""
#   i = 0
#   while i < len(multiStrings):
#     count = 0
#     j = i
#     while count < width and j < len(widths):
#       count += widths[j] + spacing
#       j += 1
#     if j < len(widths):
#       count -= widths[j] + spacing
#     for j in range(0, max([len(multiString) for multiString in multiStrings[i:j]])):
#       for k in range(i, j):
#         print(k)
#         if len(multiStrings[k]) < j:
#           r += padTo(multiStrings[k][j], widths[k] + spacing)
#         else:
#           r += padTo('', widths[k] + spacing)
#     r += '\n'
#     i = j
#   return r

# Run something in the background
def backgroundRun(f, *args, **kwargs):
  def threadFunc():
    f(*args,**kwargs)
  t = Thread(target=threadFunc, args=args, kwargs=kwargs)
  t.start()

# Run f(*args,**kwargs) for a MAXIMUM of t seconds
def runForTime(t, f, *args, **kwargs):
  ret = Manager().dict()
  def sleepFunc():
    time.sleep(t)
  def wrapper(sleepProcess, ret, *args, **kwargs):
    ret['value'] = f(*args, **kwargs)
    sleepProcess.terminate()
  sleepProcess = Process(target=sleepFunc)
  sleepProcess.start()
  mainProcess = Process(target=wrapper, args=(sleepProcess,ret)+args, kwargs=kwargs)
  mainProcess.start()
  sleepProcess.join()
  mainProcess.terminate()
  if 'value' in ret:
    return ret['value']

# [d_0,d_1,...,d_n] -> [0:d_0,1:d_1,...,n:d_n]
def xsToDict(xs):
  d = {}
  for i in range(0,len(xs)):
    d[i] = xs[i]
  return d

# Prints with a timestamp
def printts(*args, **kwargs):
  ts = time.ctime()
  ts0 = "(%s) >>>> " % ts
  tsr = "(%s)   >> " % ts
  s = []
  for arg in args:
    s += [str(arg)]
  s = ' '.join(s).split('\n')
  s = '\n'.join([(ts0 if i == 0 else tsr) + seg for i,seg in xsToDict(s).items()])
  print(s)
  if kwargs:
    print(s+'\n', **kwargs)

# Returns the location of the maximum in xs
def maxLoc(xs):
  m = xs[0]
  j = 0
  for i in range(1,len(xs)):
    if m < xs[i]:
      m = xs[i]
      j = i
  return j

# gets n unique random integers in [a,b]
def getRandInts(a, b, n):
  if b-a < n-1:
    return
  x = [random.randint(a,b-i) for i in range(0,n)]
  x.sort()
  j = 0
  for i in range(1,n):
    if x[i-1] - j == x[i]:
      j += 1
    x[i] += j
  random.shuffle(x)
  return x

# turns [x1,...,xm] into [(x1,...,xn),...,(xm-n+1,...,xm)]
def chunkIntoTuples(xs, n=2):
  r = [None] * math.ceil(len(xs) / n)
  j = 0
  for i in range(0, len(r)*n, n):
    r[j] = tuple(xs[i:i+n])
    j += 1
  return r

# Takes the nth derivative of a signal
def deriv(signal, n=1):
  if n > 1:
    return deriv(deriv(signal, n - 1))
  r = [None] * (len(signal) - 1)
  for i in range(0, len(signal)-1):
    r[i] = signal[i+1] - signal[i]
  return r

# Strikes out text
def strikeout(string):
  i = 0
  newString = ''
  for i in range(0,len(string)):
    newString += string[i] + u'\u0336'
  return newString

# Returns true with probability p
def randBool(p):
  return random.random() < p

# Mutate a real number
# def mutateReal(x, canBeNegative):
#   if canBeNegative:
#     getBin = floatToGray
#     getFloat = grayToFloat
#   else:
#     getBin = absFloatToGray
#     getFloat = grayToAbsFloat
#   x = getBin(x)
#   x ^= 1 << random.randint(0, nSignificand + nExponent - 1)
#   return getFloat(x)
def mutateReal(x, canBeNegative):
  if canBeNegative and randBool(chanceOfRealSignFlip):
    x = -x
  return x * random.normalvariate(1, realMutationSigma)

# Breed two real numbers
# PX: Harmonic average
def breedReals(a, b):
  return (a + b) / 2

# Our error handling classes
## An error in breeding
class BreedingError(Exception):
  def __init__(self, message):
    super(BreedingError, self).__init__(message)

# An error in initializing
class ElectricInitError(Exception):
  def __init__(self, message):
    super(BreedingError, self).__init__(message)

# Makes a random value from spread
def getRandomValue(multiplier):
  return multiplier * random.lognormvariate(0, .25)

# Makes a scientific number from a number
def scistring(n):
  prefixes = ['f','p','n','μ','m','','k','M','G','T','P']
  index = 5
  while abs(n) > 1000 and index < 10:
    n *= .001
    index += 1
  while abs(n) < 1 and index > 0:
    n *= 1000
    index -= 1
  return str(round(n,6))+prefixes[index]

# Makes a dictionary from an enum
def dictFromEnumValues(enumClass, initFunc):
  keys = list(enumClass)
  d = {}
  for key in keys:
    d[key.value] = initFunc()
  return d

# Search for needle in sortedIterable
def binSearchIdx(needle, sortedIterable):
  a = 0
  b = len(sortedIterable)
  while a < b:
    c = (a + b) // 2
    if sortedIterable[c] < needle:
      a = c + 1
    elif sortedIterable[c] > needle:
      b = c
    else:
      a = c
      b = 0_getand
  return c

# Types of electrical components
class ElectricTypes(Enum):
  RESISTOR = 1
  CAPACITOR = 2
  INDUCTOR = 3
  MOSFET = 4
  VSOURCE = 5
  ISOURCE = 6

# List of ElectricEdge types
electricEdgeTypes = [ElectricTypes.RESISTOR,  \
                     ElectricTypes.CAPACITOR, \
                     ElectricTypes.INDUCTOR,  \
                     ElectricTypes.VSOURCE,   \
                     ElectricTypes.ISOURCE]

# List of ElectricTriode types
electricTriodeTypes = [ElectricTypes.MOSFET]

# Counts
class Counter():
  def __init__(self, n=0):
    self.n = n - 1
    self.resetN = n - 1
  def __neg__(self):
    self.n = self.resetN
    return self
  def __call__(self):
    self.n += 1
    return self.n

# Helper function for getAndStoreUnique
def _getAndStoreUnique(n, m, js, d, counter):
  if n == m:
    if js[n] not in d:
      d[js[n]] = counter()
    return d[js[n]]
  if js[n] not in d:
    d[js[n]] = {}
  return _getAndStoreUnique(n+1, m, js, d[js[n]], counter)

# Stores a Counter() value in d[js[0]][js[1]]...[js[n]], and
#  create one if none exists
def getAndStoreUnique(js, d, counter=Counter()):
  return _getAndStoreUnique(0, len(js)-1, js, d, counter)

# Represents the state of the world
class ElectricWorldState():
  def __init__(self, **kwargs):
    for a, b in kwargs.items():
      object.__setattr__(self, a, b)
    if not hasattr(self, 'nGenerations'):
      self.nGenerations = 0
    self.attrs = [a for a,b in kwargs.items()]
    self.record = {}
    self.recordNumbers = []
  def getAttrs(self):
    r = {}
    for attr in self.attrs:
      r[attr] = getattr(self, attr)
    return r
  def __iadd__(self, other):
    self.nGenerations += other
    return self
  def __pos__(self):
    self.record[self.nGenerations] = self.getAttrs()
    self.recordNumbers += [self.nGenerations]
  def __getitem__(self, n):
    return self.record[binSearchIdx(n, self.recordNumbers)]

# Represents the world in which ElectricCircuit creatures interact
class ElectricWorld():
  def __init__(self, nPopulation=1000, t=1, tstep=1e-3): # NEW: nP..n
    self.counter = Counter()
    self.population = [None] * nPopulation
    self.maxFitness = -1
    self.maxFitnessLoc = -1
    self.fitnesses = [-1] * nPopulation
    self.indices = {}
    self.state = ElectricWorldState(nGenerations=0)
    self.t = t
    self.tstep = tstep
    self.threads = [None] * nThreads
    self.maxFitnessCandidate = [-1] * nThreads
    self.maxFitnessLocCandidate = [-1] * nThreads
    # def threadFunc(ew, nStart, nPerThread):
    #   printts("DOING %d THROUGH %d" % (nStart, min(nStart + nPerThread, nPopulation)), end="", file=mainOutput)
    #   for i in range(nStart, min(nStart + nPerThread, nPopulation)):
    #     ew.population[i] = ElectricCircuit(ew, random.choice(("ADAM","EVE"))+"-%d" % i)
    #     ew.fitnesses[i] = ew.population[i].seed()
    # threads = [None] * math.ceil(nPopulation / nThreads)
    # j = 0
    # nPerThread = math.ceil(nPopulation / nThreads)
    # for i in range(0, nPopulation, nPerThread):
    #   self.threads[j] = Thread(target=threadFunc, args=(self, i, nPerThread))
    #   self.threads[j].daemon = True
    #   self.threads[j].start()
    #   j += 1
    # for i in range(0, nThreads):
    #   if self.threads[i] is not None:
    #     self.threads[i].join()
    for i in range(0, nPopulation):
      self.population[i] = ElectricCircuit(self, random.choice(('Adam','Eve')), str(i))
      self.fitnesses[i] = self.population[i].seed()
  def save(self):
    self.threads = [None] * len(self.population)
    return saveThing64(self)
  def __str__(self):
    return tablify([range(1,len(self.population)+1),[pop.name() for pop in self.population], self.fitnesses], 2, stripesEvery=5, stripeChar='-')
  def __repr__(self):
    return str(self)
  def __getitem__(self, nGenerations): # NEW fxn
    while self.state.nGenerations < nGenerations:
      self.step(nGenerations - self.state.nGenerations)
  def step(self, n):
    printts(self.state.nGenerations, end="", file=mainOutput)
    n = min(n, nThreads)
    self.state += n
    def threadFunc(self, a, b, c, threadNum):
      aFitness = self.fitnesses[a]
      bFitness = self.fitnesses[b]
      cFitness = self.fitnesses[c]
      printts("Thread %d operating on %d,%d, and %d." % (threadNum, a, b, c), end="", file=mainOutput)
      minFitness = min(aFitness, bFitness, cFitness)
      self.maxFitnessCandidate[threadNum] = max(aFitness, bFitness, cFitness, self.maxFitness)
      if aFitness == self.maxFitnessCandidate[threadNum]:
        self.maxFitnessLocCandidate[threadNum] = a
        printts("FITNESS: %s AT %s" % (aFitness, a), end="", file=mainOutput)
      elif bFitness == self.maxFitnessCandidate[threadNum]:
        self.maxFitnessLocCandidate[threadNum] = b
        printts("FITNESS: %s AT %s" % (bFitness, b), end="", file=mainOutput)
      elif cFitness == self.maxFitnessCandidate[threadNum]:
        self.maxFitnessLocCandidate[threadNum] = c
        printts("FITNESS: %s AT %s" % (cFitness, c), end="", file=mainOutput)
      if aFitness == minFitness:
        m = b
        n = c
        r = a
      elif bFitness == minFitness:
        m = a
        n = c
        r = b
      else:
        m = a
        n = b
        r = c
      while True:
        self.population[r] = self.population[m] % self.population[n]
        ~self.population[r]
        rFitness = self.population[r]()
        if rFitness != 0:
          break
        else:
          self.population[r].versionNum += 1
      self.fitnesses[r] = rFitness
    abcs = chunkIntoTuples(getRandInts(0, len(self.population)-1, n * 3), 3)
    for i in range(0, n):
      self.threads[i] = Thread(target=threadFunc, args=(self,) + abcs[i] + (i,))
      self.threads[i].daemon = True
      self.threads[i].start()
    for i in range(0, n):
      self.threads[i].join()
    i = maxLoc(self.maxFitnessCandidate)
    self.maxFitness = self.maxFitnessCandidate[i]
    self.maxFitnessLoc = self.maxFitnessLocCandidate[i]
  def getIndex(self, electricComponent):
    specs = electricComponent.listNodes() + [electricComponent.componentType]
    specs = [len(specs)] + specs
    return getAndStoreUnique(specs, self.indices, self.counter)

# ew = ElectricWorld(15)

# Is a graph (FIX IT)
class Graph():
  def __init__(self):
    self.edges = []
    self.edgeList = {}
    self.nodeList = {}
  def addEdge(self, a, b):
    self.edges += [(a, b)]
    if a in self.edgeList:
      self.edgeList[a] += [b]
    else:
      self.edgeList[a] = [b]
    if b in self.edgeList:
      self.edgeList[b] += [a]
    else:
      self.edgeList[b] = [a]
    if not a in self.nodeList:
      self.nodeList[a] = False
    if not b in self.nodeList:
      self.nodeList[b] = False
  def clearNodeList(self):
    for k in self.nodeList:
      self.nodeList[k] = False
  def findCyclicallyConnected(self, firstNode):
    return set(itertools.chain.from_iterable(self.findCycles(firstNode)))
  def findCycles(self, firstNode):
    if firstNode not in self.nodeList:
      return []
    return self.__findCycles(firstNode, [], False)
  def __findCycles(self, currNode, nodeList, lastNode):
    cycles = []
    if currNode in nodeList:
      return [nodeList[nodeList.index(currNode):]]
    if self.edgeList[currNode]:
      for node in self.edgeList[currNode]:
        if node != lastNode:
          cycles += self.__findCycles(node, nodeList + [currNode], currNode)
    return cycles

# Represents an ElectricCircuit Creature
class ElectricCircuit():
  def __init__(self, electricWorld=None, name="[NO NAME]", circuitNum=-1, generation=0, versionNum=0):
    self.electricWorld = electricWorld
    self.components = {}
    self.componentAddresses = []
    self.circuit = None
    self.componentCounts = dictFromEnumValues(ElectricTypes, lambda: Counter(1))
    self.nNodes = 1
    self.circuitName = name
    self.circuitNum = circuitNum
    self.versionNum = versionNum
    self.generation = generation
  def copy(self, other):
    self.electricWorld = other.electricWorld
    for _, component in other.components.items():
      self.addComponent(component)
    self.circuitName = other.circuitName
    self.circuitNum  = other.circuitNum
    self.versionNum  = other.versionNum
    self.generation  = other.generation
  def randomize(self):
    self.components = {}
    self.componentAddresses = []
    self.nNodes = 1
    self.addComponent(VSource(1,2,12))
    self.addComponent(Resistor(2,3,300))
    self.addComponent(Resistor(3,1,300))
    for j in range(0, numSeedAddNodes):
      self.randomAddNode()
  def seed(self):
    while True:
      self.randomize()
      r = self()
      if r != 0:
        return r
      else:
        self.versionNum += 1
  def __str__(self):
    r = self.name() + '\n'
    nodesInvolved = self.getGraph().findCyclicallyConnected(1)
    nodesInvolved.add(0)
    normalsA = [None] * len(self.components)
    normalsB = [None] * len(self.components)
    trimmeds = [' '] * len(self.components)
    i = 0
    for k, component in self.components.items():
      if set(component.listNodes()) < nodesInvolved:
        templateA = "<%s>"
        templateB = "%s"
      else:
        templateA = "(<%s>"
        templateB = "%s)"
        trimmeds[i] = "TRIMMED"
      normalsA[i] = templateA % k
      normalsB[i] = templateB % self.components[k]
      i += 1
    return ("%s\n" % self.name()) + tablify([normalsA, ['::'] * len(self.components), normalsB, trimmeds], 2, \
                                            stripesEvery=5, stripeChar='-')
  def __repr__(self):
    return str(self)
  def __pos__(self):
    clone = ElectricCircuit(self.electricWorld)
    for (k,component) in self.components.items():
      newComponent = +component
      newComponent.electricCircuit = clone
      clone.addComponent(newComponent)
    return clone
  def name(self):
    return "%s-%s.%s" % (self.circuitName, self.circuitNum, self.versionNum)
  def reset(self):
    self.circuit = None
    self.componentCounts = dictFromEnumValues(ElectricTypes, lambda: Counter(1))
  def getCircuit(self):
    if self.circuit is None:
      self.circuit = ahkab.Circuit(self.name())
      self.circuit.add_model('ekv', 'nmos', dict(TYPE='n', VTO=.4, KP=10e-6))
    return self.circuit
  def setCircuit(self):
    self.reset()
    nodesInvolved = self.getGraph().findCyclicallyConnected(1)
    nodesInvolved.add(0)
    for (k,component) in self.components.items():
      if component.activated:
        addToCircuit = False
        if set(component.listNodes()) < nodesInvolved:
          component.addToCircuit()
  def fitness(self):
    x = max([max(deriv(result, 2)) for k, result in self.res.items()])
    printts("%s's result is %s" % (self.name(), x), end="", file=mainOutput)
    return x
  def __call__(self):
    def processFunc(ec):
      ec.setCircuit()
      returnHighFunctioning = False
      try:
        tran = ahkab.new_tran(tstart=0., tstop=ec.electricWorld.t, tstep=ec.electricWorld.t/2, method='trap')
        ahkab.run(ec.circuit, tran)['tran']
        returnHighFunctioning = True
      except:
        pass
      if returnHighFunctioning:
        tran = ahkab.new_tran(tstart=0., tstop=ec.electricWorld.t, tstep=ec.electricWorld.tstep, method='trap')
        return ahkab.run(ec.circuit, tran)['tran']
    self.res = runForTime(callTimeLimit, processFunc, self)
    if self.res is None:
      self.res = {0: [0,0,0,0]}
    return self.fitness()
  def getComponentNumber(self, componentType):
    return (self.componentCounts[componentType.value])()
  def addComponent(self, electricComponent):
    electricComponent.electricCircuit = self
    self.nNodes = max(self.nNodes, electricComponent.maxNode())
    n = self.electricWorld.getIndex(electricComponent)
    self.components[n] = electricComponent
    self.componentAddresses += [n]
  def getGraph(self):
    g = Graph()
    c = Counter(self.nNodes)
    for (k, component) in self.components.items():
      if component.activated:
        if component.nConnections == 2:
          g.addEdge(component.a, component.b)
        else:
          i = c()
          g.addEdge(component.a, i)
          g.addEdge(component.b, i)
          g.addEdge(component.c, i)
    return g
  def addNode(self):
    self.nNodes += 1
    return self.nNodes - 1
  def __invert__(self):
    random.choice([self.randomAddEdge, self.randomSplitComponent, self.randomComponentMutate, \
                   self.randomAddTriode, self.randomAddNode])()
  def randomAddNode(self):
    componentA = ElectricEdge.random(self.nNodes)
    componentB = ElectricEdge.random(self.nNodes)
    nodeAddress = self.addNode()
    b = componentA.b
    componentA.b = nodeAddress
    componentB.a = nodeAddress
    componentB.b = b
    self.addComponent(componentA)
    self.addComponent(componentB)
  def randomAddEdge(self):
    self.addComponent(ElectricEdge.random(self.nNodes))
  def randomSplitComponent(self):
    nodeAddress = self.addNode()
    componentAddress = random.choice(self.componentAddresses)
    if self.components[componentAddress].nConnections == 3:
      return
    self.components[componentAddress].deactivate()
    componentA = +(self.components[componentAddress])
    componentA.activate()
    componentB = ElectricEdge.random(self.nNodes)
    componentB.b = componentA.b
    componentA.b = nodeAddress
    componentB.a = nodeAddress
    self.addComponent(componentA)
    self.addComponent(componentB)
  def randomComponentMutate(self):
    componentAddress = random.choice(self.componentAddresses)
    ~self.components[componentAddress]
  def randomAddTriode(self):
    self.addComponent(ElectricTriode.random(self.nNodes))
  def __mod__(self, other):
    if self.electricWorld != other.electricWorld:
      raise BreedingError('Worlds must collide.')
    childGeneration = (self.generation + other.generation) // 2 + 1
    child = ElectricCircuit(self.electricWorld, random.choice(generationNames[childGeneration % len(generationNames)]), \
                            '(%s.%d+%s.%d)' % (self.circuitNum, self.versionNum, other.circuitNum, other.versionNum))
    selfComponentAddressSet = set(self.componentAddresses)
    otherComponentAddressSet = set(other.componentAddresses)
    newComponents = []
    for k in selfComponentAddressSet & otherComponentAddressSet:
      if self.components[k].activated or other.components[k].activated or randBool(0.5):
        newComponents += [self.components[k] % other.components[k]]
    for k in selfComponentAddressSet - otherComponentAddressSet:
      newComponent = (+self.components[k])
      if randBool(0.5):
        newComponent.deactivate()
      newComponents += [newComponent]
    for k in otherComponentAddressSet - selfComponentAddressSet:
      newComponent = (+other.components[k])
      if randBool(0.5):
        newComponent.deactivate()
      newComponents += [newComponent]
    for component in newComponents:
      component.electricCircuit = child
      child.addComponent(component)
    return child

# for i in range(0,len(ew.population)):
#   temp = ElectricCircuit()
#   temp.copy(ew.population[i])
#   ew.population[i] = temp
# Represents any electric component
class ElectricComponent():
  def __init__(self, value=0):
    self.value = value
    self.multiplier = 1
    self.activated = True
  def __pos__(self):
    return copy.copy(self)
  def __str__(self):
    return "%s-%s %s" % (scistring(self.value), self.unit, self.nym)
  def __repr__(self):
    return str(self)
  def __invert__(self):
    self.value = mutateReal(self.value, self.canBeNegative)
    if randBool(chanceOfToggleActivation):
      self.toggleActivation()
  def deactivate(self):
    self.activated = False
    return self
  def activate(self):
    self.activated = True
    return self
  def toggleActivation(self):
    self.activated ^= True
    return self
  def name(self):
    if self.electricCircuit:
      return self.nameStem + str(self.electricCircuit.getComponentNumber(self.componentType))

# Represents a triode between a, b, and c
class ElectricTriode(ElectricComponent):
  def __init__(self, a=None, b=None, c=None, value=0):
    ElectricComponent.__init__(self, value)
    self.a = a
    self.b = b
    self.c = c
    self.nConnections = 3
    if b is None:
      self.randomize()
  def __str__(self):
    repString = str((self.a, self.b, self.c))
    if not self.activated:
      repString = strikeout(repString)
    return repString
  def maxNode(self):
    return max(self.a, self.b, self.c)
  def listNodes(self):
    return [self.a, self.b, self.c]
  def A(self):
    if self.a == 0:
      return self.electricCircuit.getCircuit().gnd
    return 'n'+str(self.a)
  def B(self):
    if self.b == 0:
      return self.electricCircuit.getCircuit().gnd
    return 'n'+str(self.b)
  def C(self):
    if self.c == 0:
      return self.electricCircuit.getCircuit().gnd
    return 'n'+str(self.c)
  def randomize(self):
    nNodes = self.a
    value = getRandomValue(self.multiplier)
    a,b,c = getRandInts(0, nNodes - 1, 3)
    self.a = a
    self.b = b
    self.c = c
    self.value = value
  def __mod__(self, other):
    if self.componentType != other.componentType:
      raise BreedingError("Component types must match.")
    if self.a != other.a or self.b != other.b or self.c != other.c:
      raise BreedingError("Component nodes must match.")
    child = ElectricTriode.getClassFromType(self.componentType) \
    (self.a, self.b, self.c, breedReals(self.value, other.value))
    child.activated = random.choice([self.activated, other.activated])
    return child
  def getClassFromType(electricType):
    if electricType is ElectricTypes.MOSFET:
      return Mosfet
  def random(nNodes):
    return ElectricTriode.getClassFromType(random.choice(electricTriodeTypes))(nNodes)

# Represents a MOSFET
class Mosfet(ElectricTriode):
  def __init__(self, a=None, b=None, c=None, value=0):
    ElectricTriode.__init__(self, a, b, c, value)
    self.nameStem = 'M'
    self.componentType = ElectricTypes.MOSFET
  def __str__(self):
    prefix = "MOSFET Transistor"
    if not self.activated:
      prefix = strikeout(prefix)
    return prefix + ElectricTriode.__str__(self)
  def __invert__(self):
    pass
  def addToCircuit(self):
    self.electricCircuit.getCircuit().add_mos(self.name(), nd=self.A(), ng=self.B(), ns=self.C(), nb=self.C(), model_label='nmos',w=600e-6, l=100e-9)

# Represents a connection from a to b
class ElectricEdge(ElectricComponent):
  def __init__(self, a=None, b=None, value=0):
    ElectricComponent.__init__(self, value)
    self.a = a
    self.b = b
    self.nConnections = 2
  def __str__(self):
    repString = ElectricComponent.__str__(self) + str((self.a, self.b))
    if not self.activated:
      repString = strikeout(repString)
    return repString
  def maxNode(self):
    return max(self.a, self.b)
  def listNodes(self):
    return [self.a, self.b]
  def A(self):
    if self.a == 0:
      return self.electricCircuit.getCircuit().gnd
    return 'n'+str(self.a)
  def B(self):
    if self.b == 0:
      return self.electricCircuit.getCircuit().gnd
    return 'n'+str(self.b)
  def randomize(self):
    value = getRandomValue(self.multiplier)
    nNodes = self.a
    a,b = getRandInts(0, nNodes - 1, 2)
    self.a = a
    self.b = b
    self.value = value
  def __mod__(self, other):
    if self.componentType != other.componentType:
      raise BreedingError("Component types must match.")
    if self.a != other.a or self.b != other.b:
      raise BreedingError("Component nodes must match.")
    child = ElectricEdge.getClassFromType(self.componentType) \
    (self.a, self.b, breedReals(self.value, other.value))
    if not random.choice([self.activated, other.activated]):
      child.deactivate()
    return child
  def getClassFromType(electricType):
    if electricType is ElectricTypes.RESISTOR:
      return Resistor
    elif electricType is ElectricTypes.CAPACITOR:
      return Capacitor
    elif electricType is ElectricTypes.INDUCTOR:
      return Inductor
    elif electricType is ElectricTypes.VSOURCE:
      return VSource
    elif electricType is ElectricTypes.ISOURCE:
      return ISource
  def random(nNodes):
    return ElectricEdge.getClassFromType(random.choice(electricEdgeTypes))(nNodes)

# Is a resistor
class Resistor(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'R'
    self.componentType = ElectricTypes.RESISTOR
    self.unit = 'Ohm'
    self.nym  = 'Resistor'
    self.multiplier = 100
    self.canBeNegative = False
    if self.b is None:
      self.randomize()
  def addToCircuit(self):
    self.electricCircuit.getCircuit().add_resistor(self.name(), n1=self.A(), n2=self.B(), value=self.value)

# Is a capacitoree
class Capacitor(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'C'
    self.componentType = ElectricTypes.CAPACITOR
    self.unit = 'Farad'
    self.nym  = 'Capacitor'
    self.multiplier = 1e-9
    self.canBeNegative = False
    if self.b is None:
      self.randomize()
  def addToCircuit(self):
    self.electricCircuit.getCircuit().add_capacitor(self.name(), n1=self.A(), n2=self.B(), value=self.value)

# Is a inductor
class Inductor(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'L'
    self.componentType = ElectricTypes.INDUCTOR
    self.unit = 'Henry'
    self.nym  = 'Inductor'
    self.multiplier = 1e-5
    self.canBeNegative = False
    if self.b is None:
      self.randomize()
  def addToCircuit(self):
    self.electricCircuit.getCircuit().add_inductor(self.name(), n1=self.A(), n2=self.B(), value=self.value)

# Is a Voltage Source
class VSource(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'V'
    self.componentType = ElectricTypes.VSOURCE
    self.unit = 'Volt'
    self.nym  = 'Voltage Source'
    self.canBeNegative = True
    if self.b is None:
      self.randomize()
  def addToCircuit(self):
    self.electricCircuit.getCircuit().add_vsource(self.name(), n1=self.A(), n2=self.B(), dc_value=self.value)

# Is a Current Source
class ISource(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'I'
    self.componentType = ElectricTypes.ISOURCE
    self.unit = 'Ampere'
    self.nym  = 'Current Source'
    self.canBeNegative = True
    if self.b is None:
      self.randomize()
  def addToCircuit(self):
    self.electricCircuit.getCircuit().add_isource(self.name(), n1=self.A(), n2=self.B(), dc_value=self.value)

# Turns a gray coded binary number into a two's complement binary number
def grayToBin(gray, n):
  mask = 1 << n - 1
  if mask & gray != 0:
    lastbit = 1
  else:
    lastbit = 0
  mask >>= 1
  binary = 0
  while mask != 0:
    binary = (binary << 1) | lastbit
    if mask & gray != 0:
      currbit = 1
    else:
      currbit = 0
    lastbit ^= currbit
    mask >>= 1
  binary = (binary << 1) | lastbit
  return binary

# Turns a two's complement binary number into a gray coded binary number
def binToGray(bin):
  return bin ^ (bin >> 1)

# take a positive floating point number and encode it in our gray scheme
def absFloatToGray(num):
  if num == 0:
    return 0
  exponent = math.log2(num)
  if exponent >= 0:
    exponent = math.floor(exponent)
    num /= 1 << exponent
  else:
    exponent = math.ceil(-exponent)
    num *= 1 << exponent
    exponent = -exponent
  significand = math.floor((num - 1) * (1 << nSignificand))
  exponent = (exponent + (1 << nExponent - 1)) & ((1 << nExponent) - 1)
  significand = binToGray(significand)
  exponent = binToGray(exponent)
  return (significand << nExponent) | exponent

# take a floating point number and encode it in our gray scheme
def floatToGray(num):
  if num == 0:
    return 0
  if num < 0:
    sign = 1 << nSignificand + nExponent - 1
    num = -num
  else:
    sign = 0
  exponent = math.log2(num)
  if exponent >= 0:
    exponent = math.floor(exponent)
    num /= 1 << exponent
  else:
    exponent = math.ceil(-exponent)
    num *= 1 << exponent
    exponent = -exponent
  significand = math.floor((num - 1) * (1 << nSignificand - 1))
  exponent = (exponent + (1 << nExponent - 1)) & ((1 << nExponent) - 1)
  significand = binToGray(significand)
  exponent = binToGray(exponent)
  return sign | (significand << nExponent) | exponent

# Turns a gray coded fp number into a native fp number (no negatives)
def grayToAbsFloat(gray):
  if gray == 0:
    return 0
  significand = gray >> nExponent
  exponent = gray & ((1 << nExponent) - 1)
  significand = grayToBin(significand,nSignificand)
  exponent = grayToBin(exponent,nExponent)
  exponent -= 1 << nExponent - 1
  if exponent >= 0:
    return (1 + significand / (1 << nSignificand)) * (1 << exponent)
  else: 
    return (1 + significand / (1 << nSignificand)) / (1 << -exponent)

# Turns a gray coded fp number into a native fp numberd
def grayToFloat(gray):
  if gray & ((1 << nExponent + nSignificand - 1) - 1) == 0:
    return 0
  significand = gray >> nExponent
  if significand & (1 << nSignificand - 1):
    sign = -1
  else:
    sign = 1
  significand &= (1 << nSignificand - 1) - 1
  exponent = gray & ((1 << nExponent) - 1)
  significand = grayToBin(significand,nSignificand)
  exponent = grayToBin(exponent,nExponent)
  exponent -= 1 << nExponent - 1
  if exponent >= 0:
    return sign * (1 + significand / (1 << nSignificand - 1)) * (1 << exponent)
  else:
    return sign * (1 + significand / (1 << nSignificand - 1)) / (1 << -exponent)  

mainOutput = open("%s-RUN.txt" % time.ctime(), "a")
ew = ElectricWorld(10)
ew[100]

printts("-------------\n-------------\n-------------\n-------------\n-------------\n-------------\n-------------\n", end="", file=mainOutput)
def foo():
  ew = ElectricWorld(100)
  ew[100]

backgroundRun(foo)
