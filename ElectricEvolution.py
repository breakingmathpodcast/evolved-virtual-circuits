# Evolves a circuit.
# Copyright 2017 Gabriel Hesch and Jonathan Baca

from bintrees import RBTree
import os
import re
import numpy
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

maxTries = 50

nThreads = 5
callTimeLimit = 1

nSignificand = 53 # Precision of floating point numbers
nExponent    = 11 # Range of floating point numbers

tDefault     = 1e-3
tstepDefault = 1e-6

grimReaperMinQuota = 10000
grimReaperMaxQuota = 15000
grimReaperMin = 1/100
grimReaperMax = 1
grimReaperQuota = 1/3
grimReaperQuotaFudge = 1/4
mateStudProbability = 1/3
chanceOfToggleActivation = 1/50 # Ask an expert
chanceOfRealSignFlip = 1/40 # Ask an expert
realMutationSigma    = .1   # Ask an expert
numSeedAddNodesMin = 3
numSeedAddNodesMax = 6
grimReaperNotice = '                              __________\n                           .~#########%%;~.\n                          /############%%;`\\\n                         /######/~\\/~\\%%;,;,\\\n                        |#######\\    /;;;;.,.|\n                        |#########\\/%;;;;;.,.|\n               XX       |##/~~\\####%;;;/~~\\;,|       XX\n             XX..X      |#|  o  \\##%;/  o  |.|      X..XX\n           XX.....X     |##\\____/##%;\\____/.,|     X.....XX\n      XXXXX.....XX      \\#########/\\;;;;;;,, /      XX.....XXXXX\n     X |......XX%,.@      \\######/%;\\;;;;, /      @#%,XX......| X\n     X |.....X  @#%,.@     |######%%;;;;,.|     @#%,.@  X.....| X\n     X  \\...X     @#%,.@   |# # # % ; ; ;,|   @#%,.@     X.../  X\n      X# \\.X        @#%,.@                  @#%,.@        X./  #\n       ##  X          @#%,.@              @#%,.@          X   #\n     , "# #X            @#%,.@          @#%,.@            X ##\n        `###X             @#%,.@      @#%,.@             ####\'\n       . \' ###              @#%.,@  @#%,.@              ###`"\n         . ";"                @#%.@#%,.@                ;"` \' .\n           \'                    @#%,.@                   ,.\n           ` ,                @#%,.@  @@                `\n                               @@@  @@@  \n\n                e88\'Y88  888 88e  888     e   e     \n               d888  \'Y  888 888D 888    d8b d8b    \n              C8888 eeee 888 88"  888   e Y8b Y8b   \n               Y888 888P 888 b,   888  d8b Y8b Y8b  \n                "88 88"  888 88b, 888 d888b Y8b Y8b                       \n                                      \n888 88e  888\'Y88     e Y8b     888 88e  888\'Y88 888 88e  888 888 888 \n888 888D 888 ,\'Y    d8b Y8b    888 888D 888 ,\'Y 888 888D 888 888 888 \n888 88"  888C8     d888b Y8b   888 88"  888C8   888 88"  "8" "8" "8" \n888 b,   888 ",d  d888888888b  888      888 ",d 888 b,    e   e   e  \n888 88b, 888,d88 d8888888b Y8b 888      888,d88 888 88b, "8" "8" "8" \''

generationNames = [['Adam'], ['Seth'], ['Enos'], ['Cainan'], ['Mahaleel'], ['Jared'],\
['Enoch'], ['Methusaleh'], ['Lamech'], ['Noah'], ['Shem'], ['Arphaxad'], ['Cainan'], ['Sala'],\
['Eber'], ['Peleg'], ['Ragau'], ['Saruch'], ['Nahor'], ['Terah'], ['Abraham'], ['Isaac'],\
['Jacob'], ['Juda'], ['Pharez'], ['Esrom'], ['Aram'], ['Amminadab'], ['Naason'], ['Salmon'],\
['Boaz'], ['Obed'], ['Jesse'], ['David'], ['Solomon'], ['Rehoboam'], ['Abia'], ['Asa'],\
['Josophat'], ['Joram'], ['Ozias'], ['Joatham'], ['Achaz'], ['Ezekias'], ['Manasses'],\
['Amon'], ['Josias'], ['Jehoikim'], ['Jechonias'], ['Salathiel'], ['Zerubbabel'], ['Rhesa'],\
['Joanna'], ['Juda'], ['Joseph'], ['Semei'], ['Mattathias '], ['Maath'], ['Nagge'], ['Esli'],\
['Nahum'], ['Amos'], ['Mattathias'], ['Joseph'], ['Janna'], ['Melchi'], ['Levi'], ['Matthat'],\
['Heli'], ['Mary '], ['Jesus']];

# Calculates f(SUM(i=0->n) (xs(n)-ys(n))^2)
def funcCartesianDistance(xs, ys, bias=1e6, func=lambda x:1/x):
  s = bias
  s += sum([(lambda x:abs(x*x))((lambda w,q:w-q)(*z)) for z in itertools.zip_longest(xs,ys)])
  return func(s)

def norm(xs):
  return math.sqrt(sum([(lambda x:abs(x*x))(x) for x in xs]))

def targetFunc(n):
  x = 30 * n
  if x == 6:
    return 12
  return 12 * (math.sin(x-6)) / (x-6)

def sampleObjectiveFunction(xs, bias=1e-16):
  cmpSig = [targetFunc(n/(len(xs)-1)) for n in range(0,len(xs))]
  denominator = norm(xs)*norm(cmpSig)
  if denominator == 0:
    return 0
  return norm(numpy.asarray(xs) * numpy.asarray(cmpSig)) / denominator
  # newXs = [0] * (2 * len(xs))
  # for i in range(0, len(xs)):
  #   newXs[i] = xs[i]
  # sampleObjectiveFunctionData = [0] * len(xs) + [targetFunc(n/(len(xs) - 1)) for n in range(len(xs) - 1, -1, -1)]
  # sampleObjectiveFunctionData = numpy.fft.fft(sampleObjectiveFunctionData)
  # newXs = numpy.fft.fft(newXs)
  # sampleObjectiveFunctionData = numpy.fft.ifft(newXs * sampleObjectiveFunctionData)
  # return 1 + abs(max(sampleObjectiveFunctionData)) / (norm(xs) * norm([targetFunc(n/(len(xs) - 1)) for n in range(0, len(xs))]))

def plotText(data, width, height=75):
  minDatum = min(data[1:])
  maxDatum = max(data[1:])
  diffDatum = maxDatum - minDatum
  if diffDatum == 0:
    diffDatum = 1000
  print("MIN: %s, MAX: %s" % (minDatum, maxDatum))
  ret = ''
  for i in range(1, len(data), math.ceil((len(data) - 1) / width)):
    datum = data[i]
    r = '*' * math.floor(width * (datum - minDatum) / diffDatum)
    ret += r + '\n'
  return ret

# Replaces "ab <c> d<<e>> <f><<f>>gh", {'c': 'x', 'f': 23} with
# "ab x d<e> 23<f>gh"
# BUGGY! FIX BUGS!
def replaceTagged(string, d):
  for k,x in d.items():
    kesc = re.escape(k)
    string = re.sub('((?!<).<'+kesc+'>(?!>))|^<'+kesc+'>', \
                    re.escape(str(x)), string)
    string = re.sub('<'+kesc+'>', kesc, string)
  return stringa

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
    message = ElectricMessage(infoID=self.infoID, returnInfo=self.returnInfo)
    for job in self.__docket__:
      if job == 'breed':
        message += self.breed()
      if job == 'exec':
        message += self.execute()
  def breed(self):
    mother = loadThing64(self.mother)
    father = loadThing64(self.father)
    while True:
      offspring = mother % father
      ~offspring
      fitness = offspring()
      if fitness != 0:
        break
    return ElectricMessage(offspring=offspring, fitness=fitness)
  def execute(self):
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
def maxLoc(xs, key=lambda x:x):
  m = key(xs[0])
  j = 0
  for i in range(1,len(xs)):
    if m < key(xs[i]):
      m = key(xs[i])
      j = i
  return j

# gets n unique random integers in [a,b]
def getRandInts(a, b, n):
  return random.sample(range(a,b+1), n)

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
def getRandomValue(minValue, maxValue):
  return random.random() * (maxValue - minValue) + minValue
  # return multiplier * random.lognormvariate(0, .25)

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
      b = 0
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
  def __str__(self):
    return '[%d]' % self.nGenerations
  def __repr__(self):
    return str(self)
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
  def __init__(self, nPopulation=1000, t=tDefault, tstep=tstepDefault): # NEW: nP..n
    self.counter = Counter()
    self.nodeCounter = Counter(4) # KLUDGE: FIX THIS.
    self.population = []
    self.maxFitness = -1
    self.maxFitnessLoc = -1
    self.fitnesses = [] * nPopulation
    self.indices = {}
    self.nodeIndices = {}
    self.state = ElectricWorldState(nGenerations=0)
    self.t = t
    self.tstep = tstep
    for i in range(0, nPopulation):
      newCircuit = ElectricCircuit(self, 'Eve', str(i))
      newCircuit.seed()
      self.addCircuit(newCircuit)
      # self.fitnesses[i] = self.population[i].seed()
      # if self.fitnesses[i] > self.maxFitness:
      #   self.maxFitness = self.fitnesses[i]
      #   self.maxFitnessLoc = i
      # self.population[i].address = i
  def grimReaper(self):
    printts(grimReaperNotice, end="", file=mainOutput)
    damned = set()
    quota = grimReaperQuota * (1 + grimReaperQuotaFudge * random.random()) * len(self.population)
    while len(damned) < quota:
      for i in range(0, len(self.population)):
        if len(damned) < quota:
          if i == self.maxFitnessLoc:
            pSurvival = 2.0
          else:
            pSurvival = (self.fitnesses[i] / self.maxFitness) * (grimReaperMax - grimReaperMin) + grimReaperMin
          if not randBool(pSurvival):
            damned.add(i)
    damned = sorted(list(damned))
    offset = 0
    for soul in damned:
      del self.population[soul + offset]
      del self.population[soul + offset]
      offset -= 1
    self.maxFitnessLoc = maxLoc(self.fitnesses)
    self.maxFitness = self.fitnesses[self.maxFitnessLoc]
  def addCircuit(self, electricCircuit):
    newCircuit = ElectricCircuit(self)
    newCircuit.mimic(electricCircuit)
    n = len(self.population)
    for k, component in electricCircuit.components.items():
      self.getIndex(component)
    electricCircuit.address = n
    self.population += [electricCircuit]
    self.fitnesses += [electricCircuit()]
    if self.fitnesses[n] > self.maxFitness:
      self.maxFitness = self.fitnesses[n]
      self.maxFitnessLoc = n
  def addWorld(self, electricWorld):
    for circuit in electricWorld.population:
      self.addCircuit(circuit)
  def save(self):
    return saveThing64(self)
  def __str__(self):
    return tablify([range(1,len(self.population)+1),[pop.name() for pop in self.population], self.fitnesses], 2, stripesEvery=5, stripeChar='-')
  def __repr__(self):
    return str(self)
  def __getitem__(self, nGenerations): # NEW fxn
    while self.state.nGenerations < nGenerations:
      self.step()
  def step(self):
    if randBool(max(0, len(self.population) - grimReaperMinQuota) / (grimReaperMaxQuota - grimReaperMinQuota)):
      self.grimReaper()
    nTries = 0
    mateStud = randBool(mateStudProbability)
    while True:
      while True:
        a,b,c = getRandInts(0, len(self.population)-1, 3)
        if mateStud:
          a = self.maxFitnessLoc
        fa,fb,fc = self.fitnesses[a],self.fitnesses[b],self.fitnesses[c]
        (a,fa),(b,fb),(c,fc) = sorted(((a,fa),(b,fb),(c,fc)),key=lambda x:x[1])
        x = self.population[b] * self.population[c]
        if x > .3:
          break
      count = 0
      printts("Trying %d and %d" % (c, b), end="", file=mainOutput)
      while count < 3:
        count += 1
        new = self.population[c] % self.population[b]
        ~new
        k = new()
        if k != 0:
          break
        nTries += 1
      if k != 0:
        break
    new.versionNum = nTries
    if randBool(.5):
      printts("Replacing %d." % a, end="", file=mainOutput)
      new.address = a
      self.population[a] = new
      self.fitnesses[a] = k
      if k > self.maxFitness:
        self.maxFitness = k
        self.maxFitnessLoc = a
    else:
      new.address = len(self.population)
      printts("New address %d." % new.address, end="", file=mainOutput)
      self.population += [new]
      self.fitnesses += [k]
      if k > self.maxFitness:
        self.maxFitness = k
        self.maxFitnessLoc = new.address
    self.state += 1
    printts("Max fitness: %s." % self.maxFitness, end="", file=mainOutput)
    printts("Curr generation: %s" % self.state, end="", file=mainOutput)
    return new
  def getIndex(self, electricComponent):
    specs = electricComponent.listNodes() + [electricComponent.componentType]
    specs = [len(specs)] + specs
    return getAndStoreUnique(specs, self.indices, self.counter)
  def getNodeIndex(self, a, b, n):
    if a < b:
      a,b = b,a
    return getAndStoreUnique([a,b,n], self.nodeIndices, self.nodeCounter)

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
    return self.__findCycles(firstNode, firstNode, [], False)
  def __findCycles(self, firstNode, currNode, nodeList, lastNode):
    cycles = []
    if currNode in nodeList:
      newCycle = nodeList[nodeList.index(currNode):]
      if firstNode in newCycle:
        return [newCycle]
      return []
    if self.edgeList[currNode]:
      for node in self.edgeList[currNode]:
        if node != lastNode:
          cycles += self.__findCycles(firstNode, node, nodeList + [currNode], currNode)
    return cycles

# Represents an ElectricCircuit Creature
class ElectricCircuit():
  def __init__(self, electricWorld=None, name="[NO NAME]", lineage=-1, generation=0, versionNum=0):
    self.electricWorld = electricWorld
    self.components = {}
    self.componentAddresses = []
    self.circuit = None
    self.componentCounts = dictFromEnumValues(ElectricTypes, lambda: Counter(1))
    self.maxNodes = 1
    self.nodes = []
    self.nodeIndices = {}
    self.nodeCounter = Counter()
    self.circuitName = name
    self.lineage = lineage
    self.versionNum = versionNum
    self.generation = generation
  def mimic(self, other):
    self.circuitName = other.circuitName
    for _, component in other.components.items():
      self.addComponent(component)
    self.circuitName = other.circuitName
    self.lineage  = other.lineage
    self.versionNum  = other.versionNum
    self.generation  = other.generation
  def copy(self, other):
    self.electricWorld = other.electricWorld
    for _, component in other.components.items():
      self.addComponent(component)
    self.circuitName = other.circuitName
    self.lineage  = other.lineage
    self.versionNum  = other.versionNum
    self.generation  = other.generation
  def randomize(self):
    self.components = {}
    self.componentAddresses = []
    self.maxNodes = 1
    self.nodes = []
    self.nodeIndices = {}
    self.nodeCounter = Counter()
    self.addComponent(VSource(1,2,12))
    for j in range(0, random.randint(numSeedAddNodesMin, numSeedAddNodesMax)):
      self.randomAddNode()
  def seed(self):
    while True:
      self.randomize()
      r = self()
      if r != 0:
        return r
      else:
        self.versionNum += 1
  def __mul__(self, other):
    myAddresses = set(self.componentAddresses)
    theirAddresses = set(other.componentAddresses)
    return len(myAddresses&theirAddresses)/len(myAddresses|theirAddresses)
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
    return "%s(%s)-%s.%s" % (self.circuitName, self.generation, self.lineage, self.versionNum)
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
    nodeDictionary = dict(itertools.zip_longest(list(nodesInvolved), range(0,len(nodesInvolved))))
    for (k,component) in self.components.items():
      if component.activated:
        addToCircuit = False
        if set(component.listNodes()) < nodesInvolved:
          component.addToCircuit(nodeDictionary)
  def fitness(self):
    print(self)
    self.setCircuit()
    print(self.circuit)
    x = 0
    if len(self.res) > 0:
      try:
        vs = [(k, sampleObjectiveFunction(xs)) if str(k)[0] == 'V' else (k, 0) for k, xs in self.res.items()]
        i = maxLoc(vs, key=lambda x:x[1])
        x = vs[i][1]
      except:
        x = 0
      # x = max([sampleObjectiveFunction(xs) if str(k)[0] == 'V' else 0 for k, xs in self.res.items()])
    # x = max([max(deriv(result, 2)) for k, result in self.res.items()])
    if x != 0:
      print(tablify([plotText(self.res[vs[i][0]], 50).split('\n'), plotText([targetFunc(n/49) for n in range(0,50)], 50).split('\n')]))
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
    self.maxNodes = max(self.maxNodes, electricComponent.maxNode())
    for node in electricComponent.listNodes():
        if node not in self.nodes:
            self.nodes += [node]
    n = self.electricWorld.getIndex(electricComponent)
    self.components[n] = electricComponent
    self.componentAddresses += [n]
  def getGraph(self):
    g = Graph()
    c = Counter(self.maxNodes + 1)
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
  def addNode(self, a, b):
    n = getAndStoreUnique([a, b], self.nodeIndices, self.nodeCounter)
    newNode = self.electricWorld.getNodeIndex(a, b, n)
    self.nodes += [newNode]
    return newNode
  def __invert__(self):
    random.choice([self.randomAddEdge, self.randomSplitComponent, self.randomComponentMutate, \
                   self.randomAddTriode, self.randomAddNode])()
  def randomAddNode(self):
    componentA = ElectricEdge.random(self.nodes)
    componentB = ElectricEdge.random(self.nodes)
    nodeAddress = self.addNode(componentA.a, componentA.b)
    b = componentA.b
    componentA.b = nodeAddress
    componentB.a = nodeAddress
    componentB.b = b
    self.addComponent(componentA)
    self.addComponent(componentB)
  def randomAddEdge(self):
    self.addComponent(ElectricEdge.random(self.nodes))
  def randomSplitComponent(self):
    componentAddress = random.choice(self.componentAddresses)
    if self.components[componentAddress].nConnections == 3:
      return
    nodeAddress = self.addNode(self.components[componentAddress].a, self.components[componentAddress].b)
    self.components[componentAddress].deactivate()
    componentA = +(self.components[componentAddress])
    componentA.activate()
    componentB = ElectricEdge.random(self.nodes)
    if randBool(0.5):
      componentB.a = nodeAddress
      componentB.b = componentA.b
      componentA.b = nodeAddress
    else:
      componentB.a = componentA.a
      componentB.b = nodeAddress
      componentA.a = nodeAddress
    self.addComponent(componentA)
    self.addComponent(componentB)
  def randomComponentMutate(self):
    componentAddress = random.choice(self.componentAddresses)
    ~self.components[componentAddress]
  def randomAddTriode(self):
    self.addComponent(ElectricTriode.random(self.nodes))
  def __mod__(self, other):
    if self.electricWorld != other.electricWorld:
      raise BreedingError('Worlds must collide.')
    childGeneration = (self.generation + other.generation) / 2 + 1
    child = ElectricCircuit(self.electricWorld, random.choice(generationNames[math.floor(childGeneration) % len(generationNames)]), \
                            '(%s@%d.%d+%s@%d.%d)' % (self.circuitName, self.address, self.versionNum, other.circuitName, other.address, other.versionNum))
    selfComponentAddressSet = set(self.componentAddresses)
    otherComponentAddressSet = set(other.componentAddresses)
    newComponents = []
    for k in selfComponentAddressSet & otherComponentAddressSet:
      if self.components[k].activated or other.components[k].activated or randBool(0.5):
        newComponents += [self.components[k] % other.components[k]]
    for k in selfComponentAddressSet - otherComponentAddressSet:
      newComponent = +self.components[k]
      # if randBool(0.15):
      #   newComponent.deactivate()
      newComponents += [newComponent]
    for k in otherComponentAddressSet - selfComponentAddressSet:
      newComponent = +other.components[k]
      keepComponent = True
      if randBool(0.75):
        if randBool(0.5):
          keepComponent = False
        else:
          newComponent.deactivate()
      if keepComponent:
        newComponents += [newComponent]
    for component in newComponents:
      component.electricCircuit = child
      child.addComponent(component)
    child.generation = childGeneration
    return child

# Represents any electric component
class ElectricComponent():
  def __init__(self, value=0):
    self.value = value
    self.multiplier = 1
    self.minValue = 1
    self.maxValue = 30
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
    nodes = self.a
    value = getRandomValue(self.minValue, self.maxValue)
    self.a,self.b,self.c = random.sample(nodes, 3)
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
  def random(nodes):
    return ElectricTriode.getClassFromType(random.choice(electricTriodeTypes))(nodes)

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
  def addToCircuit(self, nodeDictionary):
    atemp, btemp, ctemp = self.a,self.b,self.c
    self.a = nodeDictionary[self.a]
    self.b = nodeDictionary[self.b]
    self.c = nodeDictionary[self.c]
    self.electricCircuit.getCircuit().add_mos(self.name(), nd=self.A(), ng=self.B(), ns=self.C(), nb=self.C(), model_label='nmos',w=600e-6, l=100e-9)
    self.a,self.b,self.c = atemp, btemp, ctemp

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
    nodes = self.a
    value = getRandomValue(self.minValue, self.maxValue)
    self.a, self.b = random.sample(nodes, 2)
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
  def random(nodes):
    return ElectricEdge.getClassFromType(random.choice(electricEdgeTypes))(nodes)

# Is a resistor
class Resistor(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'R'
    self.componentType = ElectricTypes.RESISTOR
    self.unit = 'Ohm'
    self.nym  = 'Resistor'
    self.multiplier = 100
    self.minValue = 1
    self.maxValue = 1e6
    self.canBeNegative = False
    if self.b is None:
      self.randomize()
  def addToCircuit(self, nodeDictionary):
    atemp, btemp = self.a, self.b
    self.a = nodeDictionary[self.a]
    self.b = nodeDictionary[self.b]
    self.electricCircuit.getCircuit().add_resistor(self.name(), n1=self.A(), n2=self.B(), value=self.value)
    self.a, self.b = atemp, btemp

# Is a capacitoree
class Capacitor(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'C'
    self.componentType = ElectricTypes.CAPACITOR
    self.unit = 'Farad'
    self.nym  = 'Capacitor'
    self.multiplier = 1e-9
    self.minValue = 1e-13
    self.maxValue = 1e-5
    self.canBeNegative = False
    if self.b is None:
      self.randomize()
  def addToCircuit(self, nodeDictionary):
    atemp, btemp = self.a, self.b
    self.a = nodeDictionary[self.a]
    self.b = nodeDictionary[self.b]
    self.electricCircuit.getCircuit().add_capacitor(self.name(), n1=self.A(), n2=self.B(), value=self.value)
    self.a, self.b = atemp, btemp

# Is a inductor
class Inductor(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'L'
    self.componentType = ElectricTypes.INDUCTOR
    self.unit = 'Henry'
    self.nym  = 'Inductor'
    self.multiplier = 1e-5
    self.minValue = 1e-9
    self.maxValue = 1e-3
    self.canBeNegative = False
    if self.b is None:
      self.randomize()
  def addToCircuit(self, nodeDictionary):
    atemp, btemp = self.a, self.b
    self.a = nodeDictionary[self.a]
    self.b = nodeDictionary[self.b]
    self.electricCircuit.getCircuit().add_inductor(self.name(), n1=self.A(), n2=self.B(), value=self.value)
    self.a, self.b = atemp, btemp

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
  def addToCircuit(self, nodeDictionary):
    atemp, btemp = self.a, self.b
    self.a = nodeDictionary[self.a]
    self.b = nodeDictionary[self.b]
    self.electricCircuit.getCircuit().add_vsource(self.name(), n1=self.A(), n2=self.B(), dc_value=self.value)
    self.a, self.b = atemp, btemp

# Is a Voltage Source
class ISource(ElectricEdge):
  def __init__(self, a=None, b=None, value=0):
    ElectricEdge.__init__(self, a, b, value)
    self.nameStem = 'V'
    self.componentType = ElectricTypes.VSOURCE
    self.unit = 'Volt'
    self.nym  = 'Voltage Source'
    self.minValue = 0.01
    self.maxValue = 1
    self.canBeNegative = True
    if self.b is None:
      self.randomize()
  def addToCircuit(self, nodeDictionary):
    atemp, btemp = self.a, self.b
    self.a = nodeDictionary[self.a]
    self.b = nodeDictionary[self.b]
    self.electricCircuit.getCircuit().add_vsource(self.name(), n1=self.A(), n2=self.B(), dc_value=self.value)
    self.a, self.b = atemp, btemp

mainOutput = open("%s-RUN.txt" % time.ctime(), "a")

def main():
  ew = ElectricWorld(1000)
  ewList = sorted([int(f[2:len(f)-2]) for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and re.match('ew\d+.p', f)])
  if len(ewList) < 2:
    if len(ewList) == 1:
      n = 2 if ewList[0] == 1 else 1
    else:
      n = 1
  else:
    for i in range(0, len(ewList) - 1):
      if ewList[i+1] - ewList[i] > 1:
        n = ewList[i] + 1
  pickle.dump(ew, open("ew%d.p" % n, "wb"))
# 
  with open("ew%d.txt","w") as f:
    f.write(time.ctime())

if __name__ == "__main__":
  main()