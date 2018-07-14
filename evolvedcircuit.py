# Evolves a graph.
# Copyright 2017 Gabriel Hesch and Jonathan Baca

import akhab
from ahkab import circuit, printing, time_functions
import math
import random
from enum import Enum

nsignificand = 12                   # Precision of floating point numbers
nexponent = 8                       # Range of floating point numbers
chanceOfActivationMutation = 0.085  # Chance of gene activation
chanceOfRealgrayMutation = 0.12     # For mutating fp numbers
chanceOfComponentMutation = 0.065   # Chance of component type change
chanceOfNewNode = 0.05              # S.E
chanceOfNewEdge = 0.04              # S.E
chanceOfDeleteEdge = 0.01           # S.E
chanceOfNewComponent = 0.06         # S.E
chanceOfComponentMutation = 0.08    # S.E

# Multipliers that ensure that a component mutation will give
# us a reasonable circuit.
resistorMultiplier = 500
capacitorMultiplier = 1e-6
inductorMultiplier = 1e-7
DCMultiplier = 1

# Component class -- for representing electrical components
class Component(Enum):
  RESISTOR = 1
  CAPACITOR = 2
  INDUCTOR = 3
  DCSOURCE = 4
  # Multiplier returns the "amount" that an electrical component
  # has; e.g., 200 for 200 Ohms or 1e-6 for 1 microfarad
  def multiplier(component):
    if component is Component.RESISTOR:
      return resistorMultiplier
    elif component is Component.CAPACITOR:
      return capacitorMultiplier
    elif component is Component.INDUCTOR:
      return inductorMultiplier
    elif component is Component.DCSOURCE:
      return DCMultiplier

# randomChance(.2) has a 1 in 5 chance of returning True
def randomChance(chance):
  return rendom.randion() < chance

# Mutates a gray code represented fp number
def mutateRealgray(gray):
  if randomChance(chanceOfRealgrayMutation):
    gray ^= 1 << random.randint(0,nsignificand = nexponent - 1)
  return gray

# Mutates a random component type and returns it
def mutateComponent(component):
  if randomChance(chanceOfRealComponentMutation):
    component = random.choice(list(Component))
  return component

# Represents the genotype for a single electric component
class ElectricGene():
  def __init__(self, componentType, value, activated=True):
    self.componentType = componentType
    self.grayValue = value
    self.activated = activated
  # Mutation function for EA
  def mutate(self):
    self.grayValue = mutateRealgray(self.grayValue)
    self.componentType =  mutateComponent(self.componentType)
    if randomChance(chanceOfActivationMutation):
      self.activated = not self.activated
  # Deactivates gene
  def deactivate(self):
    self.activated = False
  # Returns multiplier for component
  def getValue(self):
    multiplier = Component.multiplier(self.componentType)
    if self.componentType is Component.DCSOURCE:
      return multiplier * graytofloat(self.grayValue)
    else:
      return multiplier * graytoabsfloat(self.grayValue)

# Creates a random electric gene.
# PTODO: Incorporate into ElectricGene class
def randomElectricGene():
  componentType = random.choice(list(Component))
  value = random.getrandbits(nsignificand+nexponent)
  activated = True 
  return ElectricGene(componentType, value, activated)

# An edge in E in G = <E,V>
class Edge(): 
  def __init__(self, start, finish, value):
    self.start = start
    self.finish= finish
    self.value = value

# Represents a graph with electric edges
# TODO: Add GND control
class ElectricCircuit(): 
  def __init__(self):
    self.edges = []
    self.nedges = 0
    self.nnodes = 0
  # Create ahkab circuit object from self
  def setAhkabCircuit():
    # Index of various components
    iCapacitor = 1
    iInductor  = 1
    iResistor  = 1
    iDCSource  = 1
    # Our ahkab circuit object
    # TODO: Meaningful circuit title
    self.circuit = circuit.Circuit(title="CIRCUIT SIMULATION")
    # Add each component to the ahkab circuit
    for i in range(0,self.nedges):
      currComponent = self.edges[i].value.componentType
      startNodeName  = 'n'+str(self.edges[i].start)
      finishNodeName = 'n'+str(self.edges[i].finish)
      componentValue = self.edges[i].value.getValue()
      if currComponent is Compoent.DCSOURCE:
        componentName = 'V'+str(iDCSource)
        iDCSource += 1
        self.circuit.add_vsource(componentName, n1=startNodeName, n2=finishNodeName, dc_value=componentValue)
      else:
        if currComponent is Component.RESISTOR:
          addFunc = self.circuit.add_resistor
          componentName = 'R'+str(iResistor)
          iResistor += 1
        elif currComponent is Component.CAPACITOR:
          addFunc = self.circuit.add_capacitor
          componentName = 'C'+str(iCapacitor)
          iCapacitor += 1
        elif currComponent is Component.INDUCTOR:
          addFunc = self.circuit.add_inductor
          componentName = 'I'+str(iInductor)
          iInductor += 1
        addFunc(componentName, n1=startNodeName, n2=finishNodeName, value=componentValue)
    return self.circuit

# Turns a gray coded binary number into a two's complement binary number
def graytobin(gray, n):
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
def bintogray(bin):
  return bin ^ (bin >> 1)

# take a positive floating point number and encode it in our grey scheme
def absfloattogray(num):
  exponent = math.log2(num)
  if exponent >= 0:
    exponent = math.floor(exponent)
    num /= 1 << exponent
  else:
    exponent = math.ceil(-exponent)
    num *= 1 << exponent
    exponent = -exponent
  significand = math.floor((num - 1) * (1 << nsignificand))
  exponent = (exponent + (1 << nexponent - 1)) & ((1 << nexponent) - 1)
  significand = bintogray(significand)
  exponent = bintogray(exponent)
  return (significand << nexponent) | exponent

# take a floating point number and encode it in our grey scheme
def floattogray(num):
  if num < 0:
    sign = 1 << nsignificand + nexponent - 1
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
  significand = math.floor((num - 1) * (1 << nsignificand - 1))
  exponent = (exponent + (1 << nexponent - 1)) & ((1 << nexponent) - 1)
  significand = bintogray(significand)
  exponent = bintogray(exponent)
  return sign | (significand << nexponent) | exponent

# Turns a gray coded fp number into a native fp number (no negatives)
def graytoabsfloat(gray):
  significand = gray >> nexponent
  exponent = gray & ((1 << nexponent) - 1)
  significand = graytobin(significand,nsignificand)
  exponent = graytobin(exponent,nexponent)
  exponent -= 1 << nexponent - 1
  if exponent >= 0:
    return (1 + significand / (1 << nsignificand)) * (1 << exponent)
  else: 
    return (1 + significand / (1 << nsignificand)) / (1 << -exponent)

# Turns a gray coded fp number into a native fp numberd
def graytofloat(gray):
  significand = gray >> nexponent
  if significand & (1 << nsignificand - 1):
    sign = -1
  else:
    sign = 1
  significand &= (1 << nsignificand - 1) - 1
  exponent = gray & ((1 << nexponent) - 1)
  significand = graytobin(significand,nsignificand)
  exponent = graytobin(exponent,nexponent)
  exponent -= 1 << nexponent - 1
  if exponent >= 0:
    return sign * (1 + significand / (1 << nsignificand - 1)) * (1 << exponent)
  else: 
    return sign * (1 + significand / (1 << nsignificand - 1)) / (1 << -exponent)

##########################################################################################
# HERE BEGINS OLD CODE
##########################################################################################

# class Gene():
#   def __init__(self, g, n):
#     self.g = g
#     self.n = n
#   def __repr__(self):
#     r = ''
#     g = self.g
#     n = self.n
#     while g != 0:
#       if g % 2 == 1:
#         r += '1'
#       else:
#         r += '0'
#       g //= 2
#       n -= 1
#     while n > 0:
#       r = '0'+r
#       n -= 1
#     r = '('+r+','+str(self.n)+')'
#     return r
#   def phenotype(self):
#     return graytobin(self.g,self.n) / (1 << self.n)
#   def fitness(self):
#     animal = self.phenotype()
#     fit = .5 - animal * animal
#     return math.exp(abs(fit))-1
#   def psurvival(self):
#     fitness = self.fitness()
#     return 1 / (fitness + 1)
#   def breed(self, mate):
#     selfg = graytobin(self.g,self.n)
#     mateg = graytobin(mate.g,mate.n)
#     if self.n <= mate.n:
#       if self.n == 0:
#         mask = 0
#       else:
#         mask = 1 << self.n - 1
#       newn = self.n
#       newg = 0
#       while mask != 0:
#         if selfg & mask == 0:
#           selfbit = 0
#         else:
#           selfbit = 1
#         if mateg & mask == 0:
#           matebit = 0
#         else:
#           matebit = 1
#         if random.randint(0,1) == 0:
#           newg = (newg << 1) | selfbit
#         else:
#           newg = (newg << 1) | matebit
#         mask >>= 1
#       newg = (newg << (mate.n - self.n)) | (mate.g & (1 << mate.n - self.n))
#       child = Gene(bintogray(newg), newn)
#       child.mutate()
#       return child
#     else:
#       return mate.breed(self)
#   def randomflip(self):
#     if self.n == 0:
#       return self 
#     randloc = random.randint(0,self.n-1)
#     mask = 1 << randloc
#     gnew = self.g ^ mask
#     self.g = gnew
#     return self
#   def randominsert(self):
#     randspace = random.randint(0,self.n)
#     insertdigit = random.randint(0,1)
#     glow = self.g & ((1 << randspace) - 1)
#     ghigh = self.g >> randspace
#     ghigh <<= randspace + 1
#     insertdigit <<= randspace
#     gnew = insertdigit | ghigh | glow
#     self.g = gnew
#     self.n += 1
#     return self
#   def randomdelete(self):
#     if self.n == 1:
#       return self
#     randloc = random.randint(0,self.n-1)
#     ghigh = self.g >> randloc + 1
#     glow = self.g & ((1 << randloc) - 1)
#     ghigh <<= randloc
#     gnew = glow | ghigh
#     self.g = gnew
#     self.n -= 1
#     return self
#   def mutate(self):
#     if random.randint(0,3) != 0:
#       mutationtype = random.randint(0,2)
#       if mutationtype == 0:
#         self.randomflip()
#       elif mutationtype == 1:
#         self.randominsert()
#       else:
#         self.randomdelete()
#       self.mutate()

# def randomgene():
#   n = 4
#   g = random.getrandbits(n)
#   return Gene(g,n)

# def evolve(popsize, ngenerations, maxsurvivalchance, minsurvivalchance):
#   population = [randomgene() for x in range (0,popsize)]
#   lastbest = 0
#   for k in range(0,ngenerations):
#     population = sorted(population, key=lambda x:x.fitness())
#     survivalchances = [x.psurvival() for x in population]
#     if lastbest != population[0]:
#       print(k, (lambda x:x*x)(population[0].phenotype()),population[0])
#       print(survivalchances[0])
#       lastbest = population[0]
#     elif k % (ngenerations >> 5) == 0:
#       print(k)
#     survivalchances = [minsurvivalchance + (maxsurvivalchance - minsurvivalchance) * x / (survivalchances[0] - survivalchances[len(survivalchances)-1]) for x in survivalchances]
#     j = 0
#     n = len(population)
#     for i in range(0, popsize):
#       r = random.random()
#       if r < survivalchances[i] or n <= 2:
#         j += 1
#       else:
#         del population[j]
#         n -= 1
#     newpopulation = [0] * popsize
#     for i in range(1, popsize):
#       mother = math.floor((lambda x:x*x)(random.random())*(n-.001))
#       father = random.randint(0, n-2)
#       if father >= mother:
#         father += 1
#       mother = population[mother]
#       father = population[father]
#       newpopulation[i] = mother.breed(mother)
#     newpopulation[0] = population[0]
#     population = newpopulation

# evolve(500,100000,1,0)dd