# Evolves a graph.
# Copyright 2017 Gabriel Hesch and Jonathan Baca

import akhab
from ahkab import circuit, printing, time_functions
import math
import random
from enum import Enum

nsignificand               = 12    # Precision of floating point numbers
nexponent                  = 8     # Range of floating point numbers
chanceOfActivationMutation = 0.085 # Chance of gene activation
chanceOfRealGrayMutation   = 0.12  # For mutating fp numbers
chanceOfComponentMutation  = 0.065 # Chance of component type change
chanceOfNewNode            = 0.05  # S.E
chanceOfNewEdge            = 0.04  # S.E
chanceOfDeleteEdge         = 0.01  # S.E
chanceOfNewComponent       = 0.06  # S.E
chanceOfComponentMutation  = 0.08  # S.E

# Multipliers that ensure that a component mutation will give
# us a reasonable circuit.
# TODO:  Add reasonable values 
resistorMultiplier   = 500
capacitorMultiplier  = 1e-6
inductorMultiplier   = 1e-7
DCMultiplier         = 1
IMultiplier          = 1
transistorMultiplier = 1

# Component class -- for representing electrical components
class Component(Enum):
  RESISTOR   = 1
  CAPACITOR  = 2
  INDUCTOR   = 3
  DCSOURCE   = 4
  ISOURCE    = 5
  TRANSISTOR = 6
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
    elif component is Component.ISOURCE:
      return IMultiplier
    elif component is Component.transistor:
      return transistor

# randomChance(.2) has a 1 in 5 chance of returning True
def randomChance(chance):
  return random.random() < chance

# Mutates a gray code represented fp number
def mutateRealGray(gray):
  if randomChance(chanceOfRealGrayMutation):
    gray ^= 1 << random.randint(0,nsignificand = nexponent - 1)
  return gray

# Mutates a random component type and returns it
def mutateComponent(component):
  if randomChance(chanceOfRealComponentMutation):
    component = random.choice(list(Component))
  return component

# Represents the genotype for a single electric component
class ElectricGene():
  def __init__(self, componentType, value, activated=True, isgrayvalue=True, unmultiplied=False):
    self.componentType = componentType
    if isgrayvalue:
      self.grayValue = value
    else:
      if unmultiplied:
        multiplier = Component.multiplier(self.componentType)
      else:
        multiplier = 1
      if componentType is Component.DCSOURCE:
        self.grayValue = floattogray(value / multiplier)
      else:
        self.grayValue = absfloattogray(value / multiplier)
    self.activated = activated
  # Mutation function for EA
  def mutate(self):
    self.grayValue = mutateRealGray(self.grayValue)
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

# A Triode
class Triode():
  def __init__(self, a, b, c, value):
    self.a     = a
    self.b     = b
    self.c     = c
    self.value = value

# Represent the world in which ElectricCircuit objects exist
class ElectricWorld():
  def __init__(self):
    self.temporalMarker = 0
    self.edgeDict = {}
    self.triodeDict = {}
  def addTriode(self, a,b,c):
    if a in self.triodeDict:
      if b in self.triodeDict[a]:
        if c in self.triodeDict[a][b]:
          return self.triodeDict[a][b][c]
        else:
          temporalMarker = self.addTemporalMarker
          self.triodeDict[a][b][c] = temporalMarker
      else:
        temporalMarker = self.addTemporalMarker
        self.triodeDict[a][b] = {c: temporalMarker}
    else:
      temporalMarker = self.addTemporalMarker
      self.triodeDict[a] = {b: {c: temporalMarker}}
      return temporalMarker
  def addEdge(self, start, finish):
    if start in self.edgeDict:
      if finish in self.edgeDict[start]:
        return self.edgeDict[start][finish]
      else:
        temporalMarker = self.addTemporalMarker()
        self.edgeDict[start][finish] = TemporalMarker
      else:
        temporalMarker = self.addTemporalMarker()
        self.edgeDict[start]= {finish: TemporalMarker}
    def addTemporalMarker(self):
      self.temporalMarker += 1 
      return self.temporalMarker
      return self.temporalMarker

# Represents a graph with electric edges
  # Create ahkab circuit object from self
  #   setAhkabCircuit():
  # Creates a new edge from start to finish with
  # weight value
  #   addEdge(start, finish, value):
  # Increments the node number
  #   addNode():
  # Removes an edge
  #   deleteEdge(edgeIndex):
  # Mutates the graph in one of five ways
  #   mutate():
  # Creates a new random node, and connects it with two random edges
  #   mutateNewNode():
  # Adds a random connection
  #   mutateNewEdge():
  # Deletes a random edge.
  # PTODO: Add checking functionality for noncyclicity
  #   mutateDeleteEdge():
  # Adds a new component by splitting an edge
  #   mutateNewComponent():
  # Mutate a random component
  #   mutateComponentMutation():
class ElectricCircuit(): 
  def __init__(self, world):
    self.world        = world
    self.edges        = {}
    self.nedges       = 0
    self.triodes      = {}
    self.ntriodes     = 0
    self.nnodes       = 1 # We already have the GND node
  def setAhkabCircuit(self):
    # Index of various components
    iCapacitor  = 1
    iInductor   = 1
    iResistor   = 1
    iDCSource   = 1
    iISource    = 1
    iTransistor = 1
    # Our ahkab circuit object
    # TODO: Meaningful circuit title
    self.circuit = circuit.Circuit(title="CIRCUIT SIMULATION")
    self.circuit.add_model('ekv', 'nmos', dict(TYPE ='n, VTO=.4, KP=10e-6'))
    gnd = self.circuit.get_ground_node()
    # Add each component to the ahkab circuit
    for i in range(0,self.ntriodes):
      currComponent = self.triodes[i].value.componentType
      if currComonent is Component.TRANSISTOR:
        pass # TODO Write This 
    for i in range(0,self.nedges):
      if self.edges[i].start == 0:
        startNode = gnd
      else:
        startNode  = 'n'+str(self.edges[i].start)
      if self.edges[i].finish == 0:
        finishNode = gnd
      else:
        finishNode  = 'n'+str(self.edges[i].finish)
      componentValue = self.edges[i].value.getValue()
      if currComponent is Component.DCSOURCE:
        componentName = 'V'+str(iDCSource)
        iDCSource += 1
        self.circuit.add_vsource(componentName, n1=startNode, n2=finishNode, dc_value=componentValue)
      elif currComponent is Component.ISOURCE:
        pass # TODO Write This
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
        addFunc(componentName, n1=startNode, n2=finishNode, value=componentValue)
    return self.circuit
  # Creates a new edge from start to finish with
  # weight value
  def addEdge(self, start, finish, value):
    temporalMarker = self.world.addEdge(start, finish)
    self.edges[temporalMarker] = [Edge(start, finish, value)]
    self.nedges += 1
    return self.nedges - 1
  def addTriode(self, a, b, c, value):
    temporalMarker = self.world.addTriode(a, b, c)
    self.triode[temporalMarker] = Triode(a, b, c, value)
  # Increments the node number
  def addNode(self):
    self.nnodes += 1
    return self.nnodes - 1
  # Removes an edge
  def deleteEdge(self, edgeIndex):
    del self.edges[edgeIndex]
    self.nedges -= 1
  # Mutates the graph in one of five ways
  # TODO  Rewrite mutaion function and subfunctions to USE THE EDGECLASS!!  
  def mutate(self):
    if randomChance(chanceOfNewNode):
      self.mutateNewNode()
    if randomCHance(chanceOfNewNode):
      self.mutateNewEdge()
    if randomChnace(chanceOfDeleteEdge):
      self.mutatedeleteEdge()
    if randomChance(chanceOfNewComponent):
      self.mutateNewComponent()
    if randomChance(chanceOfComponentMutation):
      self.mutateComponentMutation()
  # Creates a new random node, and connects it with two random edges
  def mutateNewNode(self):
    newNodeIndex = self.addNode()
    if newNodeIndex < 2:
      return
    aNodeIndex = random.randint(0,newNodeIndex - 1)
    bNodeIndex = random.randint(0,newNodeIndex - 2)
    if bNodeIndex >= aNodeIndex: 
      bNodeIndex += 1
    self.addEdge(aNodeIndex, newNodeIndex, randomElectricGene())
    self.addEdge(bNodeIndex, newNodeIndex, randomElectricGene())
  # Adds a random connection
  def mutateNewEdge(self):
    startNodeIndex = random.randint(0,self.nnodes-1)
    endNodeIndex = random.randint(0,self.nnodes-2)
    if endNodeIndex >= startNodeIndex:
      endNodeIndex += 1
    self.addEdge(startNodeIndex, endNodeIndex, randomElectricGene())
  # Deletes a random edge.
  # PTODO: Add checking functionality for noncyclicity
  def mutateDeleteEdge(self):
    doomedEdgeIndex = random.randint(0,self.nedges-1)
    self.deleteEdge(doomedEdgeIndex)
  # Adds a new component by splitting an edge
  def mutateNewComponent(self):
    splitEdgeIndex = random.int(0,self.nedged-1)
    while not edges[splitEdgeIndex].value.activated:
      splitEdgeIndex = random.int(0,self.nedged-1)
    edged[splitEdgeIndex].value.deactivate()
    isbefore = randomChance(0.5)
    newNodeIndex = self.addNode()
    if isBefore:
      addEdge(edges[splitEdgeIndex].start,newNodeIndex, edges[splitEdgeIndex].value)
      self.addEdge(edges[splitEdgeIndex].end, newNodeIndex, randomElectricGene())
    else:
      addEdge(edges[splitEdgeIndex].start,newNodeIndex, randomElectricGene())
      self.addEdge(edges[splitEdgeIndex].end, newNodeIndex, [splitEdgeIndex].value)
  # Mutate a random component
  def mutateComponentMutation(self):
    mutatedComponentIndex = random.randint(0, self.nedges-1)
    edges[mutatedComponentIndex].value.mutate() 
    # Breed with another ElectricCircuit object

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

# take a positive floating point number and encode it in our gray scheme
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

# take a floating point number and encode it in our gray scheme
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