import random
import math
import treelib
import numpy
import copy
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import delaunay

##TODO? change beliefState to not include graph?

##TODO: change classic MCTS to single expansion

###############
### UTILITY ###
###############

ONGOING = -1
WIN     = -2
LOSS    = -3
DRAW    = -4

#the content of the tree nodes
class nodeData():
	def __init__(self,win=0,tries=0,score=0):
			
			#number of tries
			self.tries=tries

			#for games with wins or losses
			self.win=win
			#for games with a score
			self.score=0
			
			#used for MCTS
			self.value=-1
			if tries >0:
				self.scoreAverage = score/tries
			else :
				self.scoreAverage = 0


#the belief state is a way to represent the knowledge of a weather in a graph
class beliefState():

	def __init__(self,graph,openEdge=[],closedEdge=[]):
		self.graph=graph
		self.unknownEdge = list(graph.edges())
		self.openEdge    = openEdge
		self.closedEdge  = closedEdge

	#update the beliefstate on a given weather
	def look(self,node,weather):		
		self.openEdge += weather.edges(node)
		self.closedEdge += list( set(self.graph.edges(node)) - set(weather.edges(node)))
		self.unknownEdge = list(set(self.unknownEdge) - set(self.graph.edges(node)))

	#show the beliefState
	def show(self):
		print("# Belief State :")
		print("? : ",self.unknownEdge)
		print("V : ",self.openEdge)
		print("X : ",self.closedEdge)
		print("###########")


#check if a list is contained within another one
def check(subset,set1):
	return set(subset).issubset(set(set1))

#get the shortest path and weight of said shortest path from "inital" node to all nodes
def dijsktra(beliefState, initial):

	# weight : the weight of the shortest path to i
	pathWeight = {initial: 0}
	# path : the path to get to i
	path = {initial: [initial]}

	nodes = set(beliefState.graph.nodes)

	while nodes: 

		min_node = None
		
		#we take the visited node with the smallest weight (not very important when we calculate the shortest path to all nodes)
		for node in nodes:
			if node in pathWeight:
				if min_node is None:
					min_node = node			
				elif pathWeight[node] < pathWeight[min_node]:
					min_node = node

		if min_node is None:
			break

		#we remove our current node (in order not not have to calculate it again)
		nodes.remove(min_node)
		
		current_weight = pathWeight[min_node]
		current_path = path[min_node]

		for edge in beliefState.graph.edges(min_node):

			# weather check	
			if not (edge in beliefState.closedEdge) :

				#ensure we get the node we are moving to and not the node we are coming from
				if min(edge) == min_node:
					edge2=max(edge)
				else :
					edge2=min(edge)

				#get the cost of moving to the new node
				addWeight=beliefState.graph[min(edge)][max(edge)]['weight']

				weight = current_weight + addWeight
			
				# if we have no current path to that node or if we found a better path, we write that path
				if edge2 not in pathWeight or weight < pathWeight[edge2]:
					pathWeight[edge2] = weight
					path[edge2] =  path[min_node] + [edge2]
	
	return pathWeight, path

#modified for speed, used for the shortest path in known belief state
def dijsktra2(beliefState, initial):

	# weight : the weight of the shortest path to i
	pathWeight = {initial: 0}
	# path : the path to get to i
	path = {initial: [initial]}

	#nodes = set(beliefState.graph.nodes)
	nodes = set( [x[0] for x in beliefState.openEdge]) | set( [x[1] for x in beliefState.openEdge])

	while nodes: 

		min_node = None
		
		#we take the visited node with the smallest weight (not very important when we calculate the shortest path to all nodes)
		for node in nodes:
			if node in pathWeight:
				if min_node is None:
					min_node = node			
				elif pathWeight[node] < pathWeight[min_node]:
					min_node = node

		if min_node is None:
			break

		#we remove our current node (in order not not have to calculate it again)
		nodes.remove(min_node)
		
		current_weight = pathWeight[min_node]
		current_path = path[min_node]

		for edge in beliefState.graph.edges(min_node):

			# weather check	
			if (edge in beliefState.openEdge) :

				#ensure we get the node we are moving to and not the node we are coming from
				if min(edge) == min_node:
					edge2=max(edge)
				else :
					edge2=min(edge)

				#get the cost of moving to the new node
				addWeight=beliefState.graph[min(edge)][max(edge)]['weight']

				weight = current_weight + addWeight
			
				# if we have no current path to that node or if we found a better path, we write that path
				if edge2 not in pathWeight or weight < pathWeight[edge2]:
					pathWeight[edge2] = weight
					path[edge2] =  path[min_node] + [edge2]
	
	return pathWeight, path

#old version of optimistic
def optimistic_old(beliefState,start,finish):

	g=copy.deepcopy(beliefState.graph)
	g.remove_edges_from(beliefState.closedEdge)

	path= nx.shortest_path(g,start,finish,"weight")

	weight=0
	for i in range(len(path)-1):
		weight+=g[path[i]][path[i+1]]['weight']

	return weight

#get the pertinents succsessors
def getDestinations(beliefState,rootID,finish):

	dest=[]
	eligibleNode=list(beliefState.graph.nodes())

	for i in rootID:
		if i in eligibleNode:
			eligibleNode.remove(i)

	for (i,j) in beliefState.openEdge :
		if (i in eligibleNode) and not (i in dest) :
			dest.append(i)
		if (j in eligibleNode) and not (j in dest):
			dest.append(j)

	return dest

####################
### GAME & GAMES ###
####################

#generic game class
class game():
	def __init__(self):
		self.startstate=""
	
	#return a list of possible moves from a state
	def moves(self,state):
		pass

	#play a move and return the new state of the game
	def play(self,state,move):
		pass

	#evaluate the position
	def evaluate(self,state):
		pass


#jeu du tictactoe(aka morpion)
class ticTacToe(game):
	def __init__(self):
		self.startstate=""

	#return a list of possible moves from a state
	def moves(self,state):
		return [x for x in ["1","2","3","4","5","6","7","8","9"] if x not in [ char for char in state] ]

	#play a move and return the new state of the game
	def play(self,state,move):
		return state+move

	#evaluate the position
	def evaluate(self,state):
		crosses =[cross for cross in state[::2]]
		circles =[circle for circle in state[1::2]]
		if   ( check(["1","2","3"],crosses) or check(["4","5","6"],crosses) or check(["7","8","9"],crosses) or check(["1","4","7"],crosses) or check(["2","5","8"],crosses) or check(["3","6","9"], crosses) or check(["1","5","9"],crosses) or check(["7","5","3"],crosses) ) :
			return WIN
		elif ( check(["1","2","3"],circles) or check(["4","5","6"],circles) or check(["7","8","9"],circles) or check(["1","4","7"],circles) or check(["2","5","8"],circles) or check(["3","6","9"], circles) or check(["1","5","9"],circles) or check(["7","5","3"],circles) ) :
			return LOSS
		elif len(state) == 9:
			return DRAW
		else :
			return ONGOING


# stochastic CTP problem as a game
class canadianTravellerStochastic(game):
	def __init__(self, graph,startNode,endNode,startstate=None):
		
		#by default the starting state is just the startNode
		if startstate == None :
			startstate= [startNode]
		#the state is represented by the name of each node visited in order
		self.state=startstate
		self.startstate=startstate
		self.graph=graph
		self.goal=endNode

	#return a list of possible moves from a state
	def moves(self,state,weather=None):
		#if we have no weather we work on the graph, otherwise we use the weather
		if weather == None:
			return self.graph.edges(state[-1])
		else :
			return list(weather.edges(state[-1]))

	#play a move and return the new state of the game
	def play(self,state,move):

		u,v=move
		state.append(v)

		return state

	#evaluate the position
	def evaluate(self,state):
		#if we are at the goal we calculate the score
		if state[-1]==self.goal:

			score=0

			for i in range(len(state)-1):
				score+=self.graph.get_edge_data(state[i],state[i+1])['weight']

			return score
		else :
			return ONGOING

	#evaluate the current score
	def score(self,state):

		score=0

		for i in range(len(state)-1):
			score+=self.graph.get_edge_data(state[i],state[i+1])['weight']

		return score


###########################
### THE 4 STEPS OF MCTS ###
###########################

def Selection(tree,root,totalTries,c=math.sqrt(2)):	
	nodes=tree.children(root)
	if (nodes == []):
		return root
	else:
		subtree=nodes[0]
		for n in nodes:		
			if (n.data.tries > 0): 
				n.data.value= float(n.data.win/n.data.tries)+float(c+math.sqrt(math.log(totalTries)/n.data.tries))
			else :
				n.data.value= math.inf
			
			if (n.data.value>= subtree.data.value):
				subtree=n
		return Selection(tree,subtree.identifier,totalTries,c)

##TODO: change to single expansion
def Expansion(tree,root,game,state):
	if game.evaluate(state) == ONGOING :
		for n in game.moves(state) :
			tree.create_node(tag=None,identifier=  root+n  ,parent=root,data=nodeData(0,0))		
		return random.choice(tree.children(root))
	else:
		return tree.get_node(root)

def Simulation(game,state):
	while (game.evaluate(state) == ONGOING ) :
		state = game.play(state,random.choice(game.moves(state)))
	return game.evaluate(state)

def BackPropagation(tree,root,result):
	currentNode =root
	while True:
		if   (result == WIN and (tree.depth(currentNode)%2)==1 ) :
			currentNode.data.win+=1
		elif (result == LOSS and (tree.depth(currentNode)%2)==0 ) :
			currentNode.data.win+=1
		else :
			currentNode.data.win+=0.5
		currentNode.data.tries+=1
		currentNode=tree.parent(currentNode.identifier)
		if currentNode == None:
			break

##################################
### THE 5 STEPS OF MTS for CTP ###
##################################

#find a good weather
def getWeather(graph,source,target,belief_State) :
	
	validated = False
	
	while(not validated):
		weather = nx.Graph()

		#a weather get the same node as the starting graph
		for node in graph.nodes():
			weather.add_node(node)
		#then we add the edges if they pass a diceroll using their weight and they are not fixed by the belief state
		#if they are we simply add them as is 
		for edge in graph.edges():
			u,v = edge
			
			if   (u,v) in belief_State.openEdge:
				weather.add_edge(u, v,weight = graph[u][v]['weight'])
			elif (v,u) in belief_State.openEdge:
				weather.add_edge(v, u,weight = graph[v][u]['weight'])
			elif (u,v) in belief_State.unknownEdge:
				#print("in",u,v)
				proba=graph[u][v]['proba']
				x=random.uniform(0,1)
				if proba < x:
					weather.add_edge(u, v,weight = graph[u][v]['weight'])		
		
		#we verify if the weather allow a path from the start node to the goal node
		try:
			nx.shortest_path(weather,source,target)
			validated = True
		except nx.exception.NetworkXNoPath:
			validated = False
	return weather

#
def SelectionCTPblind(tree,root,totalTries,weather,b):
	
	nodes=tree.children(root)
	if (nodes == []):
		return root
	else:
		subtree=None
		for n in nodes:
				ident=n.identifier.split(" ")	
				if (weather.has_edge(ident[-2],ident[-1])):
					#init of subtree
					if subtree == None :
						subtree=n
					
					if (n.data.tries > 0): 
						x=weather.get_edge_data(ident[-2],ident[-1])['weight']
						n.data.value=  float( b * ( math.sqrt( math.log( totalTries ) / n.data.tries ) ) ) - float( n.data.scoreAverage )
					else :
						n.data.value= math.inf
					
					if (n.data.value>= subtree.data.value):
						subtree=n
		return SelectionCTPblind(tree,subtree.identifier,totalTries,weather,b)

#
def ExpansionCTPBlind(tree,root,game,state,weather):
	
	##TODO? modify to add only one node? (need modify selection too)
	if game.evaluate(state) == ONGOING :
		for n in game.moves(state) :
			u,v=n
			tree.create_node(tag=None,identifier=  str(root)+" " +str(v)  ,parent=root,data=nodeData(tries=0,score=0))

		choices=[]
		for x in tree.children(root):
			x2=x.identifier.split(" ")
			if weather.has_edge(x2[-2],x2[-1]):
				choices.append(x)

		return random.choice(choices)
	else:
		return tree.get_node(root)

#
def SimulationCTP(game,state,weather,belief_State):
	while (game.evaluate(state) == ONGOING ) :
		state = game.play(state,random.choice(game.moves(state,weather)))
	return game.evaluate(state)

#
def BackPropagationCTP(tree,root,score):
	currentNode =root
	while True:
		if currentNode.data.tries==0:
			currentNode.data.scoreAverage=score
		else:
			currentNode.data.scoreAverage= ((currentNode.data.scoreAverage*currentNode.data.tries)+score)/(currentNode.data.tries+1)

		currentNode.data.tries+=1
		currentNode=tree.parent(currentNode.identifier)

		if currentNode == None:
			break


##########################
### OPTIMISTIC VARIANT ###
##########################

#
def SelectionCTPOptimistic(tree,root,totalTries,weather,b,belief_State,goal,currentScore=0,optimisticWeight=20):

	nodes=tree.children(root)

	if (nodes == []):
		return root
	else:
		subtree=None
		for n in nodes:
				id=n.identifier.split(" ")
				if (weather.has_edge(id[-2],id[-1])):

					#init of subtree
					if subtree == None :
						subtree=n
					
					##time sink
					optimisticTime= currentScore+ optimistic(belief_State,id[-1],goal)
					
					x=weather.get_edge_data(id[-2],id[-1])['weight']
					
					if (n.data.tries > 0): 
						averageAndOptimistic= float(float((n.data.scoreAverage * n.data.tries) + float(optimisticTime *optimisticWeight) )/(n.data.tries+optimisticWeight))
						exploration = ( math.sqrt( math.log( totalTries+optimisticWeight ) / (n.data.tries+optimisticWeight) ) )
						n.data.value=  float( b * exploration )  - x - averageAndOptimistic
					else :
						n.data.value=  float( b* ( math.sqrt( math.log( totalTries+optimisticWeight ) / optimisticWeight ) ) ) - x - optimisticTime
					
					if (n.data.value>= subtree.data.value):
						subtree=n
						scoreAdd=x

		subtreeId=subtree.identifier.split(" ")
		belief_State.look(subtreeId[-1],weather)

		return SelectionCTPOptimistic(tree,subtree.identifier,totalTries,weather,b,belief_State,goal,currentScore+scoreAdd ,optimisticWeight)

#
def ExpansionCTPOptimistic(tree,root,game,state,weather,belief_State):
	
	if game.evaluate(state) == ONGOING :
		
		beliefState2=copy.deepcopy(belief_State)
		x=None
		xOptimistic=0

		for n in game.moves(state) :
	
			u,v=n
			
			node=tree.create_node(tag=None,identifier=  str(root)+" "+str(v)  ,parent=root,data=nodeData(tries=0,score=0))
			
			beliefState2.look(v,weather)
			if x==None and weather.has_edge(u,v) :
				x=node
				xOptimistic= optimistic(beliefState2,v,game.goal)
			elif weather.has_edge(u,v):
				newOptimistic= optimistic(beliefState2,v,game.goal)
	
				if xOptimistic > newOptimistic:
					x=node
					xOptimistic=newOptimistic

		return x
	else:
		return tree.get_node(root)


##########################
### BLIND DEEP VARIANT ###
##########################

#
def SelectionCTPblindDeep(tree,root,game,totalTries,weather,b):

	rootID=root.split(" ")

	score=game.evaluate(rootID)
	while score == ONGOING :
		
		nodesID=[ tuple(x.identifier.split(" ")) for x in tree.children(root)]
		nodesEndID=[ tuple([nodeId[-2],nodeId[-1]]) for nodeId in nodesID]

		subtree=None
		candidates=[]
								
		for n in game.moves(rootID,weather):

			if n in nodesEndID :
				
				node=tree.get_node(root +" "+str(n[-1]))

				if subtree == None :
					subtree= node

				node.data.value=  float( b * ( math.sqrt( math.log( totalTries ) / node.data.tries ) ) ) - float( node.data.scoreAverage )
				if (node.data.value>= subtree.data.value):
					subtree=node
			else :
				candidates += [n]
				
		if candidates == [] :
			root=subtree.identifier

		else :
			x=tree.create_node(tag=None,identifier=  str(root)+" " +str(random.choice(candidates)[-1])  ,parent=root,data=nodeData(tries=0,score=0))
			root=x.identifier
		rootID=root.split(" ")
		score=game.evaluate(rootID)
	else :
		return (score,tree.get_node(root))

#
def SelectionCTPblindDeepSingle(tree,state,game,totalTries,weather,b,belief_State):


	root=state[-1]
	beliefState2=copy.deepcopy(belief_State)
	rootID=root.split(" ")
	selectedMove=None

	score=game.evaluate(rootID)
	while score == ONGOING :
		
		nodesID=[ tuple(x.identifier.split(" ")) for x in tree.children(root)]
		nodesEndID=[ nodeId[-1] for nodeId in nodesID]

		subtree=None
		candidates=[]

		moves = getDestinations(beliefState2,state[:-1]+rootID,game.goal)

		(weight,path)= dijsktra2(beliefState2,rootID[-1])

		for n in moves:						

			candidatePath=path[n]
			formatedPath =(candidatePath[1:])
			strPath=" ".join(formatedPath)
			node=tree.get_node(root +" "+strPath)
		
			if node != None :

				if subtree == None :
					subtree= node				

				node.data.value=  float( b * ( math.sqrt( math.log( totalTries ) / node.data.tries ) ) ) - float( node.data.scoreAverage )
				if (node.data.value >= subtree.data.value):
					subtree=node

			else :
				candidates += [n]
				
		if candidates == [] :
			root=subtree.identifier
			if selectedMove == None :
				selectedMove=subtree.identifier.split(" ")

		else :
			candidatePath=path[random.choice(candidates)]
			formatedPath =(candidatePath[1:])
			parent=root
			for i in formatedPath :
				try:
					x=tree.create_node(tag=None,identifier=  str(parent)+" " +i  ,parent=parent,data=nodeData(tries=0,score=0))
					parent=x.identifier
				except treelib.exceptions.DuplicatedNodeIdError:
					parent=str(parent)+" " +i
			if selectedMove == None :
				selectedMove=candidatePath

			root=parent
		rootID=root.split(" ")
		score=game.evaluate(rootID)
		beliefState2.look(rootID[-1],weather)
	else :
		return (score,tree.get_node(root)," ".join(selectedMove))

################################
### BLIND OPTIMISTIC VARIANT ###
################################

#heavy version
def SelectionCTPOptimisticDeep(tree,root,game,totalTries,weather,b,belief_State,currentScore=0,optimisticWeight=20):

	rootID=root.split(" ")

	score=game.evaluate(rootID)
	
	i=0

	while score == ONGOING :
		i+=1
		nodes=tree.children(root)
		
		nodesID=[ tuple(x.identifier.split(" ")) for x in tree.children(root)]
		nodesEndID=[ tuple([nodeId[-2],nodeId[-1]]) for nodeId in nodesID]

		candidate=None
		subtree=None
		for n in game.moves(rootID,weather):

			moveWeight=weather.get_edge_data(n[-2],n[-1])['weight']
			optimisticTime= currentScore + moveWeight + optimistic_old(belief_State,n[-1],game.goal)

			if n in nodesEndID :

				node=tree.get_node(root +" "+str(n[-1]))

				if subtree == None :
					subtree= node

				exploration = ( math.sqrt( math.log( totalTries+optimisticWeight ) / (node.data.tries+optimisticWeight) ) )
				averageAndOptimistic= float(float(((currentScore+node.data.scoreAverage) * node.data.tries) + float(optimisticTime *optimisticWeight) )/(node.data.tries+optimisticWeight))
				node.data.value=  float( b * exploration )  - (averageAndOptimistic)

				if (node.data.value > subtree.data.value):
					subtree=node
			else :

				value= optimisticTime

				if candidate == None:
					candidate = n
					candidateValue = value

				if value < candidateValue :
					candidate = n
					candidateValue = value
		
		if candidate == None  :
			root=subtree.identifier

		else :
			x=tree.create_node(tag=None,identifier=  str(root)+" " +str((candidate)[-1])  ,parent=root,data=nodeData(tries=0,score=0))
			root=x.identifier

		rootID=root.split(" ")
		score=game.evaluate(rootID)
		currentScore+=moveWeight
		rootID=root.split(" ")
		belief_State.look(rootID[-1],weather)
	else :
		return (score,tree.get_node(root))

#'light' version
def SelectionCTPOptimisticDeepSingle(tree,state,game,totalTries,weather,b,belief_State,currentScore=0,optimisticWeight=20):

	root=state[-1]

	rootID=root.split(" ")
	beliefState2=copy.deepcopy(belief_State)
	
	selectedMove=None
	score=game.evaluate(rootID)
	(opPathWeight, opPath)=dijsktra(beliefState2,game.goal)
	while score == ONGOING :
		nodesID=[ tuple(x.identifier.split(" ")) for x in tree.children(root)]
		nodesEndID=[ tuple([nodeId[-2],nodeId[-1]]) for nodeId in nodesID]

		candidate=None
		subtree=None
	
		moves = getDestinations(beliefState2,state[:-1]+rootID,game.goal)

		(weight,path)= dijsktra2(beliefState2,rootID[-1])

		for n in moves:	
			optimisticTime= currentScore + weight[n] + opPathWeight[n]
	
			candidatePath=path[n]
			formatedPath =(candidatePath[1:])
			strPath=" ".join(formatedPath)
			node=tree.get_node(root +" "+strPath)
		
			if node != None :
	
				if subtree == None :
					subtree= node


				exploration = ( math.sqrt( math.log( totalTries+optimisticWeight ) / (node.data.tries+optimisticWeight) ) )
				averageAndOptimistic= float(float(((currentScore+node.data.scoreAverage) * node.data.tries) + float(optimisticTime *optimisticWeight) )/(node.data.tries+optimisticWeight))
	
				node.data.value=  float( b * exploration )  - (averageAndOptimistic)
	
				if (node.data.value > subtree.data.value):
					subtree=node
			else :
	
				value= optimisticTime
	
				if candidate == None:
					candidate = n
					candidateValue = value

				if value < candidateValue :
					candidate = n
					candidateValue = value									
	
		if candidate == None  :
			root=subtree.identifier
			candidate =subtree.identifier.split(" ")[-1]
			if selectedMove == None :
				selectedMove=subtree.identifier.split(" ")
		else :

			candidatePath=path[candidate]
			formatedPath =(candidatePath[1:])
			parent=root
			for i in formatedPath :
				try:
					x=tree.create_node(tag=None,identifier=  str(parent)+" " +i  ,parent=parent,data=nodeData(tries=0,score=0))
					parent=x.identifier
				except treelib.exceptions.DuplicatedNodeIdError:
					parent=str(parent)+" " +i
			if selectedMove == None :
				selectedMove=candidatePath
			
			root=parent

		rootID=root.split(" ")
		score=game.evaluate(rootID)
		currentScore+=weight[candidate]
		rootID=root.split(" ")
		beliefState2.look(rootID[-1],weather)
	else :
		return (score,tree.get_node(root)," ".join(selectedMove))

#####################
### MCT Functions ###
#####################

#
def MontreCarloTreeSearch(game,state,iterationMax=10000):
	bestMove=""
	iteration=0
	
	tree=treelib.Tree()
	root=game.startstate
	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	

	while ( iteration <= iterationMax ):

		SelectedNode=Selection(tree,root,iteration)

		ExpandedNode=Expansion(tree,SelectedNode,game,state+SelectedNode)

		result = Simulation(game,state+ExpandedNode.identifier)
		
		BackPropagation(tree,ExpandedNode,result)
		
		iteration=iteration+1
	
	print(list(map( lambda x : x.identifier, tree.children(""))))
	print(list(map( lambda x : x.data.tries, tree.children(""))))
	print(list(map( lambda x : x.data.win, tree.children(""))))
	
	bestMove1= numpy.argmax(list(map( lambda x : x.data.tries, tree.children(""))))
	bestMove= tree.children("")[bestMove1].identifier
	return bestMove

#monte carlo tree search for CTP
def MontreCarloTreeSearchCTP(game,graph,state,belief_State,iterationMax=10000):
	
	bestMove=""
	iteration=0
	
	tree=treelib.Tree()

	root = state[-1]

	trimmed=state[:-1]

	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	while ( iteration <= iterationMax ):
		weather=getWeather(graph,root,game.goal,belief_State)	

		SelectedNode=SelectionCTPblind(tree,root,iteration,weather,tree.get_node(root).data.scoreAverage,debug =debug)

		newstate=trimmed + (str(SelectedNode)).split(" ")

		ExpandedNode=ExpansionCTPBlind(tree,SelectedNode,game,newstate,weather)

		newstate=trimmed +(str(ExpandedNode.identifier)).split(" ")

		result = SimulationCTP(game,newstate,weather,belief_State)

		BackPropagationCTP(tree,ExpandedNode,result)

		iteration=iteration+1
	
	return list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root)))

#monte carlo tree search for CTP, no simulation version
def MontreCarloTreeSearchCTPDeepSelection(game,graph,state,belief_State,iterationMax=10000):
	
	iteration=0
	tree=treelib.Tree()
	root = state[-1]
	
	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	moves={}

	while ( iteration <= iterationMax ):
		weather=getWeather(graph,root,game.goal,belief_State)	

		(score,SelectedNode,selected)=SelectionCTPblindDeepSingle(tree,state,game,iteration,weather,tree.get_node(root).data.scoreAverage,belief_State)
		if not selected in moves:
			moves[selected]=1
		else :
			moves[selected]= moves[selected] + 1
		BackPropagationCTP(tree,SelectedNode,score)

		iteration=iteration+1
	newVal=list( zip(moves.keys(),moves.values()) )
	return list(map(lambda x: (tuple(x[0].split(" ")),x[1]), newVal))

#
def MontreCarloTreeSearchCTPOptimistic(game,graph,state,belief_State_source,iterationMax=10000,optimisticWeight=20):

	timeS=time.time()
	
	bestMove=""
	iteration=0

	trimmed=state[:-1]
		
	tree=treelib.Tree()
	
	root=state[-1]

	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	


	b2=(game.score(state)+optimistic(belief_State_source, root, game.goal)*optimisticWeight)


	while ( iteration <= iterationMax ):
		
		belief_State=copy.deepcopy(belief_State_source)
		
		weather=getWeather(graph,root,game.goal,belief_State_source)

		b1=(tree.get_node(root).data.scoreAverage*tree.get_node(root).data.tries)
		
		b3=(tree.get_node(root).data.tries+optimisticWeight)
		b=(b1+b2)/b3

		SelectedNode=SelectionCTPOptimistic(tree,root,iteration,weather,b,belief_State,game.goal,game.score(state),optimisticWeight)
		
		newstate=trimmed + (str(SelectedNode)).split(" ")

		ExpandedNode=ExpansionCTPOptimistic(tree,SelectedNode,game,newstate,weather,belief_State)

		newstate=trimmed + (str(ExpandedNode.identifier)).split(" ")

		result = game.score(state) + SimulationCTP(game,newstate,weather,belief_State)

		BackPropagationCTP(tree,ExpandedNode,result)

		iteration=iteration+1
	
	#print("OPTIMISTIC base :",game.score(state), " " ,    list(map( lambda x : (graph.get_edge_data((x.identifier.split()[-2]),(x.identifier.split()[-1]))['weight']+optimistic(belief_State_source, ((x.identifier.split()[-1])), game.goal) ), tree.children(root) )))
	#print("SCORE AVERAGES: ", list(map( lambda x : (tuple(x.identifier.split()),x.data.tries,x.data.scoreAverage), tree.children(root))))
	return list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root)))

#
def MontreCarloTreeSearchCTPOptimisticDeep(game,graph,state,belief_State_source,iterationMax=10000,optimisticWeight=20):

	iteration=0
	tree=treelib.Tree()
	root = state[-1]
	
	initialScore=game.score(state)

	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	(pathWeight, path)=dijsktra(belief_State_source,game.goal)
	
	b2=(initialScore+pathWeight[root]*optimisticWeight)

	moves={}

	while ( iteration <= iterationMax ):

		belief_State=copy.deepcopy(belief_State_source)

		weather=getWeather(graph,root,game.goal,belief_State)	

		b1=((tree.get_node(root).data.scoreAverage)*tree.get_node(root).data.tries)
		b3=(tree.get_node(root).data.tries+optimisticWeight)

		b=(b1+b2)/b3
													   
		(score,SelectedNode,selected)=SelectionCTPOptimisticDeepSingle(tree,state,game,iteration,weather,b,belief_State,initialScore,optimisticWeight)
		if not selected in moves:
			moves[selected]=1
		else :
			moves[selected]= moves[selected] + 1

		BackPropagationCTP(tree,SelectedNode,score)

		iteration=iteration+1
	#print("###########SCORE AVERAGES: ", list(map( lambda x : (tuple(x.identifier.split()),x.data.tries,x.data.scoreAverage), tree.children(root))))
	#print("########## RESULT :",list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root))),"##########")
	
	newVal=list( zip(moves.keys(),moves.values()) )

	return list(map(lambda x: (tuple(x[0].split(" ")),x[1]), newVal))

##########
###TEST###
##########

def playCTP(graph,start,end,iterationMax=10000):
	
	CTPgameBlind    = canadianTravellerStochastic(graph,start,end)
	CTPgameOptimist = canadianTravellerStochastic(graph,start,end)

	belief_State=beliefState(CTPgameBlind.graph,[],[])

	weather=getWeather(graph,start,end,belief_State)

	belief_State.look(start,weather)
	

	####

	print("playing Blind")
	
	state=CTPgameBlind.startstate
	
	f=open("log3.txt","a+")

	print("Weather : ",file=f)
	print(weather.edges(),file=f)
	f.close()
	
	graph2=copy.deepcopy(graph)


	time1=time.time()
	while (CTPgameBlind.evaluate(state)==ONGOING):
		
		graph2.remove_edges_from(belief_State.closedEdge)		
	
		moves=MontreCarloTreeSearchCTPDeepSelection(CTPgameBlind,graph2,state,belief_State,iterationMax)
	
		move=max(moves,key=lambda item:item[1])[0]
	
		print("playing : ",move,state)

		for i in range(0,len(move)-1):
			j=list(move)[i]
			k=list(move)[i+1]
		
			belief_State.look(k,weather)
			state=CTPgameBlind.play(state,(j,k))

	time2=time.time()
	timeBlind=time2-time1
	print("path : ",state)
	print("time for blind :",timeBlind)
	print("Score : ", CTPgameBlind.evaluate(state))
	scoreblind=CTPgameBlind.evaluate(state)

	#######

	print("playing Optimist")

	state=CTPgameOptimist.startstate

	belief_State=beliefState(CTPgameBlind.graph,[],[])
	belief_State.look(start,weather)

	graph2=copy.deepcopy(graph)

	time1=time.time()
	while (CTPgameOptimist.evaluate(state)==ONGOING):
		
		graph2.remove_edges_from(belief_State.closedEdge)

		moves=MontreCarloTreeSearchCTPOptimisticDeep(CTPgameBlind,graph2,state,belief_State,iterationMax)

		weathermoves=CTPgameOptimist.moves(state,weather)
		
		filtermove=[t for t in moves if t[0] in  weathermoves]
		move=max(moves,key=lambda item:item[1])[0]
		print("playing : ",move,state)

		for i in range(0,len(move)-1):
			j=list(move)[i]
			k=list(move)[i+1]
		
			belief_State.look(k,weather)
			state=CTPgameBlind.play(state,(j,k))
	
	time2=time.time()
	timeOpt=time2-time1
	print("path : ",state)
	print("Time for Optimist : ",timeOpt)
	scoreOpti=CTPgameOptimist.evaluate(state)
	print("Score : ", scoreOpti)

	print("shortest path ")
	bestState=nx.shortest_path(weather,start,end,"weight")
	print(bestState)
	bestScore=CTPgameOptimist.evaluate(bestState)
	print("best score : ", bestScore)

	f=open("logResult.txt","a+")
	print("Blind : ",scoreblind," : ",scoreblind/bestScore,", Opti : ",scoreOpti," : ",scoreOpti/bestScore,", best : ",bestScore,file=f)
	f.close()

	print("shortest path on clear weather ")
	bestState=nx.shortest_path(graph,start,end,"weight")
	print(bestState)
	print("best score : ", CTPgameOptimist.evaluate(bestState))





for i in range(0,100) :

	delau=delaunay.Delaunay(20)
	g=delau.graph

	f=open("log3.txt","w+")

	print("Graph : ",file=f)
	print(g.nodes(),file=f)
	print(g.edges(),file=f)

	print("Info : ",file=f)
	for (u,v) in g.edges():
		print(u,v, " : ",g[u][v],file=f)

	f.close()
	playCTP(g,"1","20",iterationMax=10000)
	print("#################################",i+1,"OUT OF 100","#################################")
