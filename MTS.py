import random
import math
import treelib
import numpy

import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import copy

##TODO? CTP MCTS work only with int as node name, fix?

###############
### UTILITY ###
###############

ONGOING = -1
WIN     = -2
LOSS    = -3
DRAW    = -4

#
class nodeData():
	def __init__(self,win=0,tries=0,score=0):
			self.win=win
			self.tries=tries
			self.value=-1
			self.score=0
			if tries >0:
				self.scoreAverage = score/tries
			else :
				self.scoreAverage = 0


class beliefState():
	def __init__(self,graph,openEdge=[],closedEdge=[]):
			self.graph=graph
			self.unknownEdge = list(graph.edges())
			self.openEdge    = openEdge
			self.closedEdge  = closedEdge

	def look(self,node,weather):		
		self.openEdge += weather.edges(node)
		self.closedEdge += list( set(self.graph.edges(node)) - set(weather.edges(node)))
		self.unknownEdge = list(set(self.unknownEdge) - set(self.graph.edges(node)))


#check if a list is contained within another one
def check(subset,set1):
	return set(subset).issubset(set(set1))


def optimistic(beliefState,start,finish):
	g=copy.deepcopy(beliefState.graph)
	g.remove_edges_from(beliefState.closedEdge)
	path= nx.shortest_path(g,start,finish,"weight")
	weight=0

	for i in range(len(path)-1):
		weight+=g[path[i]][path[i+1]]['weight']
	return weight

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

'''
class canadianTravellerClassic(game):
	def __init__(self, graph,startNode,endNode, NbRouteFermee=1):
		self.startstate=startNode
		self.graph=graph
		self.score = 0
		self.goal=endNode

	#return a list of possible moves from a state
	def moves(self,state):
		return self.graph.edges(state)

	#play a move and return the new state of the game
	def play(self,state,move):
		self.score +=move.weight()

	#evaluate the position
	def evaluate(self,state):
		if state==self.goal:
			return 1
		else :
			return 0
	#return the current score
	def getScore(self):
		return self.score
'''

# stochastic CTP problem as a game
class canadianTravellerStochastic(game):
	def __init__(self, graph,startNode,endNode,startstate=None):
		
		#by default the starting state is just the startNode
		if startstate == None :
			startstate= startNode
		#the state is represented by the name of each node visited in order
		self.state=startstate
		self.startstate=startstate
		self.graph=graph 
		self.goal=endNode

	#return a list of possible moves from a state
	def moves(self,state,weather=None):
		#if we have no weather we work on the graph, otherwise we use the weather
		if weather == None:
			return self.graph.edges(int(state[-1]))
		else :
			return list(weather.edges(int(state[-1])))

	#play a move and return the new state of the game
	def play(self,state,move):
		u,v=move
		return state+str(v)

	#evaluate the position
	def evaluate(self,state):
		#if we are at the goal we calculate the score
		if int(state[-1])==self.goal:
			score=0
			for i in range(len(state)-1):
				score+=self.graph.get_edge_data(int(state[i]),int(state[i+1]))['weight']
			return score
		else :
			return ONGOING

##########################
### THE 4 STEPS OF MTS ###
##########################

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

####REVISER definition expansion
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
def getWeather(graph,source,target) :
	
	validated = False
	
	while(not validated):		
		weather = nx.Graph()

		#a weather get the same node as the starting graph
		for node in graph.nodes():
			weather.add_node(node)

		#then we add the edges if they pass a diceroll using their weight 
		for edge in graph.edges():
			u,v = edge
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
				if (weather.has_edge(int(n.identifier[-2]),int(n.identifier[-1]))):

					#init of subtree
					if subtree == None :
						subtree=n
					
					if (n.data.tries > 0): 
						x=weather.get_edge_data(int(n.identifier[-2]),int(n.identifier[-1]))['weight']
						n.data.value=  float( b * ( math.sqrt( math.log( totalTries ) / n.data.tries ) ) ) - x - float( n.data.scoreAverage )
					else :
						n.data.value= math.inf
					
					if (n.data.value>= subtree.data.value):
						subtree=n
		return SelectionCTPblind(tree,subtree.identifier,totalTries,weather,b)

#
def ExpansionCTPBlind(tree,root,game,state,weather):
	if game.evaluate(state) == ONGOING :
		for n in game.moves(state) :
			u,v=n
			tree.create_node(tag=None,identifier=  str(root)+str(v)  ,parent=root,data=nodeData(tries=0,score=0))		
		#TODO fix? filtre possiblement plus lourd que neccesaire
		return random.choice([x for x in tree.children(root) if weather.has_edge(int(x.identifier[-2]),int(x.identifier[-1]))])
	else:
		return tree.get_node(root)

#
def SimulationCTP(game,state,weather):
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
def SelectionCTPOptimistic(tree,root,totalTries,weather,b,belief_State,goal,optimisticWeight=20):
	beliefState2=copy.deepcopy(belief_State)

	nodes=tree.children(root)
	if (nodes == []):
		return root
	else:
		subtree=None
		for n in nodes:
				if (weather.has_edge(int(n.identifier[-2]),int(n.identifier[-1]))):

					#init of subtree
					if subtree == None :
						subtree=n
					
					optimisticTime=optimistic(beliefState2,int(n.identifier[-1]),goal)
					x=weather.get_edge_data(int(n.identifier[-2]),int(n.identifier[-1]))['weight']
					
					if (n.data.tries > 0): 

						n.data.value=  float( b * ( math.sqrt( math.log( totalTries+optimisticWeight ) / n.data.tries+optimisticWeight ) ) ) - x - float( n.data.scoreAverage )
					
					else :
						n.data.value=  float( ( math.sqrt( math.log( totalTries+optimisticWeight ) / optimisticWeight ) ) ) - x - float( n.data.scoreAverage )
					
					if (n.data.value>= subtree.data.value):
						subtree=n
		beliefState2.look(int(subtree.identifier[-1]),weather)
		return SelectionCTPOptimistic(tree,subtree.identifier,totalTries,weather,b,beliefState2,goal,optimisticWeight)

#
def ExpansionCTPOptimistic(tree,root,game,state,weather,belief_State):
	if game.evaluate(state) == ONGOING :
		beliefState2=copy.deepcopy(belief_State)

		x=None
		xOptimistic=0
		for n in game.moves(state) :
			u,v=n
			node=tree.create_node(tag=None,identifier=  str(root)+str(v)  ,parent=root,data=nodeData(tries=0,score=0))
			beliefState2.look(v,weather)
			if x==None and weather.has_edge(u,v) :
				x=node
				xOptimistic= optimistic(beliefState2,v,game.goal)
			elif weather.has_edge(u,v):
				newOptimistic=optimistic(beliefState2,v,game.goal)
				if xOptimistic > newOptimistic:
					x=node
					xOptimistic=newOptimistic

		return x
	else:
		return tree.get_node(root)




#####################
### MCT Functions ###
#####################

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
def MontreCarloTreeSearchCTP(game,state,iterationMax=10000):
	
	bestMove=""
	iteration=0
	
	tree=treelib.Tree()
	root=game.startstate
	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	while ( iteration <= iterationMax ):
		weather=getWeather(game.graph,game.startstate,game.goal)	

		SelectedNode=SelectionCTPblind(tree,root,iteration,weather,tree.get_node(root).data.scoreAverage)

		ExpandedNode=ExpansionCTPBlind(tree,SelectedNode,game,str(state)+str(SelectedNode)[1:],weather)

		result = SimulationCTP(game,state+ExpandedNode.identifier[1:],weather)

		BackPropagationCTP(tree,ExpandedNode,result)
			
		iteration=iteration+1
		
	print(list(map( lambda x : x.identifier, tree.children(root))))
	print(list(map( lambda x : x.data.tries, tree.children(root))))
	print(list(map( lambda x : x.data.scoreAverage, tree.children(root))))
	
	return bestMove

#
def MontreCarloTreeSearchCTPOptimistic(game,state,iterationMax=10000,optimisticWeight=20):
	
	bestMove=""
	iteration=0
	
	tree=treelib.Tree()
	root=game.startstate
	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	

	while ( iteration <= iterationMax ):
		belief_State=beliefState(game.graph,[],[])
		weather=getWeather(game.graph,game.startstate,game.goal)	
		
		for i in str(game.startstate):
			belief_State.look(int(i),weather)		

		b=((tree.get_node(root).data.scoreAverage*tree.get_node(root).data.tries)+(optimistic(belief_State, int(str(game.startstate)[-1]), game.goal)*optimisticWeight))/(tree.get_node(root).data.tries+optimisticWeight)

		SelectedNode=SelectionCTPOptimistic(tree,root,iteration,weather,b,belief_State,game.goal,optimisticWeight)

		ExpandedNode=ExpansionCTPOptimistic(tree,SelectedNode,game,str(state)+str(SelectedNode)[1:],weather,belief_State)

		result = SimulationCTP(game,state+ExpandedNode.identifier[1:],weather)

		BackPropagationCTP(tree,ExpandedNode,result)

		iteration=iteration+1
		
	print(list(map( lambda x : x.identifier, tree.children(root))))
	print(list(map( lambda x : x.data.tries, tree.children(root))))
	print(list(map( lambda x : x.data.scoreAverage, tree.children(root))))
	
	return bestMove




##########
###TEST###
##########

#titato= ticTacToe()

#x=MontreCarloTreeSearch(titato,"",iterationMax=5000)
#print(x)

#x=MontreCarloTreeSearch(titato,"9581",iterationMax=5000)
#print(x)


g = nx.Graph()
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)
g.add_node(5)
g.add_node(6)
g.add_node(7)
g.add_node(8)

g.add_edge(1, 2,weight =10,proba=0.5)
g.add_edge(3, 2,weight =5,proba=0.4)
g.add_edge(3, 6,weight =25,proba=0.8)
g.add_edge(4, 7,weight =2,proba=0.3)
g.add_edge(3, 8,weight =15,proba=0.5)
g.add_edge(4, 1,weight =12,proba=0.5)
g.add_edge(7, 6,weight =16,proba=0.7)
g.add_edge(2, 5,weight =8,proba=0.8)
g.add_edge(5, 4,weight =1,proba=0.9)
g.add_edge(4, 3,weight =9,proba=0.5)
g.add_edge(4, 8,weight =12,proba=0.7)
g.add_edge(7, 8,weight =4,proba=0.4)
g.add_edge(1, 8,weight =5,proba=0.99)
g.add_edge(1, 5,weight =6,proba=0.25)


#x22=list(g.edges(1))
#print(x22[0])
#print(g.get_edge_data(x22[0][0],x22[0][1])['weight'])
#print(g[1][2]['weight'])
#print(g.edges())
#print(g.nodes())

#print(nx.shortest_path(g,1,3))
#try:
#	print(nx.shortest_path(g,1,4))
#except nx.exception.NetworkXNoPath:
#	print("Nope no path")

#tree=treelib.Tree()
#tree.create_node(tag=None,identifier="1",parent=None,data=nodeData(0,0))	
#root="1"

CTPgame=canadianTravellerStochastic(g,1,8)

print("Blind : ")
MontreCarloTreeSearchCTP(CTPgame,'1',iterationMax=1000)

print("Optimistic : ")
MontreCarloTreeSearchCTPOptimistic(CTPgame,'1',iterationMax=1000)

#SelectedNode=SelectionCTPblind(tree,root,0,weather,0)

#ExpandedNode=ExpansionCTPBlind(tree,SelectedNode,CTPgame,""+SelectedNode,weather)

#SelectedNode=SelectionCTPblind(tree,root,0,weather,10)

#weather=getWeather(g,1,8)
#bob=beliefState(g)
#bob.look(1,weather)

#optimistic(bob,1,8)

#x=[1,2,3,4,5,6,7,8]
#print(x)
#print(x[1:])
#print(x[:-1])

#pos = nx.spring_layout(g)  # positions for all nodes

# nodes
#nx.draw_networkx_nodes(g, pos, node_size=700)

# edges
#nx.draw_networkx_edges(g, pos, g.edges, width=6)

# labels
#nx.draw_networkx_labels(g, pos, font_size=20, font_family="sans-serif")

#plt.axis("off")
#plt.show()