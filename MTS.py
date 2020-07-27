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

##TODO? optimise optimistic performance

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


#
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
		print("###########")
		print("? : ",self.unknownEdge)
		print("V : ",self.openEdge)
		print("X : ",self.closedEdge)
		print("###########")


#check if a list is contained within another one
def check(subset,set1):
	return set(subset).issubset(set(set1))




def dijsktra(beliefState, initial):
	visited = {initial: 0}
	path = {}

	g=copy.deepcopy(beliefState.graph)
	g.remove_edges_from(beliefState.closedEdge)

	nodes = set(g.nodes)
	#print("NODES :",nodes)
	while nodes: 
		#print("NODES :",nodes)
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] < visited[min_node]:
					min_node = node

		if min_node is None:
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]

		for edge in g.edges(min_node):
			
			#print("HHH")
			#print(g.edges(min_node))
			#print(min_node)
			#print(edge)

			if min(edge) == min_node:
				edge2=max(edge)
			else :
				edge2=min(edge)

			addWeight=g[min(edge)][max(edge)]['weight']

			weight = current_weight + addWeight
			
			if edge2 not in visited or weight < visited[edge2]:
				visited[edge2] = weight
			path[edge2] = min_node

	return visited, path



#calculate the cost of the optimistic path on a beliefState
def optimistic(beliefState,finish):

	
	(visited,path) = dijsktra(beliefState,finish)
	#print("Visited :",visited)
	#print("Path :",path)
	#print(visited['1'])

	return visited

def optimistic_old(beliefState,start,finish):

	g=copy.deepcopy(beliefState.graph)
	g.remove_edges_from(beliefState.closedEdge)

	path= nx.shortest_path(g,start,finish,"weight")


	#(visited,path22) = dijsktra(beliefState,finish)


	weight=0
	for i in range(len(path)-1):
		weight+=g[path[i]][path[i+1]]['weight']

	#if int(weight) != int(visited[start]):
	#		print("1,2 :",visited[start],weight)
	#else :
	#	print("##Good##")

	return weight

#calculate the optimistic path on a beliefState
def optimisticMoves(beliefState,start,finish):

	g=copy.deepcopy(beliefState.graph)
	
	g.remove_edges_from(beliefState.closedEdge)

	path= nx.shortest_path(g,start,finish,"weight")

	return path

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
		#print(subtree.identifier)
		return SelectionCTPblind(tree,subtree.identifier,totalTries,weather,b)


####################################################
####################################################
####################################################

#
def SelectionCTPblindDeep(tree,root,game,totalTries,weather,b):

	rootID=root.split(" ")
	#print("Root : ",rootID)

	score=game.evaluate(rootID)
	while score == ONGOING :

		#not used
		#nodes=tree.children(root)
		
		nodesID=[ tuple(x.identifier.split(" ")) for x in tree.children(root)]
		nodesEndID=[ tuple([nodeId[-2],nodeId[-1]]) for nodeId in nodesID]

		#print("NODES  : ",nodesID)

		subtree=None
		candidates=[]
								
		for n in game.moves(rootID,weather):
			
			#print("Node:",n,". Nodes in tree : ",nodesID)
			#print("filtered nodes : ",nodesEndID)


			if n in nodesEndID :
				
				
				node=tree.get_node(root +" "+str(n[-1]))

				if subtree == None :
					subtree= node

				#print("this node exist in the tree : ",root +" "+str(n[-1]) ,node)

				x=weather.get_edge_data(n[-2],n[-1])['weight']
				node.data.value=  float( b * ( math.sqrt( math.log( totalTries ) / node.data.tries ) ) ) - float( node.data.scoreAverage )
				if (node.data.value>= subtree.data.value):
					subtree=node
			else :
				#print("this one don't",n)
				candidates += [n]
				

		#print(subtree.identifier)
		#tree.create_node(tag=None,identifier=  str(root)+" " +str(v)  ,parent=root,data=nodeData(tries=0,score=0))
		
		if candidates == [] :
			root=subtree.identifier

		else :
			x=tree.create_node(tag=None,identifier=  str(root)+" " +str(random.choice(candidates)[-1])  ,parent=root,data=nodeData(tries=0,score=0))
			root=x.identifier
		rootID=root.split(" ")
		score=game.evaluate(rootID)
	else :
		#print("YESSSSSSS")
		return (score,tree.get_node(root))


####################################################
####################################################
####################################################



#
def ExpansionCTPBlind(tree,root,game,state,weather):
	

	####TODO? modify to add only one node? (need modify selection too)
	if game.evaluate(state) == ONGOING :
		#print("ST",state)
		#print(game.moves(state))
		for n in game.moves(state) :
			u,v=n
			#tree.create_node(tag=None,identifier=  str(root)+"."+str(v)  ,parent=root,data=nodeData(tries=0,score=0))
			tree.create_node(tag=None,identifier=  str(root)+" " +str(v)  ,parent=root,data=nodeData(tries=0,score=0))

		#TODO fix? filtre possiblement plus lourd que neccesaire
		choices=[]
		for x in tree.children(root):
			x2=x.identifier.split(" ")
			if weather.has_edge(x2[-2],x2[-1]):
				choices.append(x)
				
		#return random.choice([x for x in tree.children(root) if weather.has_edge(int(x.identifier[-2]),int(x.identifier[-1]))])
		return random.choice(choices)
	else:
		#print(tree.get_node(root).identifier)
		return tree.get_node(root)

#
def SimulationCTP(game,state,weather,belief_State):
	
	#print("SImUL",state)

	#generate the list of pertient moves


	while (game.evaluate(state) == ONGOING ) :
		#print("SImUL1",state,random.choice(game.moves(state,weather)))
		state = game.play(state,random.choice(game.moves(state,weather)))
		#print("SImUL2",state)		
	return game.evaluate(state)

'''
#
def SimulationCTPALT(game,state,weather,belief_State):
	
	#print("SImUL",state)
	path= optimisticMoves(belief_State,state[-1],game.goal)

	#print(path)
	while (game.evaluate(state) == ONGOING ) :
		path=path[1:]
		#print(path)
		#print("SImUL1",state,random.choice(game.moves(state,weather)))
		#print(game.moves(state,weather))
		#print((state[-1],path[0]))
		belief_State.look(state[-1],weather)
		#print(state,path)
		if ((state[-1],path[0]) in game.moves(state,weather)):
			
			state = game.play(state,((state[-1],path[0])))
		else :
			print("ouch")
			path= optimisticMoves(belief_State,state[-1],game.goal)	

		#print("SImUL2",state)		

	#	return lolcrash
	return game.evaluate(state)
'''

#
def BackPropagationCTP(tree,root,score):
	currentNode =root
	#print(score)
	while True:
		if currentNode.data.tries==0:
			#print("newscore :",score)
			currentNode.data.scoreAverage=score
		else:
			
			currentNode.data.scoreAverage= ((currentNode.data.scoreAverage*currentNode.data.tries)+score)/(currentNode.data.tries+1)
			#print("average :",currentNode.data.scoreAverage)
		currentNode.data.tries+=1
		currentNode=tree.parent(currentNode.identifier)
		if currentNode == None:
			break


##########################
### OPTIMISTIC VARIANT ###
##########################

#
#def SelectionCTPOptimistic(tree,root,totalTries,weather,b,belief_State,goal,optimisticWeight=20):
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
					
					##bouffe tout le temps d'execution
					optimisticTime= currentScore+ optimistic(belief_State,id[-1],goal)
					
					x=weather.get_edge_data(id[-2],id[-1])['weight']
					
					if (n.data.tries > 0): 

						averageAndOptimistic= float(float((n.data.scoreAverage * n.data.tries) + float(optimisticTime *optimisticWeight) )/(n.data.tries+optimisticWeight))
	

						exploration = ( math.sqrt( math.log( totalTries+optimisticWeight ) / (n.data.tries+optimisticWeight) ) )


						n.data.value=  float( b * exploration )  - x - averageAndOptimistic

					else :
						


						n.data.value=  float( b* ( math.sqrt( math.log( totalTries+optimisticWeight ) / optimisticWeight ) ) ) - x - optimisticTime
					
					if (n.data.value>= subtree.data.value):
						#exploration = ( math.sqrt( math.log( totalTries+optimisticWeight ) / (n.data.tries+optimisticWeight) ) )
						#print("Choosing :",b,exploration,x,n.data.scoreAverage,id)
						subtree=n
						scoreAdd=x

					#print("I AM ",id[-2],id[-1]," score : ",n.data.value, "x :", x, " average : ",n.data.scoreAverage, " b : ", b )

		#print("##############",currentScore)
		#print("I AM ",int(subtree.identifier.split(" ")[-1]))
		subtreeId=subtree.identifier.split(" ")
		belief_State.look(subtreeId[-1],weather)

		return SelectionCTPOptimistic(tree,subtree.identifier,totalTries,weather,b,belief_State,goal,currentScore+scoreAdd ,optimisticWeight)



#
#def SelectionCTPOptimistic(tree,root,totalTries,weather,b,belief_State,goal,optimisticWeight=20):
def SelectionCTPOptimisticDeep(tree,root,game,totalTries,weather,b,belief_State,currentScore=0,optimisticWeight=20):

	#print("IN",totalTries)
	f=open("log3.txt","a+")

	print("###Selection ",totalTries,"###",file=f)

	rootID=root.split(" ")
	#print("Root : ",rootID)

	score=game.evaluate(rootID)
	
	i=0

	#calculate all optimistics times here
	#op= optimistic(belief_State,game.goal)

	while score == ONGOING :
		print(" ##Selection ",totalTries,", ",i,file=f)
		i+=1
		#not used
		nodes=tree.children(root)
		
		nodesID=[ tuple(x.identifier.split(" ")) for x in tree.children(root)]
		nodesEndID=[ tuple([nodeId[-2],nodeId[-1]]) for nodeId in nodesID]

		#print("NODES  : ",nodesID)

		candidate=None
		subtree=None
		print(" moves : ",game.moves(rootID,weather),file=f)
		for n in game.moves(rootID,weather):
			
			print("  Node:",n,file=f)
			#print("filtered nodes : ",nodesEndID)

			moveWeight=weather.get_edge_data(n[-2],n[-1])['weight']


			#time sink
			#don't do it here
			#print("LULULULULULULLLLLLLLLLLLLLLLLLLLLLLLLLLLLl")
			#print(op)
			#optimisticTime= currentScore + moveWeight + op[n[-1]]
			#print(optimisticTime)
			optimisticTime= currentScore + moveWeight + optimistic_old(belief_State,n[-1],game.goal)
			#print(optimisticTime)
			#if max(optimisticTime,optimisticTime2) - min(optimisticTime,optimisticTime2) > 20:
			#	print("NOPE :",optimisticTime,optimisticTime2)
			#	print(op)
			#	print(n[-1])
			#	print(optimistic_old(belief_State,n[-1],game.goal))
			#	print(op[n[-1]])
			#else :
			#	print("Clean")

			print("  optimistic result : ",optimisticTime,file=f)


			if n in nodesEndID :
				print("   node in tree",file=f)
				

				node=tree.get_node(root +" "+str(n[-1]))
				print("   tries : ",node.data.tries," scoreAverage : ",node.data.scoreAverage,file=f)

				if subtree == None :
					subtree= node


				exploration = ( math.sqrt( math.log( totalTries+optimisticWeight ) / (node.data.tries+optimisticWeight) ) )
				averageAndOptimistic= float(float(((currentScore+node.data.scoreAverage) * node.data.tries) + float(optimisticTime *optimisticWeight) )/(node.data.tries+optimisticWeight))
				print("   exploration : ",exploration," b : ",b, "result : ",b*exploration,file=f)
				print("   exploitation : ",averageAndOptimistic,file=f)

				#print("Exploration :",float( b * exploration ),"  Exploitation : ",(moveWeight + averageAndOptimistic),"i :",i,root)
				#node.data.value=  float( b * ( math.sqrt( math.log( totalTries ) / node.data.tries ) ) ) - float( node.data.scoreAverage )
				node.data.value=  float( b * exploration )  - (averageAndOptimistic)
				print("   value : ",node.data.value,file=f)

				if (node.data.value > subtree.data.value):
					subtree=node
			else :
				#print("NEW ",[str(n[-2]),str(n[-1])],optimisticTime)
				print("   node NOT in tree",file=f)

				value= optimisticTime
				print("   value : ",optimisticTime,file=f)				

				if candidate == None:
					print("    FIRST",[str(n[-2]),str(n[-1])],file=f)
					candidate = n
					candidateValue = value

				if value < candidateValue :
					print("    BETTER THAN ",candidateValue,file=f)
					candidate = n
					candidateValue = value
									

		#print(subtree.identifier)
		#tree.create_node(tag=None,identifier=  str(root)+" " +str(v)  ,parent=root,data=nodeData(tries=0,score=0))
		
		if candidate == None  :
			print(" NO CANDIDATE, ",subtree.identifier," SELECTED.",file=f)
			#print("IN",len(root))
			root=subtree.identifier

		else :
			#print("OUT",len(root),str((candidate)[-1]),optimisticTime)
			x=tree.create_node(tag=None,identifier=  str(root)+" " +str((candidate)[-1])  ,parent=root,data=nodeData(tries=0,score=0))
			print(" CANDIDATE, ",x.identifier," SELECTED.",file=f)
			root=x.identifier


		rootID=root.split(" ")
		score=game.evaluate(rootID)
		currentScore+=moveWeight
		rootID=root.split(" ")
		belief_State.look(rootID[-1],weather)
	else :
		#print("YESSSSSSS")

		f.close()
		return (score,tree.get_node(root))


	







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
def MontreCarloTreeSearchCTP(game,graph,state,belief_State,iterationMax=10000):
	
	bestMove=""
	iteration=0
	
	tree=treelib.Tree()


	#root = " ".join(str(x) for x in game.startstate)
	root = state[-1]
	#print("R", root)

	trimmed=state[:-1]


	#print("T",trimmed)




	#print("ROOT IS :",root,"!",len(game.startstate))
	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	while ( iteration <= iterationMax ):
		weather=getWeather(graph,root,game.goal,belief_State)	

		if debug:
			print("NEWSTATE0",root, iteration)

		SelectedNode=SelectionCTPblind(tree,root,iteration,weather,tree.get_node(root).data.scoreAverage,debug =debug)


		newstate=trimmed + (str(SelectedNode)).split(" ")
		if debug:
			print("NEWSTATE1",newstate,trimmed)

		ExpandedNode=ExpansionCTPBlind(tree,SelectedNode,game,newstate,weather)

		print("EXPANDED NODE :",ExpandedNode)

		newstate=trimmed +(str(ExpandedNode.identifier)).split(" ")
		if debug:
			print("NEWSTATE2",newstate,trimmed)

		#print("thatsa10",)
		result = SimulationCTP(game,newstate,weather,belief_State)
		#result = SimulationCTP(game,state+"."+ExpandedNode.identifier[1:],weather)

		if debug:
			print("NEWSTATE3")

		BackPropagationCTP(tree,ExpandedNode,result)

		if debug:
			print("NEWSTATE4")			
		iteration=iteration+1
	
	#data = (list(map( lambda x : [x.identifier,x.data.tries,x.data.scoreAverage], tree.children(root))))
	
	'''
	listOfTries=list(map( lambda x : x.data.tries, tree.children(root)))
	MostTriesID=listOfTries.index(max(listOfTries))

	MovesList=list(map( lambda x : x.identifier, tree.children(root)))

	bestMove=MovesList[MostTriesID]

	#print(list(map( lambda x : x.identifier, tree.children(root))))
	#print(list(map( lambda x : x.data.tries, tree.children(root))))
	#print(list(map( lambda x : x.data.scoreAverage, tree.children(root))))
	

	return bestMove.split(" ")
	'''

	return list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root)))



#monte carlo tree search for CTP, no simulation version
def MontreCarloTreeSearchCTPDeepSelection(game,graph,state,belief_State,iterationMax=10000):
	
	iteration=0
	tree=treelib.Tree()
	root = state[-1]
	
	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	while ( iteration <= iterationMax ):
		weather=getWeather(graph,root,game.goal,belief_State)	

		(score,SelectedNode)=SelectionCTPblindDeep(tree,root,game,iteration,weather,tree.get_node(root).data.scoreAverage)

		BackPropagationCTP(tree,SelectedNode,score)

		iteration=iteration+1

	return list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root)))


#
def MontreCarloTreeSearchCTPOptimistic(game,graph,state,belief_State_source,iterationMax=10000,optimisticWeight=20):

	timeS=time.time()
	
	bestMove=""
	iteration=0

	trimmed=state[:-1]
		
	tree=treelib.Tree()
	
	#root = " ".join(str(x) for x in game.startstate)
	root=state[-1]

	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	

	timeWea=0
	timeSel=0
	timeExp=0
	timeSim=0
	timeBak=0


	b2=(game.score(state)+optimistic(belief_State_source, root, game.goal)*optimisticWeight)


	while ( iteration <= iterationMax ):
		
		belief_State=copy.deepcopy(belief_State_source)
		
		#print("belief_State new")
		#belief_State.show()


		#print("=========================================================== DEBUG :",iteration)

		time1=time.time()
		weather=getWeather(graph,root,game.goal,belief_State_source)
		time2=time.time()
		timeWea+=time2-time1


		b1=(tree.get_node(root).data.scoreAverage*tree.get_node(root).data.tries)
		

		b3=(tree.get_node(root).data.tries+optimisticWeight)
		#print("############",iteration)
		#print("B1 :",b1)
		#print("B2 :",b2)
		#print("B3 :",b3)
		b=(b1+b2)/b3



		time1=time.time()
		SelectedNode=SelectionCTPOptimistic(tree,root,iteration,weather,b,belief_State,game.goal,game.score(state),optimisticWeight)
		time2=time.time()
		timeSel+=time2-time1
		
		newstate=trimmed + (str(SelectedNode)).split(" ")
		#newstate=(str(trimmed)+str(SelectedNode)).split(" ")

		time1=time.time()
		ExpandedNode=ExpansionCTPOptimistic(tree,SelectedNode,game,newstate,weather,belief_State)
		time2=time.time()
		timeExp+=time2-time1

		newstate=trimmed + (str(ExpandedNode.identifier)).split(" ")
		#newstate=(str(trimmed)+str(ExpandedNode.identifier)).split(" ")

		time1=time.time()
		result = game.score(state) + SimulationCTP(game,newstate,weather,belief_State)
		time2=time.time()
		timeSim+=time2-time1

		time1=time.time()
		BackPropagationCTP(tree,ExpandedNode,result)
		time2=time.time()
		timeBak+=time2-time1


		iteration=iteration+1
	

	#print("time :")
	#print("wea : ",timeWea)
	#print("sel : ",timeSel)
	#print("exp : ",timeExp)
	#print("sim : ",timeSim)
	#print("Bak : ",timeBak)	
	timeE=time.time()
	#print("Total time :",timeE-timeS)

	#data = (list(map( lambda x : [x.identifier,x.data.tries,x.data.scoreAverage], tree.children(root))))
	#print(list(map( lambda x : x.data.tries, tree.children(root))))
	'''
	listOfTries=list(map( lambda x : x.data.tries, tree.children(root)))
	MostTriesID=listOfTries.index(max(listOfTries))

	MovesList=list(map( lambda x : x.identifier, tree.children(root)))

	bestMove=MovesList[MostTriesID]

	return bestMove.split(" ")
	'''
	#													  list(map( lambda x : (), tree.children(root) ) )
	print("OPTIMISTIC base :",game.score(state), " " ,    list(map( lambda x : (graph.get_edge_data((x.identifier.split()[-2]),(x.identifier.split()[-1]))['weight']+optimistic(belief_State_source, ((x.identifier.split()[-1])), game.goal) ), tree.children(root) )))
	
	print("SCORE AVERAGES: ", list(map( lambda x : (tuple(x.identifier.split()),x.data.tries,x.data.scoreAverage), tree.children(root))))
	return list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root)))


#
def MontreCarloTreeSearchCTPOptimisticDeep(game,graph,state,belief_State_source,iterationMax=10000,optimisticWeight=20):


	iteration=0
	tree=treelib.Tree()
	root = state[-1]
	

	initialScore=game.score(state)

	tree.create_node(tag=None,identifier=root,parent=None,data=nodeData(0,0))	
	
	op=optimistic(belief_State_source,game.goal)

	
	b2=(initialScore+op[root]*optimisticWeight)

	while ( iteration <= iterationMax ):

		belief_State=copy.deepcopy(belief_State_source)
		f=open("log3.txt","a+")
		print("##@@##belief state O : ",belief_State.openEdge,file=f)
		print("##@@##belief state C : ",belief_State.closedEdge,file=f)
		print("##@@##belief state ? : ",belief_State.unknownEdge,file=f)
		f.close()

		weather=getWeather(graph,root,game.goal,belief_State)	

		b1=((tree.get_node(root).data.scoreAverage)*tree.get_node(root).data.tries)
		b3=(tree.get_node(root).data.tries+optimisticWeight)

		b=(b1+b2)/b3
													   
		(score,SelectedNode)=SelectionCTPOptimisticDeep(tree,root,game,iteration,weather,b,belief_State,initialScore,optimisticWeight)

		BackPropagationCTP(tree,SelectedNode,score)

		iteration=iteration+1
	f=open("log3.txt","a+")
	print("###########OPTIMISTIC base :",game.score(state), " " ,    list(map( lambda x : (optimistic(belief_State_source, ((x.identifier.split()[-1])), game.goal) ), tree.children(root) )),file=f)	
	print("###########SCORE AVERAGES: ", list(map( lambda x : (tuple(x.identifier.split()),x.data.tries,x.data.scoreAverage), tree.children(root))),file=f)
	print("########## RESULT :",list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root))),"##########",file=f)
	f.close()
	return list(map( lambda x : (tuple(x.identifier.split()),x.data.tries), tree.children(root)))



















##########
###TEST###
##########

#x=[1,4,2,6,855,63,2824,2,55]

#print("HOYHOY",x.index(max(x)))




#x=[1,2,3,4]

#print(x[:-1])

#CTPgame=canadianTravellerStochastic(g,"1","8")

#time1=time.time()

#print("Blind : ")
#print(MontreCarloTreeSearchCTP(CTPgame,'1',iterationMax=10000))

#time2=time.time()
#print("TIME : ")
#print(time1,time2)
#print(time2-time1)


#print("Optimistic : ")
#time1=time.time()
#MontreCarloTreeSearchCTPOptimistic(CTPgame,'1',iterationMax=10000)
#time2=time.time()
#print("TIME : ")
#print(time1,time2)
#print(time2-time1)

def playCTP(graph,start,end,iterationMax=10000):
	
	CTPgameBlind    = canadianTravellerStochastic(graph,start,end)
	CTPgameOptimist = canadianTravellerStochastic(graph,start,end)


	belief_State=beliefState(CTPgameBlind.graph,[],[])

	#weather=fixedWeather
	weather=getWeather(graph,start,end,belief_State)

	belief_State.look(start,weather)
	
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
		#state=belief_State		
		#print("start : ",state)
		#print("graph : ",CTPgameBlind.graph)


		#moves=MontreCarloTreeSearchCTP(CTPgameBlind,graph2,state,belief_State,iterationMax)
		moves=MontreCarloTreeSearchCTPDeepSelection(CTPgameBlind,graph2,state,belief_State,iterationMax)
		

		
		weathermoves=CTPgameBlind.moves(state,weather)
		filtermove=[t for t in moves if t[0] in  weathermoves]
		move=max(filtermove,key=lambda item:item[1])[0]
		#print("MOVESW",weathermoves)
	
		#print("MOVES ",moves)

		#print("MOVE F ", filtermove)
		#print("MOVE F2 ", max(filtermove,key=lambda item:item[1])[0])

		print("playing : ",move,state)
		
		(u,v) = move
		

		belief_State.look(v,weather)


		state=CTPgameBlind.play(state,move)
		#print("done : ",state)
	time2=time.time()
	timeBlind=time2-time1
	print("path : ",state)
	print("time for blind :",timeBlind)
	print("Score : ", CTPgameBlind.evaluate(state))
	scoreblind=CTPgameBlind.evaluate(state)
	'''
	'''
	'''

	'''


	print("playing Optimist")

	state=CTPgameOptimist.startstate


	belief_State=beliefState(CTPgameBlind.graph,[],[])
	belief_State.look(start,weather)

	graph2=copy.deepcopy(graph)

	time1=time.time()
	while (CTPgameOptimist.evaluate(state)==ONGOING):
		
		graph2.remove_edges_from(belief_State.closedEdge)

		#print("start : ",state)
		

		#moves=MontreCarloTreeSearchCTPOptimistic(CTPgameBlind,graph2,state,belief_State,iterationMax)
		moves=MontreCarloTreeSearchCTPOptimisticDeep(CTPgameBlind,graph2,state,belief_State,iterationMax)
		
		weathermoves=CTPgameOptimist.moves(state,weather)
		
		#print("MOVES  ",moves)
		#print("MOVES W ",weathermoves)
		filtermove=[t for t in moves if t[0] in  weathermoves]
		#print("MOVES F ",filtermove)
		#belief_State.show()

		move=max(filtermove,key=lambda item:item[1])[0]

		(u,v) = move
		belief_State.look(v,weather)
		print("playing : ",move,state)
		state=CTPgameOptimist.play(state,move)
	
	time2=time.time()
	timeOpt=time2-time1
	print("path : ",state)
	print("Time for Optimist : ",timeOpt)
	scoreOpti=CTPgameOptimist.evaluate(state)
	print("Score : ", scoreOpti)

	#print("Blind : ")
	#time2=time.time()
	#print("TIME : ")
	#print(time1,time2)
	#print(time2-time1)

	print("shortest path ")
	bestState=nx.shortest_path(weather,start,end,"weight")
	print(bestState)
	bestScore=CTPgameOptimist.evaluate(bestState)
	print("best score : ", bestScore)

	f=open("log2.txt","a+")

	print("Blind : ",scoreblind," : ",scoreblind/bestScore,", Opti : ",scoreOpti," : ",scoreOpti/bestScore,", best : ",bestScore,file=f)
	f.close()

	print("shortest path on clear weather ")
	bestState=nx.shortest_path(graph,start,end,"weight")
	print(bestState)
	print("best score : ", CTPgameOptimist.evaluate(bestState))









for i in range(0,1) :

	delau=delaunay.Delaunay(20)
	g=delau.graph

	#g=fixedGraph

	f=open("log3.txt","w+")

	print("Graph : ",file=f)
	print(g.nodes(),file=f)
	print(g.edges(),file=f)

	print("Info : ",file=f)
	for (u,v) in g.edges():
		print(u,v, " : ",g[u][v],file=f)


	f.close()
	playCTP(g,"1","20",iterationMax=10000)
	print("#################################",i,"OUT OF 1","#################################")

	
#titato= ticTacToe()

#x=MontreCarloTreeSearch(titato,"",iterationMax=5000)
#print(x)

#x=MontreCarloTreeSearch(titato,"9581",iterationMax=5000)
#print(x)

'''
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
'''

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


'''

pos = nx.spring_layout(g)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(g, pos, node_size=700)

# edges
nx.draw_networkx_edges(g, pos, g.edges, width=6)

# labels
nx.draw_networkx_labels(g, pos, font_size=20, font_family="sans-serif")

plt.axis("off")
plt.show()

'''