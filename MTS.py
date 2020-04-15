import random
import math
#import time
import treelib
import numpy

#############
###UTILITY###
#############

ONGOING = 0
WIN = 1
LOSS = 2
DRAW = 3

#les donnees d'un node
class nodeData():
	def __init__(self,win,tries):
			self.win=win
			self.tries=tries
			self.both=(win,tries)
			self.value=-1

#verifie si une liste est contenue dans une autre liste
def check(subset,set1):
	return set(subset).issubset(set(set1))

######################
###GAME & TICTACTOE###
######################

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


########################
###THE 4 STEPS OF MTS###
########################

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

###################
###Main Function###
###################

#TODO time restriction not implemented
def MontreCarloTreeSearch(game,state,timeout=10000,iterationMax=10000):
	bestMove=""
	iteration=0
	time=0
	tree=treelib.Tree()
	tree.create_node(tag=None,identifier="",parent=None,data=nodeData(0,0))	
	root=""

	while (time <= timeout and iteration <= iterationMax ):

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

##########
###TEST###
##########

titato= ticTacToe()

x=MontreCarloTreeSearch(titato,"",timeout=10000,iterationMax=50000)
print(x)

x=MontreCarloTreeSearch(titato,"9581",timeout=10000,iterationMax=50000)
print(x)
