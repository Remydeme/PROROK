import numpy as np
import random
from grid import  Map


class Agent():

    """This class represent the AI agent """

    gamma_ = 0.9
    l_ = 0.1
    epsilon_ = 0.3
    posibleAction_ = 4

    def __init__(self, grid):

        """ Init the object Agent
            @size_state : represent the number of case in the map

        """
        number_of_state = grid.height * grid.width
        self.__qTable = np.zeros((number_of_state, 4), dtype=float) # we can just go : up, down , left , or right 4 possible actions
        self.__state = 0 # position of the player on the map
        self.__stateQValue = 0
        self.grid = grid

    @property
    def gamma(self):
        return self.gamma_
    @property
    def l(self):
        return self.l_

    @property
    def epsilon(self):
        return self.epsilon_

    @gamma.setter
    def gamma(self, value):
        if value > 1:
            gamma_ = 1
        elif value <= 0:
            gamma_ = 0.01
        else:
            gamma_ = value

    @epsilon.setter
    def eps(self, value):
        if value > 1:
            self.epsilon_ = 1
        elif value < 0:
            self.epsilon_ = 0
        else:
            self.epsilon_ = value

    @l.setter
    def lerning_rate(self, value):

        if value > 1:
            self.l_ = 1
        elif value < 0:
            self.l_ = 0.01
        else:
            self.l_ = value

    def displayQTable(self):
        size = self.grid.size()
        for i in range(0, size):
            print(self.__qTable[i])


    def resetPlayer(self):
        self.grid.resetPlayerLocation()
        self.__state = 0

    def hasFinished(self):
        return self.grid.haveWon()


    def play(self):
        while self.hasFinished() != True:
            self.takeAction()
        self.resetPlayer()
        self.epsilon_ -= 0.001

    def pickBestAction(self):
        """ This function will return the next best action that will lead to the state S' and the esperance value of
            this case
        """
        action = 0
        state = int(self.__state)
        qValue = self.__qTable[state][0] # it's the esperance value of this case
        for i in range (0, self.posibleAction_):
            qValueTemp = self.__qTable[state][i]
            if qValueTemp > qValue:
                qValue = qValueTemp
                action = i
        return action

    def nextState(self, action):
        # use the ction to get the next state
        nextState = self.grid.nextState(action)

        return nextState



    def updateStateValue(self, state, value, action):
        """ Update the Qtable with the new Qvalue of a state S knowing that we take the action a"""
        self.__qTable[int(state)][action] = value




    def takeAction(self):
        """ Make one move on the map following the policy """
        randf = random.random()
        if self.epsilon_ > randf:
            action = random.randint(0, 3)
            futureState = self.nextState(action)
        else:
            action = self.pickBestAction()
            futureState = self.nextState(action=action)

        #check future state
        if futureState == None:
            return

        #get the reward of the future state 
        reward = self.grid.env(int(action))

        # check if the move was valid
        if reward == None:
            return
        # we move to the new position

        futureAction = self.pickBestAction()

        futureStateQValue = self.__qTable[futureState][futureAction]

        # now we compute the value Q(S, A)new
        # Q(S, A)new = Q(S, A)old + l * [ reward + gamma * (Q(S', A') - Q(S, A)old)]
        qValue = self.__stateQValue + self.l_ * (reward + self.gamma_ * futureStateQValue - self.__stateQValue)
        print(qValue)
        # We set the current Qvalue to the new Qvalue compute knowing that we have move to this new state S
        # doing the action a
        self.__stateQValue = qValue


        self.updateStateValue(state=self.__state, value=qValue, action=int(action))

        # finally we update the state
        self.__state = futureState

        self.grid.move(action)




