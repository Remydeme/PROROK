import random
import numpy as np

#Infos : Map Class
# This is a 2D map containing a rewardof -1 on the case that contains an obstacle and a reward of 1 where there
# is the agent house.



class Map():
    """ This class represent a 2D map. You just have to set the dimensions of the map.
        -1 is an obstacle , 0 the road and 1 the house. Those values represent reward that our agent will get when it
        get to the case.
    """
                     # up          down           right     left
    actions_ = {0 : [-1, 0] , 1 : [1, 0], 2 : [0, 1], 3 : [0, -1]}

    location_ = (0, 0)

    def __init__(self, height, width):
        self.__height = height
        self.__width = width
        self.__grid = np.zeros((height,width),dtype=int)

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    @property
    def grid(self):
        return self.__grid

    @height.setter
    def heigth(self, height):
        self.__height = height


    @width.setter
    def width(self, width):
        self.__width = width


    def size(self):
        return self.width * self.height

    def initGrid(self, epsilon=0.1):
        """ This function init the grid. It's set the las case to 1 and a certain number of case to -1
            obstacle is the parameter that permits you to set the perentage of obstacles. 0.1 => 10%
        """
        self.grid[-1][-1] = 1
        size = self.width * self.height
        for i in range(0, size):
            randf = random.random()
            if randf < epsilon:
                x_temp = random.randint(0, self.height - 2)
                y_temp = random.randint(0, self.width - 2)

                self.__grid[x_temp][y_temp] = -1

    def draw(self):
        """ This method draw a 2D map"""
        self.drawLine()

        for l in range(0, self.height):
            print("|", end='', flush=True)
            for c in range(0, self.width):
                print(" " + str(self.grid[l][c]) + " |", end='', flush=True)
            print("\n", end='', flush=True)

            self.drawLine()

    def resetPlayerLocation(self):
        self.location_ = (0, 0)

    def drawLine(self):
        for i in range(0, self.width * 4):
            print("_", end='', flush=True)
        print("\n", end='', flush=True)

    def haveWon(self):
        if self.__grid[self.location_] == 1:
            return True

    def env(self, action):
        ligne = self.location_[0] + self.actions_[action][0]
        column = self.location_[1] + self.actions_[action][1]
        newLocation = (ligne, column)

        if newLocation[0] < 0 or newLocation[0] >= self.height:
            return None

        if newLocation[1] < 0 or newLocation[1] >= self.width:
            return None

        reward = self.grid[newLocation]


        return reward



    def nextState(self, action):
        ligne = self.location_[0] + self.actions_[action][0]
        column = self.location_[1] + self.actions_[action][1]
        newLocation = (ligne, column)

        if newLocation[0] < 0 or newLocation[0] >= self.height:
            return None

        if newLocation[1] < 0 or newLocation[1] >= self.width:
            return None

        newState = (newLocation[0] * self.width ) + newLocation[1]

        return newState


    def move(self, action):
        """ Move to the new state (position) by doing action and return the newState """
        ligne = self.location_[0] + self.actions_[action][0]
        column = self.location_[1] + self.actions_[action][1]
        newLocation = (ligne, column)
        self.location_ = newLocation
        newState = (self.location_[0] * self.width ) + self.location_[1]

        if self.location_[0] == 0 and self.location_[0] == 0:
            print("newState " + str(newState) + " location " + str(self.location_), flush=True)
            return 0

        print("newState " + str(newState) + " location " + str(self.location_), flush=True)
        return newState

