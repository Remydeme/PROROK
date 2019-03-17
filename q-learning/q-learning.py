from grid import  Map
from agent import  Agent


if __name__ == "__main__":
    mapo = Map(3, 3)
    mapo.initGrid(epsilon=0.1,eta=0.0)
    mapo.draw()

    alfaromeo = Agent(grid=mapo)

    for i in range(1, 5000):
        alfaromeo.play()

    alfaromeo.displayQTable()
    mapo.draw()