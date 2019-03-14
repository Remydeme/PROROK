from grid import  Map
from agent import  Agent


if __name__ == "__main__":
    mapo = Map(5, 5)
    mapo.initGrid(epsilon=0.1)
    mapo.draw()

    alfaromeo = Agent(grid=mapo)

    for i in range(1, 150):
        alfaromeo.play()

    alfaromeo.displayQTable()
    mapo.draw()