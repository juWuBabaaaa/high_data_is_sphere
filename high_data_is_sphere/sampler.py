import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SAMPLER:

    def __init__(self):

        self.EPS = 0.001
        self.RESOLUTION = int(1.0 / self.EPS)
        self.points = 0
        self.n = 0
        self.rnd = np.random.RandomState(seed=999)
        self.points_3d = 0
        self.generate_points()

    def generate_points(self):

        _x, _y = np.linspace(0, 1, self.RESOLUTION), np.linspace(0, 1, self.RESOLUTION)
        grid_x, grid_y = np.meshgrid(_x, _y)

        grid_x[(grid_x-0.5)**2 + (grid_y-0.5)**2 > 0.0625] = 0
        grid_y[(grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2 > 0.0625] = 0

        __x = grid_x.reshape(1, -1)
        __y = grid_y.reshape(1, -1)

        _points = np.unique(
            np.concatenate((__x, __y), axis=0),
            axis=0
        )

        self.points = np.delete(
            _points,
            np.where(_points == 0)[1],
            axis=1
        )

        self.n = max(self.points.shape)

        self.points_3d = np.concatenate(
            (self.points, np.zeros((1, self.n))),
            axis=0
        )

    def sampler(self, bs):

        while True:
            index = self.rnd.randint(0, self.n, size=bs)
            yield self.points[:, index].T

    def sampler_3d(self, bs):

        while True:
            index = self.rnd.randint(0, self.n, size=bs)
            yield self.points_3d[:, index].T


if __name__ == '__main__':

    BS = 500

    s = SAMPLER()
    p = s.sampler(BS)
    q = s.sampler_3d(BS)

    P1 = p.__next__()
    P2 = q.__next__()

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(P1[0, :], P1[1, :], '.')
    ax1.axis('equal')
    ax1.axis([0, 1, 0, 1])

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(P2[0, :], P2[1, :], P2[2, :], s=2)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)

    plt.show()
