"""
This module implements the a topological z2 invariant,
according to https://arxiv.org/abs/cond-mat/0611423.
Use the getz2 function to calculate the Z2 invariant or Chern number along a tiles.
"""

from math import sin, cos, pi, sqrt, atan2

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common import *


def rotate(vector, degree, axis='x'):
    """
    Rotates the k points around axis, with degree DEG
    axis: (str) 'x', 'y' or 'z' the axis of rotation.
    degree: (float) [0:360] Degree of rotation.
    """
    ax = {'x': 0, 'y': 1, 'z': 2}
    phi = pi / 180 * degree
    rot_x = np.array([
        [1, 0, 0],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi), cos(phi)]
    ])

    rot_y = np.array([
        [cos(phi), 0., sin(phi)],
        [0, 1, 0.],
        [-sin(phi), 0., cos(phi)]
    ])

    rot_z = np.array([
        [cos(phi), -sin(phi), 0],
        [sin(phi), cos(phi), 0],
        [0., 1., 0.]
    ])
    rot = [rot_x, rot_y, rot_z]
    return rot[ax[axis]] @ vector


def translate(vector, a):
    """
    Translates all k points with a. It's just k+a
    a: (int or ndarray)
    """
    return vector + a


def apply_func(list_of_tiles, f, *args, **kwargs):
    for j in range(len(list_of_tiles)):
        tile = list_of_tiles[j]
        for i in range(tile.n):
            tile.k[i] = f(tile.k[i], *args, **kwargs)
        tile.arrow = f(tile.arrow, *args, **kwargs)
        list_of_tiles[j] = tile
    return list_of_tiles


def rotation(plane, degree, axis):
    """Used internally"""
    return apply_func(plane, rotate, degree=degree, axis=axis)


def translation(plane, a):
    """used internally"""
    return apply_func(plane, translate, a)


def resize(v, a):
    """
    Multiplies all k-points with a.
    a: (int or ndarray)
    """
    return a * v


def tosphere(v, r=1):
    """
    If the k-points are centeredd to the origin and cube shaped,
    it transforms it into a sphere, with radius r.
    r: (float) the radius of the sphere
    """
    x = v[0]
    y = v[1]
    z = v[2]
    rs = sqrt(x ** 2 + y ** 2)
    phi = atan2(y, x)
    theta = atan2(rs, z)
    xx = r * sin(theta) * cos(phi)
    yy = r * sin(theta) * sin(phi)
    zz = r * cos(theta)
    return np.array([xx, yy, zz])


class TileSet:
    """
    Need: modify the next method
    """

    def __init__(self, grid):
        y = np.linspace(0, 1, grid, endpoint=True)
        x = np.linspace(0, 1, grid, endpoint=True)
        self.x = x
        self.y = y
        self.tiles = []
        self.z2 = None
        self.chern = None
        self.link = 0
        self.f = 0
        self.n = 0
        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                k1 = np.array([x[i], y[j], 0], dtype=float)
                k2 = np.array([x[i + 1], y[j], 0], dtype=float)
                k3 = np.array([x[i + 1], y[j + 1], 0], dtype=float)
                k4 = np.array([x[i], y[j + 1], 0], dtype=float)
                t = Tile(k1, k2, k3, k4)
                # Here we flag the boundaries
                if i == 0:
                    t.left = True
                if i == len(x) - 2:
                    t.right = True
                if j == 0:
                    t.bottom = True
                if j == len(y) - 2:
                    t.top = True
                self.tiles.append(t)

    def get_tiles(self):
        return self.tiles


class FreeShape:
    """
    Base class for creating appropriate shapes for Chern calculation.
    grid: (int) grid * grid number of k-points are used per surface.
    shape_type: (str)
    if open: 1 surface is created (plane),
    if half: 4 surfaces are created (cube, without 2 opposite sides),
    if closed: 6 surfaces are created (cube).
    """
    def __init__(self, grid, shape_type='closed'):
        """
        type can be:
         open,
         half-open,
         closed
        """
        self.chern = None
        self.z2 = None
        self.link = 0
        self.f = 0
        self.n = 0
        self.tiles = []
        if shape_type == 'open':
            self.tiles = TileSet(grid).get_tiles()
        if shape_type == 'half':
            # This will be 4 sides of a cube
            self.make_half(grid=grid)
        if shape_type == 'closed':
            # This will be 6 sides of a cube
            self.make_closed(grid)

    def make_half(self, grid):
        # upper side
        tu = TileSet(grid).get_tiles()

        # right side
        tr = TileSet(grid).get_tiles()
        tr = rotation(tr, 90, 'y')
        tr = translation(tr, [1, 0, 0])

        # bottom side
        tb = TileSet(grid).get_tiles()
        tb = rotation(tb, 180, 'y')
        tb = translation(tb, [1, 0, -1])

        # left side
        tl = TileSet(grid).get_tiles()
        tl = rotation(tl, -90, 'y')
        tl = translation(tl, [0, 0, -1])

        self.tiles += tu + tr + tb + tl

    def make_closed(self, grid):
        self.make_half(grid)

        # behind side
        tb = TileSet(grid).get_tiles()
        tb = rotation(tb, -90, 'x')
        tb = translation(tb, [0, 1, 0])

        # front side
        tf = TileSet(grid).get_tiles()
        tf = rotation(tf, 90, 'x')
        tf = translation(tf, [0, 0, -1])

        self.tiles += tb + tf

    def deform(self, f, *args, **kwargs):
        """
        Defroms all the k-points according to f.
        f: arbitrary function to deform the k-points.
        It's first argument should be the positions of the k-points,
        other arguments can be passed with  *args, **kwargs.
        """
        self.tiles = apply_func(self.tiles, f, *args, **kwargs)

    def plot(self):
        """
        The plot() method plots the value of n12.
        A color of red corresponds to the value of +1, blue to the -1, white for 0
        and black otherwise. (Black means bad calculation, you shouldn't see any of them.)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = [], [], []
        # verts = []
        for i in range(len(self.tiles)):
            tile = self.tiles[i]
            xv = tile.k[:, 0]
            yv = tile.k[:, 1]
            zv = tile.k[:, 2]
            verts = [list(zip(xv, yv, zv))]
            x += list(xv)
            y += list(yv)
            z += list(zv)

            vort = round(tile.vortex, 2)
            if vort != 0:
                alpha = 0.8
                if vort == 1:
                    color = "red"
                elif vort == -1:
                    color = "blue"
                else:
                    color = "black"
            else:
                alpha = 0.2
                color = "blue"  # [0.5, 0.5, 1]
            collection = Poly3DCollection(verts, linewidths=1, alpha=alpha)
            face_color = color
            collection.set_facecolor(face_color)
            ax.add_collection3d(collection)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        ax.scatter(x, y, z)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")
        ax.set_zlabel("$k_z$")
        plt.show()


class Sphere(FreeShape):
    """
    Child class of FreeShape, it creates 6 * grid * grid k-points,
    centered to the origo, distributed along a sphere with radius 1.
    """
    def __init__(self, grid):
        super().__init__(grid, "closed")
        self.deform(translate, a=[-0.5, -0.5, 0.5])
        self.deform(resize, 2)
        self.deform(tosphere)
