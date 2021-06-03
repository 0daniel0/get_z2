import numpy as np
import matplotlib.pyplot as plt
from cmath import phase

import sisl as si


class Tile:
    """
    Tile(k1, k2, k3, k4)
    A 2 dimensional square in the momentum space,
    which stores the vortex and link.
    """

    def __init__(self, k1, k2, k3, k4, n=4):
        self.n = n  # number of nodes
        self.k = np.array([k1, k2, k3, k4], dtype=float)
        self.dx = k2[0] - k1[0]
        self.dy = k3[1] - k2[1]
        self.u = np.zeros(self.n, dtype=complex)
        self.link = np.zeros(self.n, dtype=float)
        self.f = 0.0
        self.top = False
        self.bottom = False
        self.left = False
        self.right = False
        if len(self.k[0]) == 3:
            self.arrow = (self.k[0] + self.k[1] + self.k[2] + self.k[3]) / self.n + np.array([0, 0, 1])

    @property
    def pos(self):
        return (self.k[0] + self.k[1] + self.k[2] + self.k[3]) / self.n

    @property
    def vortex(self):
        return (self.f - self.link.sum()) / (2 * np.pi)

    def _set_f(self):  # field strength
        exponent = 1
        for u12 in self.u:
            exponent *= u12
        self.f += phase(exponent)

    def _set_link(self):
        for i in range(self.n):
            self.link[i] += phase(self.u[i])

    def set_u_matrices(self, v, chern=False):
        """
        Actually this sets the determinant of U. And calculates f and link
        numpy array v: the eigenvectors of the Hamiltonian
        """
        if not chern:  # then it a z2 calculator subroutine
            for i in range(self.n):
                #             v.T@v
                vov = v[i].T.conj() @ v[(i + 1) % self.n]  # size: occupied x occupied
                u12 = np.linalg.det(vov)
                self.u[i] = u12  # u is a list of scalars
            self._set_link()
            self._set_f()

        else:  # it is a chern calculator subroutine
            for j in range(v[0].shape[1]):
                for i in range(self.n):
                    vov = v[i][:, j].T.conj() @ v[(i + 1) % self.n][:, j]  # size: 1 x 1
                    u12 = vov
                    self.u[i] = u12  # u is a list of scalars
                self._set_link()
                self._set_f()


class CoverBZ:
    """
    covers the whole/ half of the 2D Brillouin Zone with square tiles.
    half: (bool) if True than x axis will twice as long as the y axis
    default is True.
    Int grid: defines the number of Tiles along the x axis as 2 * grid + 3.
    z2: (float) the value of z2 invariant 0 is trivial, 1 is topological.
    chern: (float) the value of Chern invariant 0 is trivial, 1 is topological.
    """

    def __init__(self, grid, half=True):
        if half:
            y = np.linspace(0, 0.5, grid + 2, endpoint=True)
        else:
            y = np.linspace(0, 1, 2 * grid + 3, endpoint=True)
        x = np.linspace(0, 1, 2 * grid + 3, endpoint=True)
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
                k1 = np.array([x[i], y[j]], dtype=float)
                k2 = np.array([x[i + 1], y[j]], dtype=float)
                k3 = np.array([x[i + 1], y[j + 1]], dtype=float)
                k4 = np.array([x[i], y[j + 1]], dtype=float)
                t = Tile(k1, k2, k3, k4)
                self.tiles.append(t)

    def plot(self, *args, **kwargs):
        """
         The red circles dentote a +1,
        the blue circles denote -1 value. Black cirles dentote a not converged value.
        It returns a list containing all the positions and values of n12, as
        [[pos0, n12_0], [pos1, n12_1], ...]. Where pos is a 2 element array, n12 is a float.
        """
        fig, ax = plt.subplots()
        data = []
        for t in self.tiles:
            vort = round(t.vortex, 2)
            if vort != 0:
                if vort == 1:
                    color = "red"
                elif vort == -1:
                    color = "blue"
                else:
                    color = "black"
                r = t.dx / 2  # radius of circles
                circle = plt.Circle(t.pos, r, color=color, alpha=0.8, *args, **kwargs)
                data.append([t.pos, vort])
                ax.add_artist(circle)
            sqaure = plt.Rectangle(t.k[0], t.dx, t.dy,
                                   fc=None, fill=False,
                                   ec="blue", alpha=0.5, ls="--")
            ax.add_artist(sqaure)

        ax.set_aspect("equal")
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_ylim(self.y.min(), self.y.max())
        plt.show()
        return fig, ax


def get_eigenv(k2d, k2d_meta, hamiltonian):
    """
    Returns the proper eigenvectors of the hamiltonian
    """
    trim = [0, 0.5]
    k = np.array(k2d) % 1
    if not hamiltonian.chern:  # z2 case
        if (0.5 < k[0]) and (k[1] in trim):  # if on the boundary
            k[0] = 1 - k[0]  # 'mirroring' k[0]
            v = get_v(k, k2d_meta, hamiltonian, tr=True)
            return v
        if (k[0] in trim) and (k[1] in trim):  # if in TRIM
            v = get_v(k, k2d_meta, hamiltonian, tr=False)
            v = self_tr(v, hamiltonian)
            return v
        else:
            v = get_v(k, k2d_meta, hamiltonian, tr=False)  # if not on the boundary
            return v
    else:  # the chern case
        v = get_v(k, k2d_meta, hamiltonian, tr=False)  # simple, not on the boundary
        return v


def set_tiles(tile, hamiltonian, k2d_meta):
    """
    Calculates everything in one square Tile
    """
    v_list = []
    for k2d in tile.k:  # goes through the corners of tile
        v_list.append(get_eigenv(k2d, k2d_meta, hamiltonian))  # gets eigenvectors of 4 H(k)
    tile.set_u_matrices(v_list, chern=hamiltonian.chern)  # set U matrices, f and link
    return tile


class Hsystem:
    """
    container object for the hamiltonian,
    the occupied number of bands,
    the tau time reverse operator,
    overlap matrix.
    If tau or overlap not provided, it will create one.
    If hamiltonian is not sisl type, occuied should be provided.
    """

    def __init__(self, hamiltonian, source, k2d_meta, occupied=None,
                 chern=False, *args, **kwargs):
        self.chern = chern
        self.args = args
        self.kwargs = kwargs
        self.kwargs['format'] = 'array'

        if type(source) == str:
            sile = si.get_sile(source)
            hamiltonian = sile.read_hamiltonian()
            
        self.hamiltonian = hamiltonian.Hk
            
        if occupied is None:
            # Generates the number of occupied states
            gamma_states = hamiltonian.eigenstate()
            gamma_eigenvalues = gamma_states.eig
            mask = gamma_eigenvalues < 0
            self.occupied = len(gamma_eigenvalues[mask])
        else:
            self.occupied = occupied

        self.non_orthogonal = True
        if self.non_orthogonal:
            self.overlap = hamiltonian.Sk
        if not self.chern:
            k = setk([0, 0], k2d_meta)
            size = len(self.hamiltonian(k=k, *self.args, **self.kwargs))
            self.tau = make_tau(size)


class PositionMeta:
    """
    Data container for storing the elevation of the tiles,
    k_order, which defines the
    "orientation" of the tiles eg.: "xy"
    """

    def __init__(self, k_order, elevation):
        self.k_order = k_order
        self.elevation = elevation


def lowdin(Hk, Sk):
    """
    Calculates the Lowddin orthogonalisation of H
    numpy ndarray Hk: Hamiltonian matrix
    numpy ndarray Sk: the overlap matrix
    return: numpy ndarray the orthogonalised Hamiltonian
    """
    s, U = np.linalg.eigh(Sk)
    s_pot = np.eye(len(s))
    for i in range(len(s_pot)):
        s_pot[i, i] = s[i] ** (-1 / 2)
    Ak = U @ s_pot @ U.T.conj()
    h = Ak @ Hk @ Ak.conj().T
    return h


def get_v(k2d, k2d_meta: PositionMeta, hamiltonian, tr=False):
    """
    Bool tr: if True we apply the time reverse
    operator Tau to the eigenvectors of H(k)
    numpy array k2d:  it is a 2D momenta
    calculates the 3D k and returns the eigen vectors of H(k) which are occupied
    Position_2d_meta k2d: container object for 2d momenta
    """

    shift = 0.0  # this line is not needed
    k = setk(k2d - shift, k2d_meta)  # creates k3d
    if hamiltonian.non_orthogonal:
        Hk = hamiltonian.hamiltonian(k=k, *hamiltonian.args, **hamiltonian.kwargs)
        Sk = hamiltonian.overlap(k=k, *hamiltonian.args, **hamiltonian.kwargs)
        h = lowdin(Hk, Sk)
    else:
        h = hamiltonian.hamiltonian(k=k, *hamiltonian.args, **hamiltonian.kwargs)
    _, v = np.linalg.eigh(h)
    v = v[:, :hamiltonian.occupied]
    if tr:
        v = (hamiltonian.tau @ v).conj()
    return v


def make_tau(size):
    """
    Creates the time reversal matrix
    """
    t = np.array([[0, 1],
                  [-1, 0]], dtype=float)
    tau = np.kron(np.eye(size // 2), t)
    return tau


def time_reverse_v(v, hamiltonian):
    return (hamiltonian.tau @ v).conj()


def self_tr(v, hamiltonian):
    """
    Connects the time-reversed pairs of H.
    Every second eigen vector is replaced with it's time-reversed pair
    array v: the eigen values of H(k) up to the occupied bands
    returns: the new eigenvectors
    """
    v_res = np.zeros(shape=v.shape, dtype=v.dtype)
    for i in range(hamiltonian.occupied // 2):
        trimv = time_reverse_v(v[:, 2 * i], hamiltonian)
        v_res[:, 2 * i] = v[:, 2 * i]
        v_res[:, 1 + 2 * i] = trimv
    return v_res


def set_k_order(plane):
    """
    Sets the proper order of px, py, pz according to the tiles
    str tiles: In which tiles the calculation is done e.g.: 'xy', 'zx'
    return: array with 3 element, which contains the correct order .
    """

    k_helper = {'x': 0, 'y': 1, 'z': 2}
    coords = ['x', 'y', 'z']
    kx = k_helper[plane[0]]
    ky = k_helper[plane[1]]
    coords.remove(plane[0])
    coords.remove(plane[1])
    kz = k_helper[coords[0]]
    return np.array([kx, ky, kz], dtype=int)  # k_order


def setk(k2d, k2d_meta):
    """
    Finds out the k values from tiles and elevation.
    """
    k_order = k2d_meta.k_order
    elevation = k2d_meta.elevation

    k3d = np.array([0, 0, 0], dtype=float)
    k3d[k_order[0]] = k2d[0]
    k3d[k_order[1]] = k2d[1]
    k3d[k_order[2]] = elevation
    return k3d
