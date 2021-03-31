from get_z2.common import *


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


# main function
def getz2(h, elevation=0, grid=8, plane="xy", dimension="3d", chern=False, occupied=None,
          overlap=None, tau=None, *args, **kwargs):
    """
    This method implements the a topological z2 invariant,
    according to https://arxiv.org/abs/cond-mat/0611423.

    Parameters:

    h: the hamiltonian, which defines the system. h can be a sisl type hamiltonian,
    or any user defined function, which first argument is the momentum k,
    and returns an N x N numpy array.

    elevation: (int) if dimension="3d", then it defines the elevation of the plane.
    It should be either 0 or 0.5. Default is 0.

    grid: (int) defines the number of squares the plane will be divided.
    (2 * grid + 3) columns and (2 * grid + 3) (chern=True) or
    (grid + 2) (chern=True) rows will be used. Default is 8.

    plane: (string) defines the plane where the z2 should be calculated.
    It can be eg.: "xy", "xz", "zx", ...  Default is 'xy'.

    dimension: (string) Defines the number of dimensions. It can be either "3d" or "2d".
    In sisl type hamiltonian it should always set to "3d".  Default is '3d'.

    chern: (bool) if it is True, then the routine will calculate the chern number
    instead of the z2.  Default is False.

    occupied: (int) the number of occupied bands. If h is sisl hamiltonian it is
    calculated automatically.

    overlap: (ndarray) It defines the overlap matrix. If h is sisl hamiltonian it is
    calculated automatically. If None, it is assumed, that h is defined in an orthogonal
    basis. Default is None.

    tau: (ndarray) the matrix of time reversal symmetry. If h is sisl hamiltonian it is
    calculated automatically. If None it is assumed to be
    kron(eye(N // 2), [[0, 1],[-1, 0]]).

    Other parameters can be passed to h via *args and **kwargs.

    :returns: a class, that has a plot method and z2 and chern attributes.
    The plot() method plots the value of n12.

    example 1:

    sile = sisl.get_sile("path to sile")
    h = sile.read_hamiltonian()
    res = getz2(h)
    print(round(res.z2, 3) % 2)
    res.plot()

    example 2:

    def chern_insulator(k, t=1, mu=0, gamma=1, delta=1):
    ... some code


    h = chern_insulator
    kwargs = {"t": 1, "mu": 2, "gamma": -0.5, "delta": 1.5}
    res = getz2(h, chern=True, occupied=1, grid=10, **kwargs)
    print(round(res.chern, 3) % 2)
    res.plot()
    """

    bz = CoverBZ(grid, half=not chern)
    k_order = set_k_order(plane)  # k_order[0]='x' and k_order[1]='y'
    if dimension == "3d":
        k2d_meta = Position_meta(k_order, elevation, is3d=True)
    elif dimension == "2d":
        k2d_meta = Position_meta(k_order, elevation, is3d=False)
    else:
        print("dimension is eighter '3d' or '2d. Default is '3d'.")
        raise ValueError
    if chern:
        elevation = 0
    hamiltonian = Hsystem(hamiltonian=h, tau=tau, overlap=overlap,
                          occupied=occupied, k2d_meta=k2d_meta, chern=chern, *args, **kwargs)
    k_upper = 0.5
    k_lower = 0
    for i in range(len(bz.tiles)):
        bz.tiles[i] = set_tiles(bz.tiles[i], hamiltonian, k2d_meta)
        bz.n += bz.tiles[i].vortex
        bz.f += bz.tiles[i].f  # The F

        if not chern:
            if k_upper in bz.tiles[i].k[:, 1]:  # upper bound
                bz.link += bz.tiles[i].link[2]
            if k_lower in bz.tiles[i].k[:, 1]:  # lower bound
                bz.link += bz.tiles[i].link[0]
    if chern:
        bz.chern = ((bz.link - bz.f) / (2 * np.pi)) % 2
    else:
        bz.z2 = ((bz.link - bz.f) / (2 * np.pi)) % 2
    return bz



#
# # example code #1
#
# def chern_insulator(k, t=1, mu=0, gamma=1, delta=1):
#     """
#     example 2 band hamiltonian for chern insulator
#     """
#
#     sx = np.array([
#         [0, 1],
#         [1, 0]
#     ], dtype=float)
#     sy = 1j * np.array([
#         [0, -1],
#         [1, 0]
#     ], dtype=float)
#     sz = np.array([
#         [1, 0],
#         [0, -1]
#     ], dtype=float)
#     k = np.array(k, dtype=float)
#     k *= 2*np.pi
#     def get_d1(k, t=t, mu=mu, gamma=gamma, delta=delta):
#         d = [0, 0, 0]
#         d[0] = -2*gamma*np.sin(k[1])
#         d[1] = delta*np.sin(k[0])
#         d[2] = -2*gamma*np.cos(k[1]) -2*t*np.cos(k[0]) - mu
#         return d
#
#     def get_d(k, t=t, mu=mu, gamma=gamma, delta=delta):
#         d = [0, 0, 0]
#         m = -1*(mu+2*t+2*gamma)
#         d[0] = -2*gamma*k[1]
#         d[1] = delta*k[0]
#         d[2] = m
#         return d
#
#     d = get_d1(k)
#     hamiltonian = d[0] * sx + d[1] * sy + d[2] * sz
#     return hamiltonian
#
#
# h = chern_insulator
# for mu in np.linspace(0, 4, 11, endpoint=True):
#     kwargs = {"t": 1, "mu": mu, "gamma": -0.5, "delta": 1.5}
#     res = getz2(h, chern=True, occupied=1, grid=10, **kwargs)
#     print(round(res.chern, 3)%2, mu)
# # res.plot()
#
#
# # example code # 2
#
#
# sile = si.get_sile("path to sile")
# h = sile.read_hamiltonian()
# res = getz2(h)
# print(round(res.z2, 3)%2)
