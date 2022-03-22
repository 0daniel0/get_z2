"""
This module implements the a topological z2 invariant,
according to https://arxiv.org/abs/cond-mat/0611423.
Use the getz2 function to calculate the Z2 invariant or Chern number along a tiles.
"""

from .shapes import *
import warnings
# ignores tqdm import warning
warnings.filterwarnings("ignore")
from tqdm.autonotebook import tqdm as tqdm
# ignors sisl fermi_level() overflow warining
warnings.filterwarnings("ignore", message="overflow encountered in exp")


def alias(f, *args, **kwargs):
    """
    decorator which helps using non-sisl hamiltonians
    """
    class h():
        def __init__(self, f):
            self.H = f
            
        def Hk(self, *args, **kwargs):
            if kwargs.get("format"):
                del kwargs["format"]
            return self.H(*args, **kwargs)
        
        def Sk(self, *args, **kwargs):
            return np.eye(len(self.Hk([0, 0, 0])))
        
        def eigenstate(self):
            eig = np.linalg.eigvalsh(self.Hk(k=[0, 0, 0]))
            
            class eigvals():
                def __init__(self, eig):
                    self.eig = eig
                def eig(self):
                    return self.eig
            return eigvals(eig)
        
        def fermi_level(self):
            return 0
    return h(f)


# main function
def getz2(h=None, source=None, shape=None, elevation=0, grid=8, plane="xy",
          chern=False, nbands=None, eta=False):
    """
    This method implements the a topological z2 invariant,
    according to https://arxiv.org/abs/cond-mat/0611423.

    Parameters:

    h: the hamiltonian, which defines the system. h should be a sisl type hamiltonian.
    source: (str) path to the appropriate siseta file.
    shape: (FreeShape) class, which contains all the information of the k-points.
    If not None, eleveation, grid and plane is ignored.
    elevation: (int) if dimension="3d", then it defines the elevation of the tiles.
    It should be either 0 or 0.5. Default is 0.
    grid: (int) defines the number of squares the tiles will be divided.
    (2 * grid + 3) columns and (2 * grid + 3) (chern=True) or
    (grid + 2) (chern=True) rows will be used. Default is 8.
    plane: (string) defines the tiles where the z2 should be calculated.
    It can be eg.: "xy", "xz", "zx", ...  Default is 'xy'.
    chern: (bool) if it is True, then the routine will calculate the chern number
    instead of the z2.  Default is False.
    nbands: (int) the number of occupied bands. If None,
    everything is calculated up to the fermi energy.
    eta: (bool) If True it shows a progressbar. Default is False.

    :returns: a class, that has a plot method and z2 and chern attributes.
    The plot() method plots the value of n12.
    A color of red corresponds to the value of +1, blue to the -1, white for 0
    and black otherwise. (Black means bad calculation, you shouldn't see any of them.)
    """
    if shape is None:
        bz = CoverBZ(grid, half=not chern)
    else:
        bz = shape
        plane = "xy"
        elevation = 0
    k_order = set_k_order(plane)  # k_order[0]='x' and k_order[1]='y'
    k2d_meta = PositionMeta(k_order, elevation)
    hamiltonian = Hsystem(hamiltonian=h, source=source, k2d_meta=k2d_meta, chern=chern, occupied=nbands)
    k_upper = 0.5
    k_lower = 0
    if eta:
        main_loop = tqdm(range(len(bz.tiles)))
    else:
        main_loop = range(len(bz.tiles))
    for i in main_loop:
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
