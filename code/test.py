import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from cortex.utils import get_cmap
from matplotlib.colors import Normalize

# get_cmap can be implemented easily if you don't want to install pycortex
# https://github.com/gallantlab/pycortex/blob/01afec145f8d3ca847f9b3910ed6914df62cbbc7/cortex/utils.py#L981
# you can copy a colormap image from here https://gallantlab.org/pycortex/colormaps.html
# and load it with plt.imread like here https://github.com/gallantlab/pycortex/blob/01afec145f8d3ca847f9b3910ed6914df62cbbc7/cortex/utils.py#L1002


class Colormap2D(mcolors.Colormap):
    def __init__(self, cmap, vmin=None, vmax=None, vmin2=None, vmax2=None):
        self.cmap = get_cmap(cmap)  # TODO add try except round plt.get_cmap()
        self.vmin = vmin
        self.vmax = vmax
        self.vmin2 = vmin if vmin2 is None else vmin2
        self.vmax2 = vmax if vmax2 is None else vmax2
        N = self.cmap.colors.shape[0]
        super().__init__(cmap, N)

    def __call__(self, X, alpha=None, bytes=False):
        data1 = X[:, 0]
        data2 = X[:, 1]

        cmap = self.cmap.colors

        norm1 = Normalize(self.vmin, self.vmax)
        norm2 = Normalize(self.vmin2, self.vmax2)

        d1 = np.clip(norm1(data1), 0, 1)
        d2 = np.clip(1 - norm2(data2), 0, 1)
        dim1 = np.round(d1 * (cmap.shape[1] - 1))
        # Nans in data seemed to cause weird interaction with conversion to uint32
        dim1 = np.nan_to_num(dim1).astype(np.uint32)
        dim2 = np.round(d2 * (cmap.shape[0] - 1))
        dim2 = np.nan_to_num(dim2).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        # map r, g, b, a values between 0 and 255 to avoid problems with
        # VolumeRGB when plotting flatmaps with quickflat
        colored = (colored * 255).astype(np.uint8)
        r, g, b, a = colored.T
        r.shape = dim1.shape
        g.shape = dim1.shape
        b.shape = dim1.shape
        a.shape = dim1.shape
        # Preserve nan values as alpha = 0
        aidx = np.logical_or(np.isnan(data1), np.isnan(data2))
        a[aidx] = 0
        return r, g, b, a

    def __hash__(self):
        return hash(self.name)


cc = Colormap2D("GreenWhiteRed_2D")  # GreenWhiteBlue_2D')
