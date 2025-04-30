import numpy as np
import matplotlib as mpl
import copy
import matplotlib.cm
from matplotlib.colors import colorConverter


"""
Script that creates a cmap used in the Orca module (thus this a copy of said script) 
"""
ylorrd_cmap = copy.copy(matplotlib.cm.get_cmap("YlOrRd"))
ylorrd_cmap.set_bad(color="#AAAAAA")

newcmap2 = mpl.colors.LinearSegmentedColormap.from_list(
    "newcmap2",
    [
        colorConverter.to_rgba(c)
        for c in ["#fff1d7", "#ffda9d", "#ffb362", "#ff8241", "#ff2b29", "#d60026", "#880028",]
    ],
    256,
)
newcmap2._init()
newcmap2.set_bad(color="#AAAAAA")


hnh_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "hnhcmap",
    0.5 * newcmap2(np.linspace(0.0, 1, 256)) + 0.5 * ylorrd_cmap(np.linspace(0.0, 1, 256)),
    256,
)
hnh_cmap.set_bad(color="#AAAAAA")

hnh_cmap_ext = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext",
    np.vstack(
        [
            np.vstack(
                [
                    np.ones(34),
                    np.concatenate(
                        [np.arange(0.97254902, 1, 0.97254902 - 0.97038062), np.ones(21)]
                    ),
                    np.arange(0.82156863, 1, 0.82156863 - 0.81618608),
                    np.ones(34),
                ]
            ).T[::-1, :][:-1, :],
            hnh_cmap(np.linspace(0.0, 1, 256)),
        ]
    ),
)

hnh_cmap_ext.set_bad(color="#AAAAAA")



hnh_cmap_ext3 = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext3",
    np.vstack(
        [
            hnh_cmap_ext(np.linspace(0.0, 1, 256)),
            np.vstack(
                [
                    np.arange(0.51764706, 0.15294118, 0.51764706 - 0.52594939),
                    np.zeros(44),
                    np.ones(44) * 0.15294118,
                    np.ones(44),
                ]
            ).T[1:, :],
        ]
    ),
)
hnh_cmap_ext3.set_bad(color="#AAAAAA")

hnh_cmap_ext5 = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext5", hnh_cmap_ext3(np.linspace(0.0, 1, 512))[32:, :]
)
hnh_cmap_ext5.set_bad(color="#AAAAAA")


