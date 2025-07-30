import numpy as np
import matplotlib as mpl
import copy
import matplotlib.cm
from matplotlib.colors import colorConverter


"""
Script that creates a cmap used in the Orca module (thus this a copy of said script), plus 
a new cmap created for substracted matrices (resulting from the substraction of two matrices).
"""
ylorrd_cmap = copy.copy(matplotlib.cm.get_cmap("YlOrRd"))
ylorrd_cmap.set_bad(color="#AAAAAA")

newcmap2 = mpl.colors.LinearSegmentedColormap.from_list(
    "newcmap2",
    [
        (c)
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


# This part is new

# Step 1: Define base colormaps for different blue transitions
blue_cmap_light = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_cmap_light",
    ["#eaffea", "#b2ffb2", "#1dd3b6"], 128
)
blue_cmap_mid_light = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_cmap_mid",
    ["#1dd3b6", "#19bfcf", "#40c9e2", "#00bcd4"],128
)
blue_cmap_mid_dark = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_cmap_mid_2",
    ["#00bcd4", "#1790d2", "#1976d2"], 128
)
blue_cmap_dark = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_cmap_dark",
    ["#1976d2", "#1565c0", "#104a8c", "#072447", "#001933"], 128
)

# Step 2: Sample each colormap
light_colors = blue_cmap_light(np.linspace(0, 1, 128))
mid_l_colors = blue_cmap_mid_light(np.linspace(0, 1, 128))
mid_d_colors = blue_cmap_mid_dark(np.linspace(0, 1, 128))
dark_colors = blue_cmap_dark(np.linspace(0, 1, 128))

# Step 3: Stack the arrays for smooth transition
blue_cmap_smooth_data = np.vstack([light_colors, mid_l_colors[1:], mid_d_colors[1:] ,dark_colors[1:]])

# Step 4: Create the final smooth colormap
blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_cmap", blue_cmap_smooth_data
)
blue_cmap.set_bad(color="#AAAAAA")