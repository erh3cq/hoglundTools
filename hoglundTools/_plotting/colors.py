from matplotlib.colors import LinearSegmentedColormap

cyan_k = [(0, 0, 0, 1), (0, 1, 1, 1)]
cyan_k = LinearSegmentedColormap.from_list("cyan_k", cyan_k, N=255)

mage_k = [(0, 0, 0, 1), (1, 0, 1, 1)]
mage_k = LinearSegmentedColormap.from_list("mage_k", mage_k, N=255)

yell_k = [(0, 0, 0, 1), (1, 1, 0, 1)]
yell_k = LinearSegmentedColormap.from_list("yell_k", yell_k, N=255)

red_k = [(0, 0, 0, 1), (1, 0, 0, 1)]
red_k = LinearSegmentedColormap.from_list("red_k", red_k, N=255)

lime_k = [(0, 0, 0, 1), (0, 1, 0, 1)]
lime_k = LinearSegmentedColormap.from_list("lime_k", lime_k, N=255)

blue_k = [(0, 0, 0, 1), (0, 0, 1, 1)]
blue_k = LinearSegmentedColormap.from_list("blue_k", blue_k, N=255)

cyan_0 = [(0, 0, 0, 0), (0, 1, 1, 1)]
cyan_0 = LinearSegmentedColormap.from_list("cyan_0", cyan_0, N=255)

mage_0 = [(0, 0, 0, 0), (1, 0, 1, 1)]
mage_0 = LinearSegmentedColormap.from_list("mage_0", mage_0, N=255)

yell_0 = [(0, 0, 0, 0), (1, 1, 0, 1)]
yell_0 = LinearSegmentedColormap.from_list("yell_0", yell_0, N=255)

red_0 = [(0, 0, 0, 0), (1, 0, 0, 1)]
red_0 = LinearSegmentedColormap.from_list("red_0", red_0, N=255)

lime_0 = [(0, 0, 0, 0), (0, 1, 0, 1)]
lime_0 = LinearSegmentedColormap.from_list("lime_0", lime_0, N=255)

blue_0 = [(0, 0, 0, 0), (0, 0, 1, 1)]
blue_0 = LinearSegmentedColormap.from_list("blue_0", blue_0, N=255)



__cmaps = [k for k,v in locals().items() if isinstance(v, LinearSegmentedColormap)]
__all__ = __cmaps

def __dir__():
    return sorted(__all__)