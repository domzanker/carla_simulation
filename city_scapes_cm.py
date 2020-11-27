import collections

labels = collections.OrderedDict()
labels["Unlabeled"] = (0, 0, 0)
labels["Building"] = (70, 70, 70)
labels["Fence"] = (100, 40, 40)
labels["Other"] = (55, 90, 80)
labels["Pedestrian"] = (220, 20, 60)
labels["Pole"] = (153, 153, 153)
labels["RoadLine"] = (157, 234, 50)
labels["Road"] = (128, 64, 128)
labels["SideWalk"] = (244, 35, 232)
labels["Vegetation"] = (107, 142, 35)
labels["Vehicles"] = (0, 0, 142)
labels["Wall"] = (102, 102, 156)
labels["TrafficSign"] = (220, 220, 0)
labels["Sky"] = (70, 130, 180)
labels["Ground"] = (81, 0, 81)
labels["Bridge"] = (150, 100, 100)
labels["RailTrack"] = (230, 150, 140)
labels["GuardRail"] = (180, 165, 180)
labels["TrafficLight"] = (250, 170, 30)
labels["Static"] = (110, 190, 160)
labels["Dynamic"] = (170, 120, 50)
labels["Water"] = (45, 60, 150)
labels["Terrain"] = (145, 170, 100)


def apply_cityscapes_cm(img, order="rgb"):
    if order == "rgb":
        pos = 0
    else:
        pos = 2
    for i, rgb in enumerate(labels.values()):
        img[img[:, :, pos] == i] = rgb

    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    img = np.full([100, 100, 3], fill_value=(1, 0, 0))
    img[:50] = (10, 0, 0)
    img = apply_cityscapes_cm(img)
    plt.imshow(img)
    plt.show()
