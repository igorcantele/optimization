from pathlib import Path
import pickle
from bmg.util.measures import to_microm
import numpy as np
from bsb.morphologies import Branch, Morphology

Z_HEIGHT = 150

def write_pkl(path, res):
    with open(f"{path}.pkl", "wb") as f:
        pickle.dump(res, f)


def read_pickle(path):
    with open(path + ".pkl", "rb") as f:
        return pickle.load(f)


def open_pkl(name):
    with open(f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def open_file(image):
    path = Path(__file__).parent
    path_file = str(path) + "/imcur_prova"
    tree = open_pkl(path_file)[
        image
    ]["segs"]
    points = open_pkl(path_file)[
        image
    ]["points"]
    electrode_pos_2d = to_microm(np.array(open_pkl(path_file)[
        image
    ]["electrode_coords"]).reshape(-1))
    return tree, points, np.append(electrode_pos_2d, 0)

def from_tree_to_morph(tree, points):
    morph = []
    roots = []
    for parent, branch in tree:
        selected_points = np.column_stack(
            (points[branch].reshape(-1, 2), np.zeros(len(branch)))
        )
        selected_points = to_microm(selected_points)
        new_branch = Branch(selected_points, (np.ones(len(selected_points)) * Z_HEIGHT))
        morph.append(new_branch)
        if parent != -1 and parent != None:
            morph[parent].attach_child(new_branch)
        else:
            roots.append(new_branch)
    m = Morphology(roots)
    return m