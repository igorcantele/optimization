from bmg.util.measures import to_microm, to_channel
from bmg.environments.cilindrical import SlicedMulti3DCilinderEnv
import numpy as np

from .morph import open_file, from_tree_to_morph, write_pkl, open_pkl
from .bmg_utils import get_fibers, get_conn, filter_active

import plotly.graph_objects as go

from pathlib import Path

path = Path(__file__).parent.parent
path_file = str(path) + "/activity_data"
map = open_pkl(path_file)
def max_min(num, low=0, up=1):
    up = min(num, up)
    return max(up, low)
# for idx in range(len(map["slice_1_stim.jpg"])):
#     go.Figure().add_traces(
#         go.Heatmap(z=map["slice_1_stim.jpg"][idx])
#     ).update_yaxes(
#     scaleanchor="x",
#     scaleratio=1,
# ).show()

def compare(num_perc, num_dens):
    num_perc = max_min(num_perc)
    num_dens = max_min(num_dens)
    for key in map.keys():
        if key != "slice_1_stim.jpg":
            continue
        # Setup, extracting info from the pickle file
        tree, points, electrode_pos = open_file("C:\\Users\\igorc\\Desktop\\dbbs\\img_tool\\cell img\\"+key)
        morph = from_tree_to_morph(tree, points)
        morph.simplify(to_microm(50))

        # Generating environment and active fibers
        env = SlicedMulti3DCilinderEnv(morph)
        origins, fibers = get_fibers(env, morph)
        active_fibers = filter_active(electrode_pos, num_perc, num_dens, fibers)

        # Calculating connections based on the active fibers and creating tool to convert activity in signal
        conn_map, grcs = get_conn(env, active_fibers)
        activity_table = np.zeros(len(grcs), dtype=int)
        signal_lookup = np.array([0, 1, 1, 2, 2])

        # When a GrC has a connection it is registered in the activity table
        if len(conn_map.shape) > 1:
            for idx, fiber in enumerate(active_fibers):
                    active_conn_mask = conn_map[:, 0] == idx
                    active_grc = conn_map[active_conn_mask, 1]
                    activity_table[active_grc] += 1

        # Converting activity into signal
        grc_signals = signal_lookup[activity_table]

        # Creating 64x64 matrix to compare signal with experimental data
        aggregrated_activity = np.zeros((64, 64))
        for sig, grc in zip(grc_signals, grcs):
            aggregrated_activity[int(grc[0]), int(grc[1])] += sig
        # Standardizing aggregated activity and getting standardized experimental activity
        if np.max(aggregrated_activity):
        aggregrated_activity = aggregrated_activity / np.max(aggregrated_activity)

        experimental_activity = map[key]

        # Calculating error
        errors = []
        for trial in experimental_activity:
            error = np.abs(np.sum(aggregrated_activity - trial ** 2) / np.sum(trial))
            errors.append(error)
        return errors

