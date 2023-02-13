from bmg.util.measures import to_microm, to_channel
from bmg.environments.cilindrical import SlicedMulti3DCilinderEnv
import numpy as np

from morph import open_file, from_tree_to_morph, write_pkl, open_pkl
from bmg_utils import get_fibers, get_conn, filter_active

import plotly.graph_objects as go
map = {
    "C:\\Users\\igorc\\Desktop\\dbbs\\img_tool\\cell img\\slice_3.jpg": "prova"
}

import datetime
def optimize(num_perc, num_dens):
    for key in map.keys():
        start_time = datetime.datetime.now()
        # Setup, extracting info from the pickle file
        tree, points, electrode_pos = open_file(key)
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
        aggregrated_activity = aggregrated_activity / np.max(aggregrated_activity)
        from pathlib import Path
        path = Path(__file__).parent.parent.parent
        path_file = str(path) + "/" + map[key]
        experimental_activity = open_pkl(path_file)

        # Calculating error
        for trial in experimental_activity:
            error = np.abs(np.sum(aggregrated_activity - trial ** 2) / np.sum(trial))
            print(error)

        end_time = datetime.datetime.now()
        print(end_time - start_time)
# active = origins.reshape((-1, 3))[:num] // 30
# inactive = origins.reshape((-1, 3))[num:] // 30
# go.Figure().add_traces([
#     go.Heatmap(z=aggregrated_activity),
#     go.Scatter(x=active[:,1], y=active[:,0], name="active origins", mode="markers+lines"),
#     go.Scatter(x=inactive[:,1], y=inactive[:,0], name="inactive origins", mode="markers+lines"),
#     go.Scatter(x=grcs[activity_table > 0][:,1], y=grcs[activity_table > 0][:,0], name="active GrCs", mode="markers"),
#     go.Scatter(x=grcs[activity_table == 0][:,1], y=grcs[activity_table == 0][:,0], name="inactive GrCs", mode="markers")
# ]).update_yaxes(
#     scaleanchor = "x",
#     scaleratio = 1,
#   ).show()

optimize(.5, .8)

from deap import base, creator, tools
import random
individuals = 10

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("error", random.random())
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=individuals)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)