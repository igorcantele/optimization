from bmg.util.measures import to_microm, to_channel
from bmg.fiber_generators.first_evidence_based_model import *
from bmg.origin_generators.test_dummies import UniformOrigins
from bmg.granular_generator.random import RandomGranularGen3D
from bmg.connection_generator.by_position import PositionConnectionGen

GRCS_DENSITY = 3.9e-5
GLOMS_DENSITY = 3e-6

def get_fibers(env, morph):
    num = len(morph.branches)
    r = morph.branches[0].radii[0]
    total_len = morph.euclidean_dists
    total_vol = num * (r * r * r * np.pi) + total_len * (r * r * np.pi)

    numerosity = int(total_vol * GLOMS_DENSITY / 3 / len(morph.branches))

    og = UniformOrigins()
    fb = EvidenceBasedFibers()

    origins = og.create_origins(env, numerosity)
    return origins, fb.grow(origins)

def filter_active(electrode_pos, num, num_dens, fibers):
    length = int(len(fibers) * num)
    dens = int(length * num_dens)
    return np.random.choice(np.array(fibers)[:length], dens, replace=False)

def get_conn(env, fibers):
    gr = RandomGranularGen3D()
    con = PositionConnectionGen()
    granular_cells = gr.place(env, GRCS_DENSITY)
    left_mask = granular_cells[:,0] > 0
    bottom_mask = granular_cells[:,1] > 0
    granular_cells = granular_cells[left_mask & bottom_mask]
    conn_map = con.connect(granular_cells, fibers)
    return conn_map, granular_cells // 42