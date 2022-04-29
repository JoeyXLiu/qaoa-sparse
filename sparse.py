"""
Example use:
    python sparse.py --p 3 --graph graph1 --driver fire --iter 100 --seed 1 --save ./
Look at results:
    ipython -c 'import pickle; print(pickle.load(open("fire_p_3_seed_1", "rb")))'
"""

import numpy as np
import networkx as nx
import pickle
import argparse
import copy
from functools import partial
from operator import itemgetter

from qiskit import QuantumCircuit, execute, Aer
from qiskit.algorithms.optimizers import L_BFGS_B

from QAOAKit import (
    get_fixed_angles,
    beta_to_qaoa_format,
    gamma_to_qaoa_format,
    angles_to_qaoa_format,
)

from QAOAKit.utils import (
    get_adjacency_matrix,
    maxcut_obj,
    obj_from_statevector,
    precompute_energies
)

from QAOAKit.parameter_optimization import (
    get_median_pre_trained_kde,
)

from QAOAKit.qiskit_interface import goemans_williamson
from QAOAKit.qaoa import get_maxcut_qaoa_circuit

def get_exact_energy(G, H, p):
    backend = Aer.get_backend('statevector_simulator')
    obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    precomputed_energies = precompute_energies(obj, G.number_of_nodes())
    def f(theta):
        # let's assume first half is gammas, second half is betas
        gamma = theta[:p]
        beta = theta[p:]
        qc = get_maxcut_qaoa_circuit(H, beta, gamma)
        sv = backend.run(qc).result().get_statevector()
        return -obj_from_statevector(sv, obj, precomputed_energies = precomputed_energies)
    return f


parser = argparse.ArgumentParser()
parser.add_argument(
    "--p", type = int,
    required = True,
    help = "QAOA depth")
parser.add_argument(
    "--graph", type = str,
    required = True,
    help = "graph name")
parser.add_argument(
    "--driver", type = str,
    required = True,
    help = "driver name")
parser.add_argument(
    "--iter", type = int,
    required = True,
    help = "max iterations")
parser.add_argument(
    "--seed", type = int,
    required = True,
    help = "random seed")
parser.add_argument(
    "--save", type = str,
    required = True,
    help = "path to save the result")


args = parser.parse_args()

p = args.p
seed = args.seed
maxiter = args.iter


G = nx.read_weighted_edgelist("./graphs/" + args.graph + "/graph", nodetype = int)

H = nx.read_weighted_edgelist("./graphs/" + args.graph + "/" + args.driver, nodetype = int)
H.add_nodes_from(list(range(G.number_of_nodes())))
    
obj = get_exact_energy(G, H, p)
# Lower and upper bounds
lb = np.hstack([np.full(p, -2*np.pi), np.full(p, -2*np.pi)])
ub = np.hstack([np.full(p, 2*np.pi), np.full(p, 2*np.pi)])

ave_d = 2 * H.number_of_edges() / H.number_of_nodes()
if (seed == 1) and (p <= 3):
    # Use KDE sampled parameters
    median, kde = get_median_pre_trained_kde(p)
    n_kde_samples = 10
    maxiter /= n_kde_samples
    new_data = kde.sample(n_kde_samples, random_state=seed)
    angles = np.vstack([np.atleast_2d(median), new_data])
    assert angles.shape == (n_kde_samples + 1, p * 2)
    # scale by average degree of the graph
    if p == 1:
        scaling_factor = np.arctan(1 / np.sqrt(ave_d - 1))
    else:
        scaling_factor = 1 / np.sqrt(ave_d)
    converted_angles = []
    for angle in angles:
        converted_angles.append(
            np.hstack(
                [gamma_to_qaoa_format(angle[:p]) * scaling_factor, beta_to_qaoa_format(angle[p:])]
            )
        )
    angles = np.vstack(converted_angles)
elif (seed == 1) and (p <= 11):
    # use fixed angles
    angle_dicts = [angles_to_qaoa_format(get_fixed_angles(3, p))]
    ds = [3]
    if p <= 5:
        angle_dicts.append(angles_to_qaoa_format(get_fixed_angles(4, p)))
        ds.append(4)
    if p <= 4:
        angle_dicts.append(angles_to_qaoa_format(get_fixed_angles(5, p)))
        ds.append(5)
    maxiter /= len(angle_dicts)
    angles = np.array([
        np.hstack([angle_dict['gamma'] * d / ave_d, angle_dict['beta']]) for angle_dict, d in zip(angle_dicts, ds)
    ])
else:
    # use random
    np.random.seed(seed)
    angles = [np.random.uniform(lb, ub, 2*p)]


bounds = [(lb[i], ub[i]) for i in range(2*p)]

results = []

optimizer = L_BFGS_B(maxfun = maxiter, maxiter = maxiter)

for angle in angles:
    result = optimizer.optimize(2*p, obj, variable_bounds = bounds, initial_point = angle)
    results.append(result)
    
    best_result = min(results, key=itemgetter(1))
    res = {'energy': best_result[1], 'result': best_result, 'seed':args.seed, 'iter': best_result[2], 'args': args}
    pickle.dump(res, open(args.save + args.driver + "_p_" + str(args.p) + "_seed_" + str(args.seed), "wb"))