"""All the functions assume a soft geometric connection model with uniform node and affiliation embeddings."""

import math
import numpy as np
import random
import csv

from msean.config import Config
from msean.generation.connect import d

# --- Probability Density Functions ---

def uniform_pdf(cfg:Config, x):
    """
    Probability density function for a unit uniform distribution.
    """
    return 1

def sq_line_picking_prob(cfg:Config, delta):
    """
    Probability density function for the distance δ between two points chosen uniformly on the unit square. 

    P(δ) =  { 2δ (δ² - 4δ + 𝝅) if 0 ≤ δ ≤ 1,
            { 2δ (4√(δ² - 1) - (δ² + 2 - 𝝅) - 4tan⁻¹√(δ² -1)) if 1 ≤ δ ≤ √2
    """
    if delta <= 1:
        return 2*delta * (delta**2 - 4*delta + math.pi)
    else:
        return 2*delta * (4 * math.sqrt(delta**2 - 1) - (delta**2 + 2 - math.pi) - 4 * math.atan(math.sqrt(delta**2 - 1)))
    
# --- Connection Function ---
 
def connection_f(cfg:Config, node_pos, affiliation_pos):
    """
    Soft geometric graph connection function.
    """
    return math.exp(- cfg.connection.xi * ((d(node_pos, affiliation_pos, cfg)/cfg.connection.r_0)**cfg.connection.s))

# --- Expected Weighting for Nodes ---

def expected_weighting_for_v(cfg, x_u, y_u):
    """
    Monte-Carlo approximation of the expected value of ω for a node u at position xᵤ, yᵤ.
    """
    total = 0
    for _ in range(cfg.analysis.mc.n_trials):
        x_a, y_a = (random.uniform(0,1), random.uniform(0,1))
        total += connection_f(cfg, (x_u, y_u), (x_a, y_a)) * uniform_pdf(cfg, (x_a, y_a))

    return total / cfg.analysis.mc.n_trials
    
def gen_expected_weighting_lookup(cfg):
    """
    Generates a lookup table for expected weighting values ω for nodes at positions xᵤ, yᵤ to allow for one-time calculation.
    """
    coord_vals = np.linspace(0, 1, cfg.analysis.mc.omega_lookup_res)
    omega_lookup = np.zeros((cfg.analysis.mc.omega_lookup_res, cfg.analysis.mc.omega_lookup_res))

    for i, x_u in enumerate(coord_vals):
        for j, y_u in enumerate(coord_vals):
            omega_lookup[i][j] = expected_weighting_for_v(cfg, x_u, y_u)

    return omega_lookup, coord_vals

# --- Pairwise Connection Probability --- 

def find_closest_coords(cfg:Config, coord_vals, x, y):
    return np.argmin([abs(c - x) for c in coord_vals]), np.argmin([abs(c - y) for c in coord_vals])

def q(cfg: Config, x_u, y_u, x_v, y_v, omega_coord_vals, omega_lookup, K):
    """ 
    Monte-Carlo approximation of the probability that two nodes at positions (xᵤ, yᵤ) and (xᵥ, yᵥ) connect. 
    """
    total = 0
    for _ in range(cfg.analysis.mc.n_trials):
        # Generate a random (uniform) affiliation position.
        x_a, y_a = (random.uniform(0,1), random.uniform(0,1))

        # Find connection probability for u to a
        w_u = connection_f(cfg, (x_a, y_a), (x_u, y_u))
        i_x, i_y = find_closest_coords(cfg, omega_coord_vals, x_u, y_u)
        exp_w_u = float(omega_lookup[i_x][i_y])

        c_u = w_u / ((K-1)*exp_w_u + w_u) 

        # Find connection probability for v
        w_v = connection_f(cfg, (x_a, y_a), (x_v, y_v))
        i_x, i_y = find_closest_coords(cfg, omega_coord_vals, x_v, y_v)
        exp_w_v = float(omega_lookup[i_x][i_y])

        c_v = w_v / ((K-1)*exp_w_v + w_v) 

        # u and v connect if they both connect to a.
        total += c_u * c_v

    return total / cfg.analysis.mc.n_trials

def p_delta(cfg, delta, omega_coord_vals, omega_lookup, K):
    """
    Monte-Carlo approximation of the probability of two nodes distance δ apart connecting.
    """
    total = 0
    for _ in range(cfg.analysis.mc.n_trials):
        # Sample uniformly from the set of points δ apart inside the unit square.
        x_v = -1
        y_v = -1 
        while not 0 <= x_v <= 1 or not 0 <= y_v <= 1:
            x_u, y_u = (random.uniform(0,1), random.uniform(0,1))
            theta = random.uniform(0,2 * math.pi) # Choose θ uniformly 
            x_v, y_v = (x_u + delta * math.cos(theta), y_u + delta * math.sin(theta)) # Find the point v which is δ away from u and 

        total += q(cfg, x_u, y_u, x_v, y_v, omega_coord_vals, omega_lookup, K)

    return total / cfg.analysis.mc.n_trials

# --- Utilities --- 

def normalise_as_pdf(vals, bin_size):
    Z = 0
    for i in range(len(vals) - 1):
        mean = (vals[i] + vals[i+1]) / 2
        Z += mean * bin_size

    return [elem / Z for elem in vals]

# --- Reading and Writing --- 

def read_lookup_from_file(file_name):
    """
    Read a lookup table from a csv file.
    """
    lookup = []
    with open(file_name, mode ='r') as f:
        csvFile = csv.reader(f)
        for lines in csvFile:
                print(lines)
                lookup.append([float(elem) for elem in lines])
    
    return lookup

def write_lookup_to_file(file_name, lookup):
    """
    Write a lookup table to a csv file.
    """
    np.savetxt(file_name,lookup,delimiter=",")
