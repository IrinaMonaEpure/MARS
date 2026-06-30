"""All the functions assume a soft geometric connection model with uniform node and affiliation embeddings."""

import math
import numpy as np
import random
import csv

from msean.config import Config
from msean.generation.connect import d

SQRT2 = math.sqrt(2)

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
    return math.exp(-1 * d(node_pos, affiliation_pos, cfg)/(cfg.connection.alpha * SQRT2))

# --- Expected Weighting for Nodes ---

def mc_expected_weighting_for_v(cfg, x_u, y_u):
    """
    Monte-Carlo approximation of the expected value of ω for a node u at position xᵤ, yᵤ.
    Better method for higher dimension spaces.
    """
    total = 0
    for _ in range(cfg.analysis.mc_n_trials):
        x_a, y_a = (random.uniform(0,1), random.uniform(0,1))
        total += connection_f(cfg, (x_u, y_u), (x_a, y_a)) * uniform_pdf(cfg, (x_a, y_a))

    return total / cfg.analysis_mc_n_trials

def trapezoidal_expected_weighting_for_v(cfg, x_u, y_u):
    """
    Trapezoidal approximation of the expected value of ω for a node u at position xᵤ, yᵤ.
    """
    total = 0

    coord_vals = np.linspace(0, 1, cfg.analysis.trapezoidal_res)
    
    for x_a in coord_vals:
        for y_a in coord_vals:
            total += connection_f(cfg, (x_u, y_u), (x_a, y_a)) * uniform_pdf(cfg, (x_a, y_a))

    return total / (cfg.analysis.trapezoidal_res**2)
    
def gen_expected_weighting_lookup(cfg):
    """
    Generates a lookup table for expected weighting values ω for nodes at positions xᵤ, yᵤ to allow for one-time calculation.
    """
    coord_vals = np.linspace(0, 1, cfg.analysis.omega_lookup_res)
    omega_lookup = np.zeros((cfg.analysis.omega_lookup_res, cfg.analysis.omega_lookup_res))

    for i, x_u in enumerate(coord_vals):
        for j, y_u in enumerate(coord_vals):
            omega_lookup[i][j] = trapezoidal_expected_weighting_for_v(cfg, x_u, y_u)

    return omega_lookup, coord_vals

# --- Pairwise Connection Probability --- 

def find_closest_coords(cfg:Config, coord_vals, x, y):
    return np.argmin([abs(c - x) for c in coord_vals]), np.argmin([abs(c - y) for c in coord_vals])

def mc_pairwise_connection_prob(cfg: Config, x_u, y_u, x_v, y_v, omega_coord_vals, omega_lookup, K):
    """ 
    Monte-Carlo approximation of the probability that two nodes at positions (xᵤ, yᵤ) and (xᵥ, yᵥ) connect. 
    """
    total = 0

    # Find expected weighting for u
    i_x, i_y = find_closest_coords(cfg, omega_coord_vals, x_u, y_u)
    exp_w_u = float(omega_lookup[i_x][i_y])

    # Find expected weighting for v
    i_x, i_y = find_closest_coords(cfg, omega_coord_vals, x_v, y_v)
    exp_w_v = float(omega_lookup[i_x][i_y])

    for _ in range(cfg.analysis.mc_n_trials):
        # Generate a random (uniform) affiliation position.
        x_a, y_a = (random.uniform(0,1), random.uniform(0,1))

        # Find connection probability for u to a
        w_u = connection_f(cfg, (x_a, y_a), (x_u, y_u))
        c_u = w_u / ((K-1)*exp_w_u + w_u) 

        # Find connection probability for v
        w_v = connection_f(cfg, (x_a, y_a), (x_v, y_v))
        c_v = w_v / ((K-1)*exp_w_v + w_v) 

        # u and v connect if they both connect to a.
        total += c_u * c_v

    return total / cfg.analysis.mc_n_trials

def trapezoidal_pairwise_connection_prob(cfg: Config, x_u, y_u, x_v, y_v, omega_coord_vals, omega_lookup, K):
    """ 
    Trapezoidal integral of the probability that two nodes at positions (xᵤ, yᵤ) and (xᵥ, yᵥ) connect. 
    """
    total = 0
    coord_vals = np.linspace(0, 1, cfg.analysis.trapezoidal_res)

    # Find expected weighting for u
    i_x, i_y = find_closest_coords(cfg, omega_coord_vals, x_u, y_u)
    exp_w_u = float(omega_lookup[i_x][i_y])

    # Find expected weighting for v
    i_x, i_y = find_closest_coords(cfg, omega_coord_vals, x_v, y_v)
    exp_w_v = float(omega_lookup[i_x][i_y])

    for x_a in coord_vals:
        for y_a in coord_vals:
            # Find connection probability for u to a
            w_u = connection_f(cfg, (x_a, y_a), (x_u, y_u))
            c_u = w_u / ((K-1)*exp_w_u + w_u)

            # Find connection probability for v
            w_v = connection_f(cfg, (x_a, y_a), (x_v, y_v))
            c_v = w_v / ((K-1)*exp_w_v + w_v)  
        
            # u and v connect if they both connect to a.
            total += c_u * c_v

    return total / (cfg.analysis.trapezoidal_res**2)

def p_delta(cfg, delta, omega_coord_vals, omega_lookup, K):
    """
    Monte-Carlo approximation of the probability of two nodes distance δ apart connecting.
    """
    total = 0
    for _ in range(cfg.analysis.mc_n_trials):
        # Sample uniformly from the set of points δ apart inside the unit square.
        x_v = -1
        y_v = -1 
        while not 0 <= x_v <= 1 or not 0 <= y_v <= 1:
            x_u, y_u = (random.uniform(0,1), random.uniform(0,1))
            theta = random.uniform(0,2 * math.pi) # Choose θ uniformly 
            x_v, y_v = (x_u + delta * math.cos(theta), y_u + delta * math.sin(theta)) # Find the point v which is δ away from u and 

        total += trapezoidal_pairwise_connection_prob(cfg, x_u, y_u, x_v, y_v, omega_coord_vals, omega_lookup, K)

    return total / cfg.analysis.mc_n_trials

# --- Voronoi Case ---

def catchment_area(cfg, u_pos, v_pos, a_pos):
    """
    Calculates the area of the union of the two circles centred on u_pos = (x_u, y_u) and v_pos = (x_v, y_v) passing through a_pos = (x_a, y_a).
    This represents the area of S which has to be empty for a to be the closest affiliation to both u and v. 
    ----------
    cfg : Config
        Configuration object containing network and connection parameters.

    u_pos : tuple[float, float]
        The (x, y) position of the node u in the embedding space.

    v_pos : tuple[float, float]
        The (x, y) position of the node u in the embedding space.

    a_pos : tuple[float, float]
        The (x, y) position of the node u in the embedding space.

    Returns
    -------
    area : float
        The area of the union of the circles centred on u and v through a. 
    """
    # δ is the distance between u and v.
    delta = d(u_pos, v_pos, cfg)

    # The radii of the circles are d(u, a) and d(v, a)
    r_u = d(a_pos, u_pos, cfg)
    r_v = d(a_pos, v_pos, cfg)

    # WLOG let r₁ be greater than r₂
    r1, r2 = max(r_u, r_v), min(r_u, r_v)

    # Calculate the area of the circles as πr²
    circle_area_1 = math.pi * (r1**2)
    circle_area_2 = math.pi * (r2**2)
    
    # If r₁ + r₂ = δ then the circles meet only at a, and the area of the union is the sum of the area of the circles.
    if math.isclose(r1+r2, delta):
        return circle_area_1 + circle_area_2
    # If r₁ - r₂ = δ then the smaller circle is contained entirely by the larger one, and the area is the area of the larger circle.
    # If δ = 0 then the circles are centred on the same point and the area is the area of the larger circle.
    elif math.isclose(r1-r2, delta) or math.isclose(delta, 0):
        return circle_area_1
    
    # δₘₚ is the distance to the midpoint of the radical line of the two circles from the centre of the larger circle.
    distance_to_midpoint = (r1**2 - r2**2)/(2*delta) + (delta/2)
    # h is the length of the radical line.
    h = 2 * math.sqrt(r1**2 - distance_to_midpoint**2)

    # θ₁ is the angle from the centre of the bigger circle to the points of intersection of the two circles.
    theta_1 = 2 * math.asin(h/(2*r1))
    # The area of the segment with radius r and angle θ is (r²/2)*(θ - sin θ)
    segment_area_1 = (r1**2)/2 * (theta_1 - math.sin(theta_1))

    # If the centre of the smaller circle is on the radical line, then the area is the area of the large circle + half the area of the small circle, minus the overlapping segment.
    if math.isclose(distance_to_midpoint, delta):
        return circle_area_1 + (circle_area_2/2) - segment_area_1

    try:
        theta_2 = 2 * math.asin(h/(2*r2))
    except:
        print(u_pos, v_pos, a_pos, h, r2, h/(2*r2))


    # θ₂ is the angle from the centre of the smaller circle to the points of intersection of the two circles.
    theta_2 = 2 * math.asin(min(h/(2*r2), 1))
    # The area of the segment with radius r and angle θ is (r²/2)*(θ - sin θ)
    segment_area_2 = (r2**2)/2 * (theta_2 - math.sin(theta_2))

    # If the radical line is inbetween the circle centres, the area is the sum of the area of the circles, minus the overlapping segments.
    if distance_to_midpoint < delta:
        return circle_area_1 + circle_area_2 - segment_area_1 - segment_area_2
    # Otherwise, the area is the area of the large circle and the small segment, minus the overlapping large segment.
    else:
        return circle_area_1 + segment_area_2 - segment_area_1 
    
def mc_voronoi_pairwise_connection_prob(cfg, u_pos, v_pos, K):
    """
    Monte-Carlo approximation of the probability that two nodes at positions u_pos = (xᵤ, yᵤ) and v_pos = (xᵥ, yᵥ) connect in the Voronoi scheme.
    """
    total = 0
    for _ in range(cfg.analysis.mc_n_trials):
        a_pos = (random.uniform(0,1), random.uniform(0,1))
        area = catchment_area(cfg, u_pos, v_pos, a_pos)
        void_prob = math.e ** (-K*area)
        total += void_prob
    return K * total / cfg.analysis.mc_n_trials

def trapezoidal_voronoi_pairwise_connection_prob(cfg, u_pos, v_pos, K):
    """
    Trapezoidal approximation of the probability that two nodes at positions u_pos = (xᵤ, yᵤ) and v_pos = (xᵥ, yᵥ) connect in the Voronoi scheme.
    """
    total = 0
    n = int(math.sqrt(cfg.analysis.mc_n_trials))
    for x_a in np.linspace(0, 1, n):
        for y_a in np.linspace(0, 1, n):
            a_pos = (x_a, y_a)
            area = catchment_area(cfg, u_pos, v_pos, a_pos)
            void_prob = math.e ** (-(K-1)*area)
            total += void_prob
    return K * total / (n**2)

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
