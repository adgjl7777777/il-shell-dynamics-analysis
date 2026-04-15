"""
Compute theoretical soft/hard state duration distributions h_soft(n) and h_hard(n)
using the analytical two-state model (paper §Validation, eqs 10-19).

Parameters are taken from TABLE1 in config.py (fitted from inter-event distributions).
Outputs normalized h_soft and h_hard distributions saved to ./math/ for comparison
with simulation data (Fig 5 of the paper).

Usage:
    python mathing_mean_calc.py <anion> <T_index>
    T_index: 0=298K, 1=353K, 2=373K, 3=423K
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TABLE1

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def compute_power_function(threshold, a_values):
    results = {}
    x_values = np.arange(1, threshold + 1)

    for a in [a_values]:
        power_values = x_values ** (-a)  # y = x^(-a)
        sum_values = np.sum(power_values)  # Function sum
        normalized_values_unpadded = power_values / sum_values
        mean_value = np.sum(normalized_values_unpadded * x_values)  # Mean of normalized function

        if len(power_values) < n_max:
            normalized_values = np.zeros(n_max)
            normalized_values[:len(power_values)] = normalized_values_unpadded
        else:
            normalized_values = normalized_values_unpadded
        results[a] = {
            'sum': sum_values,
            'normalized': normalized_values,
            'mean': mean_value
        }
    
    return results

def compute_g_function(normalized_values, mean_value, n_max):
    # Implements the convolution-based function g_r(n) from eq. 12
    # g_r(n) = Σ_{k=1 to min(Δt, n)} Y(k) * g_r(n-k)
    # with g_r(0) = 1
    
    delta_t = int(np.ceil(mean_value))
    g_values = np.zeros(n_max + 1)
    g_values[0] = 1  # g_r(0) = 1
    
    for n in range(1, n_max + 1):
        g_n = 0
        # Summation over k from 1 to min(delta_t, n)
        for k in range(1, min(delta_t, n) + 1):
            # Y(k) is normalized_values[k-1], g_r(n-k) is g_values[n-k]
            g_n += normalized_values[k-1] * g_values[n - k]
        g_values[n] = g_n
    
    return g_values

def compute_h_function(normalized_values, mean_value, g_values, n_max, n2_max):
    # Implements the recursive relationship for the soft state duration distribution h₁(n)
    # based on equations 13, 14, 15 from the paper.
    # NOTE: Equation 14 in the paper is computationally intensive and may contain typos.
    # This implementation follows the equation as literally as possible.

    delta_t = int(np.ceil(mean_value))
    
    s_delta_t = np.sum(normalized_values[delta_t:])
    s_2delta_t = np.sum(normalized_values[2*delta_t:])

    g_r = np.array(g_values)

    g1_values = []

    # Calculate g₁₁(n) (for m=1) based on eq. 13
    g11_factor = s_2delta_t + (s_delta_t - s_2delta_t) * s_delta_t
    g11 = g11_factor * g_r[:n_max]
    g1_values.append(g11)

    # Recursively calculate g₁m(n) for m >= 2, based on eq. 14
    tol = 1e-10  # early-stop: stop when new term contributes negligibly
    for m in range(1, n2_max):
        g1_prev = g1_values[-1]
        g1m = np.zeros(n_max)
        print(f"Calculating g1, m={m+1}")
        for n_idx in trange(n_max):
            n = n_idx + 1
            for k in range(1, n + 1):
                if k >= len(g_r): continue
                g_r_k = g_r[k]
                for j in range(delta_t + 1, 2 * delta_t + 1):
                    if j > len(normalized_values): continue
                    y_j = normalized_values[j-1]

                    target_idx = n - k - j
                    if target_idx >= 0:
                         g1m[n_idx] += y_j * g_r_k * g1_prev[target_idx]
        g1_values.append(g1m)
        # Early stopping: if this term is negligible relative to accumulated sum
        accumulated = np.sum(g1_values, axis=0)
        if np.max(np.abs(g1m)) < tol * max(np.max(np.abs(accumulated)), 1e-30):
            print(f"  → converged at m={m+1}, stopping early")
            break

    h1_final = np.sum(g1_values, axis=0)
    return h1_final

def wasans_hard(normalized_values, mean_value, n_max, n3_max):
    # Implements the recursive relationship for the hard state duration distribution h₂(n)
    # based on equations 16, 17, 18, 19 from the paper.
    
    delta_t = int(np.ceil(mean_value))
    
    s_delta_t = np.sum(normalized_values[delta_t:])
    
    g2_values = []

    # Calculate g₂₁(n) (for m=1) based on eq. 16 (using g'21 version for n > Δt)
    g21 = np.zeros(n_max)
    for n in range(delta_t + 1, n_max + 1):
        g21[n-1] = normalized_values[n-1] * (1 - s_delta_t)
    g2_values.append(g21)

    # Recursively calculate g₂m(n) for m >= 2, based on eq. 18
    # This is a convolution of g₂,{m-1} with Y(k) for k <= Δt
    kernel = normalized_values[:delta_t]
    
    tol = 1e-10
    for m in range(1, n3_max):
        g2_prev = g2_values[-1]
        g2m = np.convolve(g2_prev, kernel, mode='full')[:n_max]
        g2_values.append(g2m)
        accumulated = np.sum(g2_values, axis=0)
        if np.max(np.abs(g2m)) < tol * max(np.max(np.abs(accumulated)), 1e-30):
            print(f"  → hard converged at m={m+1}, stopping early")
            break

    h2_final = np.sum(g2_values, axis=0)
    return h2_final
            
T_list   = [298, 353, 373, 423]
my_anion = sys.argv[1]
my_T_idx = int(sys.argv[2])           # 0=298K, 1=353K, 2=373K, 3=423K
my_T     = T_list[my_T_idx]

n_threshold, a_values, delta_t_exp = TABLE1[my_anion][my_T]
threshold = n_threshold               # truncation point for power-law Y(n)
n_max = 5000  # Number of h(n) values to compute
n2_max=500    # max soft iterations (early-stop if converged)
n3_max=500    # max hard iterations (early-stop if converged)
results = compute_power_function(threshold, a_values)

h_results = {}
for a, data in results.items():
    delta_t = data['mean']
    g_values = compute_g_function(data['normalized'], delta_t, n_max)
    
    soft_values = compute_h_function(data['normalized'], delta_t, g_values, n_max,n2_max)
    hard_values = wasans_hard(data['normalized'], delta_t,  n_max, n3_max)
    # Normalize results safely
    soft_sum = np.sum(soft_values)
    hard_sum = np.sum(hard_values)
    
    norm_soft = soft_values / soft_sum if soft_sum > 0 else np.zeros(n_max)
    norm_hard = hard_values / hard_sum if hard_sum > 0 else np.zeros(n_max)

    np.savetxt(f"./math/{a}_soft.txt", norm_soft)
    np.savetxt(f"./math/{a}_hard.txt", norm_hard)

    # Calculate ratio
    soft_mean_duration = np.sum(norm_soft * np.arange(1, n_max + 1))
    hard_mean_duration = np.sum(norm_hard * np.arange(1, n_max + 1))
    total_mean_duration = soft_mean_duration + hard_mean_duration
    
    ratio = soft_mean_duration / total_mean_duration if total_mean_duration > 0 else 0

    with open("ratio.txt", "a") as zzz:
        zzz.write(f"{a}, {ratio}\n")
    plt.plot(range(1, n_max + 1), norm_soft, label=f'coeff={a}')

    plt.xlabel('n')
    plt.ylabel("$h_{{{}}}(n)$".format("soft"))
    plt.yscale('log')  # Set y-axis to log scale
    plt.title('Soft duration distribution')
    plt.legend()
    plt.xlim([0,n_max])
    plt.ylim([1e-12,1])
    plt.savefig(f"./math/{a}_soft.png",dpi=600)
    
    plt.cla()
    plt.clf()
    plt.close()
    plt.plot(range(1, n_max + 1), norm_hard, label=f'coeff={a}')

    plt.xlabel('n')
    plt.ylabel("$h_{{{}}}(n)$".format("hard"))
    plt.yscale('log')  # Set y-axis to log scale
    plt.title('Hard duration distribution')
    plt.legend()
    plt.xlim([0,n_max])
    plt.ylim([1e-12,1])
    plt.savefig(f"./math/{a}_hard.png",dpi=600)

    plt.cla()
    plt.clf()
    plt.close()
