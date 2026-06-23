import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# functions
# *******************

K = 3               # Alphabet length 
N = K + 1           # Number of auxiliary variables
p0 = 0.25           # Crossover probability of Y
D_target = 0.15     # Targeted distortion constraint

def H_KSC(q, K):            # K-array entropy function
    q_max = (K - 1) / K
    q = np.clip(q, 1e-10, q_max - 1e-10) # Avoid log(0)
    term1 = -(1 - q) * np.log2(1 - q)
    term2 = -q * np.log2(q / (K - 1))
    return term1 + term2

def kary_conv(a, b, K):     # Cascade of two KSCs
    return a + b - a * b * (K / (K - 1))

def R_wz_kary(q, p0, K):     # Wyner-Ziv rate for a single KSC
    return H_KSC(kary_conv(p0, q, K), K) - H_KSC(q, K)

def D_dec(q, p0): # Optimal decoder distortion
    # Decoder chooses U (distortion q) or Y (distortion p0)
    return np.minimum(q, p0)


# Parametric Solver (finding N -alphas(time sharing prob) and N- distortions, one for each KSC) 
# total 2N parameters
# **********************

def solve_parametric_KSC(D_target, p0, K, N):
    if D_target >= p0:
        return 0.0, np.full(N, p0), np.array([1.0] + [0.0]*(N-1))

    def objective(vars):
        q = vars[0:N]
        alpha = vars[N:2*N]
        return np.sum(alpha * R_wz_kary(q, p0, K))

    def constraint_distortion(vars):
        q = vars[0:N]
        alpha = vars[N:2*N]
        return D_target - np.sum(alpha * D_dec(q, p0))

    def constraint_probability(vars):
        alpha = vars[N:2*N]
        return np.sum(alpha) - 1.0

    q_max = (K - 1) / K
    
    # Initial guess: spreading q evenly across [0.01, q_max], equal weights for alpha
    q_guess = np.linspace(0.01, q_max, N)
    alpha_guess = np.ones(N) / N
    x0 = np.concatenate((q_guess, alpha_guess))

    # Bounds: q in [0, q_max], alpha in [0, 1]
    bounds = [(0, q_max)] * N + [(0, 1)] * N

    constraints = [
        {'type': 'ineq', 'fun': constraint_distortion},
        {'type': 'eq', 'fun': constraint_probability}
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 1000})
    
    opt_rate = result.fun
    opt_q = result.x[0:N]
    opt_alpha = result.x[N:2*N]
    
    return opt_rate, opt_q, opt_alpha


# generating Rate-Distortion curve
# **********************

D_vals = np.linspace(0.001, p0, 50)
R_vals_parametric = []
for D in D_vals:
    rate, _, _ = solve_parametric_KSC(D, p0, K, N)
    R_vals_parametric.append(rate)

#Rate for given distortion
rate_at_D, opt_q, opt_alpha = solve_parametric_KSC(D_target, p0, K, N)

# Generating the single-letter curve (without time-sharing)
q_max = (K - 1) / K
q_single = np.linspace(0, q_max, 200)
R_single = R_wz_kary(q_single, p0, K)
D_single = D_dec(q_single, p0)



# Outputs
# *********************

rate_at_D, opt_q, opt_alpha = solve_parametric_KSC(D_target, p0, K, N)

print(f"--- System Parameters ---")
print(f"Source X ~ Uniform(0 to {K-1}) (K={K})")
print(f"Side Info Y ~ KSC({p0})")
print(f"Number of KSCs (N) = {N}")
print(f"Target Distortion D = {D_target}\n")

print(f"--- Results ---")
print(f"Parametric Optimum Rate: {rate_at_D:.6f} bits/symbol\n")

print(f"--- Optimal Parametric Variables (Active Components) ---")
active_count = 0
for i in range(N):
    if opt_alpha[i] > 1e-4:
        active_count += 1
        eff_D = D_dec(opt_q[i], p0)
        print(f"KSC {i+1}: q = {opt_q[i]:.4f}, alpha = {opt_alpha[i]:.4f}")
        print(f"       -> Effective Distortion = {eff_D:.4f}, Rate = {R_wz_kary(opt_q[i], p0, K):.4f}")



# Plotting
# ********************



plt.figure(figsize=(9, 6))

# Plot single-letter R(D) curve (without time-sharing)
plt.plot(D_single, R_single, 'k--', linewidth=1.5, label=f'Single KSC $H_{{KSC}}(p_0 * q) - H_{{KSC}}(q)$')

# Plot parametric convex envelope (with time-sharing)
plt.plot(D_vals, R_vals_parametric, 'b-', linewidth=1.5, label=f'Parametric Convex Envelope (N={N} KSCs)')

# Mark the optimal point found for the specific D_target
plt.plot(D_target, rate_at_D, 'ro', markersize=5, label=f"Optimum at D={D_target}")


plt.xlabel('Distortion (D)', fontsize=12)
plt.ylabel('Rate R(D) [bits/symbol]', fontsize=12)
plt.title(f'{K}-ary Source (N={N} Parameters)', fontsize=14)
plt.xlim([0, p0 + 0.02])
plt.ylim([-0.05, max(R_single) + 0.1])
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()