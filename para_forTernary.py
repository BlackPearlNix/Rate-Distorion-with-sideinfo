import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# functions
# *******************

p0 = 0.3  # crossover probability of Y
D_target_single = 0.225  # targeted distortion constraint for the single printout

def H_TSC(q):               # Ternary entropy function
    # Clip to avoid log(0). Max crossover for ternary is 2/3 (independent)
    q = np.clip(q, 1e-10, 2/3 - 1e-10) 
    term1 = -(1 - q) * np.log2(1 - q)
    term2 = -q * np.log2(q / 2)
    return term1 + term2

def ternary_conv(a, b):     # Cascade of two TSCs
    return a + b - 1.5 * a * b

def R_wz_ternary(q, p0):    # Wyner-Ziv rate for a single TSC
    return H_TSC(ternary_conv(p0, q)) - H_TSC(q)

def D_dec(q, p0):           # Optimal decoder distortion
    # Decoder chooses U (distortion q) or Y (distortion p0)
    return np.minimum(q, p0)


# Parametric Solution Solver
# **********************

def solve_parametric(D_target, p0):
    """
    Solves the parametric optimization for a given target distortion.
    Variables: vars = [q_a, q_b, q_c, alpha_a, alpha_b, alpha_c]
    """
    if D_target >= p0:
        return 0.0, [p0, p0, p0], [1.0, 0.0, 0.0]

    def objective(vars):
        q = vars[0:3]
        alpha = vars[3:6]
        return np.sum(alpha * R_wz_ternary(q, p0))

    def constraint_distortion(vars):
        q = vars[0:3]
        alpha = vars[3:6]
        return D_target - np.sum(alpha * D_dec(q, p0))

    def constraint_probability(vars):
        alpha = vars[3:6]
        return np.sum(alpha) - 1.0

    # Initial guess: Spread q across domain, including the zero-rate point (q=2/3)
    x0 = [0.01, D_target, 2/3, 0.33, 0.33, 0.34]

    # Bounds: q in [0, 2/3] for ternary, alpha in [0, 1]
    bounds = [(0, 2/3), (0, 2/3), (0, 2/3), (0, 1), (0, 1), (0, 1)]

    constraints = [
        {'type': 'ineq', 'fun': constraint_distortion},
        {'type': 'eq', 'fun': constraint_probability}
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    return result.fun, result.x[0:3], result.x[3:6]


# Generate Rate-Distortion Curve
# **********************

D_vals = np.linspace(0.001, p0, 50)
R_vals_parametric = []

for D in D_vals:
    rate, _, _ = solve_parametric(D, p0)
    R_vals_parametric.append(rate)


# Output for Single Target Distortion
# *********************

rate_at_D, opt_q, opt_alpha = solve_parametric(D_target_single, p0)

print(f"System Parameters")
print(f"Source X ~ Uniform(0, 1, 2)")
print(f"Side Info Y ~ TSC({p0})")
print(f"Target Distortion D = {D_target_single}\n")

print(f" Results ")
print(f"Parametric Rate: {rate_at_D:.6f} bits/symbol\n")

print(f"Optimal Parametric Variables")
for i, name in enumerate(['a', 'b', 'c']):
    if opt_alpha[i] > 1e-4:
        eff_D = D_dec(opt_q[i], p0)
        print(f"TSC {name}: q_{name} = {opt_q[i]:.4f}, alpha_{name} = {opt_alpha[i]:.4f}, Distortion = {eff_D:.4f}, Rate = {R_wz_ternary(opt_q[i], p0):.4f}")


# Plotting
# ********************

# Generate the single-letter curve (without time-sharing) for comparison
q_single = np.linspace(0, p0, 100)
R_single = R_wz_ternary(q_single, p0)

plt.figure(figsize=(9, 6))

# Plot single-letter R(D) curve (without time-sharing)
plt.plot(q_single, R_single, 'k--', linewidth=1.5, label='Single TSC $H_{TSC}(p_0 * q) - H_{TSC}(q)$')

# Plot parametric convex envelope (with time-sharing)
plt.plot(D_vals, R_vals_parametric, 'b-', linewidth=2, label='Parametric Convex Envelope (Time-Shared)')

# Mark the optimal point found for the specific D_target
plt.plot(D_target_single, rate_at_D, 'ro', markersize=6, label=f"Optimum at D={D_target_single}")

plt.xlabel('Distortion (D)', fontsize=12)
plt.ylabel('Rate R(D) [bits/symbol]', fontsize=12)
plt.title(f'Ternary Source (Uniform, Y ~ TSC({p0}))', fontsize=14)
plt.xlim([0, p0 + 0.02])
plt.ylim([0, max(R_single) + 0.05])
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()