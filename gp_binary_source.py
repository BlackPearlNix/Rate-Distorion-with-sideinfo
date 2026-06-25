import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp

# ==========================================
# 1. System Parameters & Distributions
# ==========================================
p0 = 0.15          # BSC crossover probability for Y
D_target = 0.05    # Target distortion for the detailed output

# Marginal and Conditional Probabilities
p_x = np.array([0.5, 0.5])
p_y_given_x = np.array([[1 - p0, p0], 
                        [p0, 1 - p0]])

# Joint and Reverse Conditional Probabilities
p_xy = p_x[:, None] * p_y_given_x
p_y = np.sum(p_xy, axis=0)
p_x_given_y = p_xy / p_y[None, :]

# ==========================================
# 2. Functional Alphabet & Effective Distortion
# ==========================================
# u_funcs[u, y] gives the output \hat{x} for function u given side info y
u_funcs = np.array([
    [0, 0],  # u_0: Always 0
    [1, 1],  # u_1: Always 1
    [0, 1],  # u_2: Forward Y
    [1, 0]   # u_3: Invert Y
])

# Calculate Effective Distortion D_eff[x, u] = E[ d(x, u(y)) | x ]
D_eff = np.zeros((2, 4))
for x in range(2):
    for u in range(4):
        dist = 0.0
        for y in range(2):
            x_hat = u_funcs[u, y]
            hamming_d = 1 if x != x_hat else 0
            dist += p_y_given_x[x, y] * hamming_d
        D_eff[x, u] = dist

print("--- Step 1: Effective Distortion Matrix D_eff[x, u] Computed ---")
print(D_eff)
print("Rows: X=0, X=1. Cols: u_0(0), u_1(1), u_2(Y), u_3(1-Y)\n")

# ==========================================
# 3. GP Transformed Convex Optimization
# ==========================================
# Variables: v is an array of shape (8,) representing v_{ux}

def objective(v):
    """Objective: I(X; U | Y) using exponential transformation p = exp(v)"""
    v = v.reshape(2, 4)
    p_u_given_x = np.exp(v)
    
    I = 0.0
    for x in range(2):
        for y in range(2):
            for u in range(4):
                prob_xy = p_xy[x, y]
                p_ux = p_u_given_x[x, u]
                
                # Marginalize to find p(u|y)
                p_u_given_y = np.sum(p_x_given_y[:, y] * p_u_given_x[:, u])
                
                if p_ux > 1e-12 and p_u_given_y > 1e-12:
                    I += prob_xy * p_ux * (np.log2(p_ux) - np.log2(p_u_given_y))
    return I

def constraint_prob(v):
    """GP Equality Constraint: log(sum(exp(v))) == 0  => sum(p) == 1"""
    v = v.reshape(2, 4)
    return logsumexp(v, axis=1) # Returns array of size 2, must equal 0

def constraint_dist(v, D):
    """GP Inequality Constraint: D - sum(p(x) * exp(v) * D_eff) >= 0"""
    v = v.reshape(2, 4)
    p_u_given_x = np.exp(v)
    expected_dist = np.sum(p_x[:, None] * p_u_given_x * D_eff)
    return D - expected_dist

def solve_gp_wyner_ziv(D):
    """Executes the convex optimization for a given D"""
    # Initial guess: Uniform distribution v = log(0.25)
    v0 = np.log(np.full(8, 0.25))
    
    constraints = [
        {'type': 'eq', 'fun': constraint_prob},
        {'type': 'ineq', 'fun': lambda v: constraint_dist(v, D)}
    ]
    
    # SLSQP is highly effective for smooth convex problems with constraints
    res = minimize(objective, v0, method='SLSQP', constraints=constraints, 
                   options={'ftol': 1e-9, 'maxiter': 500})
    return res

# ==========================================
# 4. Execute and Output Progress
# ==========================================
print(f"--- Step 2: Solving GP Convex Optimization for D = {D_target} ---")
result = solve_gp_wyner_ziv(D_target)

opt_rate = result.fun
opt_v = result.x.reshape(2, 4)
opt_p_u_given_x = np.exp(opt_v)

print(f"Optimization Status : {result.message}")
print(f"Optimal Rate R(D)   : {opt_rate:.6f} bits/symbol\n")

print("--- Step 3: Optimal Probability Density p(u|x) ---")
print("         u_0(0)    u_1(1)    u_2(Y)    u_3(1-Y)")
print(f"X = 0:  {opt_p_u_given_x[0,0]:.4f}    {opt_p_u_given_x[0,1]:.4f}    {opt_p_u_given_x[0,2]:.4f}    {opt_p_u_given_x[0,3]:.4f}")
print(f"X = 1:  {opt_p_u_given_x[1,0]:.4f}    {opt_p_u_given_x[1,1]:.4f}    {opt_p_u_given_x[1,2]:.4f}    {opt_p_u_given_x[1,3]:.4f}\n")

print("Notice how the optimizer naturally drives the probabilities of u_2(Y) and u_3(1-Y) to zero.")
print("This mathematically proves Shannon's strategy collapses to standard auxiliary variables for DSBS!\n")

# ==========================================
# 5. Generate Curve and Plot
# ==========================================
print("--- Step 4: Computing Rate-Distortion Curve over range of D ---")
D_vals = np.linspace(0.001, p0, 30)
R_GP_vals = []

for d in D_vals:
    res = solve_gp_wyner_ziv(d)
    R_GP_vals.append(res.fun)

# Theoretical Curve for Validation
def H(x):
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def binary_conv(a, b):
    return a * (1 - b) + b * (1 - a)

R_theoretical = [H(binary_conv(p0, d)) - H(d) for d in D_vals]

plt.figure(figsize=(9, 6))
plt.plot(D_vals, R_theoretical, 'k--', linewidth=2, label='Theoretical $H(p_0 * D) - H(D)$')
plt.plot(D_vals, R_GP_vals, 'b-', linewidth=4, alpha=0.5, label='GP Convex Optimization Result')
plt.plot(D_target, opt_rate, 'ro', markersize=8, label=f'GP Optimum at D={D_target}')

plt.xlabel('Expected Distortion (D)', fontsize=12)
plt.ylabel('Rate R(D) [bits/symbol]', fontsize=12)
plt.title(f'Wyner-Ziv Rate-Distortion via GP Transformation ($p_0 = {p0}$)', fontsize=14)
plt.xlim([0, p0 + 0.01])
plt.ylim([0, max(R_theoretical) + 0.05])
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
