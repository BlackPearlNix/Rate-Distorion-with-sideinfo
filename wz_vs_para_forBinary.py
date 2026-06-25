import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# functions
# *******************

p0 = 0.15  #crossover probability of Y
D_target = 0.05  # targeted distortion constraint

def H(x):			        #binary entropy fnction
    x = np.clip(x, 1e-10, 1 - 1e-10) #avoid log(0)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def binary_conv(a, b):		# (p*d)
    return a * (1 - b) + b * (1 - a)

def R_wz(q, p0):		    #wyner-ziv rate
    return H(binary_conv(p0, q)) - H(q)

def D_dec(q, p0):           # decoder distortion
    return np.minimum(q, p0)


# Wyner-Ziv
# ******************
q_vals = np.linspace(0, p0, 500)    
R_vals = R_wz(q_vals, p0)           

points = np.column_stack((q_vals, R_vals))
points = np.vstack((points, [p0, 0.0])) # append the zero-rate point

# Lower convex envelope (Graham Scan)
hull_indices = []
for i in range(len(points)):
    while len(hull_indices) >= 2:
        p1, p2 = points[hull_indices[-2]], points[hull_indices[-1]]
        p3 = points[i]
        # Cross product to check for convexity (right turn)
        if (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) <= 0:
            hull_indices.pop()
        else:
            break
    hull_indices.append(i)

D_hull = points[hull_indices, 0]
R_hull = points[hull_indices, 1]
WZ_rate_at_D = np.interp(D_target, D_hull, R_hull)



# Parametric Solution
# **********************
 
# to find = [q_a, q_b, q_c, alpha_a, alpha_b, alpha_c]
def objective(vars):
    q = vars[0:3]
    alpha = vars[3:6]
    return np.sum(alpha * R_wz(q, p0))

def constraint_distortion(vars):
    q = vars[0:3]
    alpha = vars[3:6]
    # Using optimal decoder distortion D_dec
    return D_target - np.sum(alpha * D_dec(q, p0))

def constraint_probability(vars):
    alpha = vars[3:6]
    return np.sum(alpha) - 1.0

# Initial guess:
x0 = [0.01, D_target, 0.5, 0.33, 0.33, 0.34]

# Bounds
bounds = [(0, 0.5), (0, 0.5), (0, 0.5), (0, 1), (0, 1), (0, 1)]

constraints = [
    {'type': 'ineq', 'fun': constraint_distortion},
    {'type': 'eq', 'fun': constraint_probability}
]

result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
rate_at_D = result.fun
opt_q = result.x[0:3]
opt_alpha = result.x[3:6]


# Output and Comparison
# *********************

print(f"System Parameters")
print(f"Source X ~ Bern(0.5)")
print(f"Side Info Y ~ BSC({p0})")
print(f"Target Distortion D = {D_target}\n")

print(f" Results ")
print(f"Wyner-Ziv Rate : {WZ_rate_at_D:.6f} bits/symbol")
print(f"Parametric Rate: {rate_at_D:.6f} bits/symbol")
print(f"Difference    : {abs(WZ_rate_at_D - rate_at_D):.2e}\n")

print(f"Optimal Parametric Variables")
for i, name in enumerate(['a', 'b', 'c']):
    if opt_alpha[i] > 1e-4:
        eff_D = D_dec(opt_q[i], p0)
        print(f"BSC {name}: q_{name} = {opt_q[i]:.4f}, alpha_{name} = {opt_alpha[i]:.4f}, Distortion = {eff_D:.4f}, Rate = {R_wz(opt_q[i], p0):.4f}")


# Plotting
# ********************
plt.figure(figsize=(9, 6))

# Plot R(D) curve (without time-sharing)
plt.plot(q_vals, R_vals, 'k--', linewidth=1.5, label='$H(p_0 * q) - H(q)$')

# Plot lower convex envelope (with time-sharing)
plt.plot(D_hull, R_hull, 'b-', linewidth=1.5, label='Convex envelope with time sharing')

# Mark the optimal point found by the parametric method
plt.plot(D_target, rate_at_D, 'ro', markersize=5, label="Parametric Optimum at D_target")


plt.xlabel('Distortion (D)', fontsize=12)
plt.ylabel('Rate R(D) [bps]', fontsize=12)
plt.title(f'Binary Source (Bern(0.5), Y ~ BSC({p0}), D_target={D_target})', fontsize=14)
plt.xlim([0, p0 + 0.05])
plt.ylim([0, max(R_vals) + 0.05])
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
