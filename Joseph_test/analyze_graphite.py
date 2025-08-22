#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING AND CURATION
# ============================================

# Load the data
data = pd.read_csv('graphite.csv', header=None)
x_raw = data.iloc[:, 0].values  # Lithium molar ratio
V_raw = data.iloc[:, 1].values  # Voltage in volts

print("="*60)
print("DATA CURATION AND PREPROCESSING")
print("="*60)
print(f"Raw data points: {len(x_raw)}")

# Step 1: Remove duplicates and sort
df = pd.DataFrame({'x': x_raw, 'V': V_raw})
df = df.drop_duplicates(subset=['x'], keep='first')
df = df.sort_values('x')
x_clean = df['x'].values
V_clean = df['V'].values
print(f"After removing duplicates: {len(x_clean)} points")

# Step 2: Handle non-monotonic regions by averaging nearby points
tolerance = 1e-6
x_monotonic = []
V_monotonic = []
i = 0
while i < len(x_clean):
    current_x = x_clean[i]
    current_V = [V_clean[i]]
    j = i + 1
    
    # Collect all points within tolerance
    while j < len(x_clean) and abs(x_clean[j] - current_x) < tolerance:
        current_V.append(V_clean[j])
        j += 1
    
    # Average the voltage values for nearby x values
    x_monotonic.append(current_x)
    V_monotonic.append(np.mean(current_V))
    i = j

x_monotonic = np.array(x_monotonic)
V_monotonic = np.array(V_monotonic)
print(f"After ensuring monotonicity: {len(x_monotonic)} points")

# Step 3: Interpolate to uniform grid for better numerical stability
n_points = 200  # Increased density for better resolution
x_uniform = np.linspace(x_monotonic.min(), x_monotonic.max(), n_points)

# Use cubic spline for smooth interpolation
spline = UnivariateSpline(x_monotonic, V_monotonic, k=3, s=1e-4)
V_uniform = spline(x_uniform)

# Step 4: Apply smoothing to reduce noise
V_smooth = savgol_filter(V_uniform, window_length=21, polyorder=3)

# Choose final data
x = x_uniform
V = V_smooth

print(f"Final interpolated points: {len(x)}")
print(f"Lithium ratio range: [{x.min():.3f}, {x.max():.3f}]")
print(f"Voltage range: [{V.min():.3f}, {V.max():.3f}] V")

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Compute derivatives using multiple methods and average
dx = np.gradient(x)
dV_dx_gradient = np.gradient(V, x)
dV_dx_spline = spline.derivative()(x)
dV_dx = (dV_dx_gradient + dV_dx_spline) / 2  # Average for robustness

# Compute second derivative for phase transition detection
d2V_dx2 = np.gradient(dV_dx, x)

# Detect phase transitions from peaks in second derivative
threshold = np.percentile(np.abs(d2V_dx2), 85)
transition_indices = np.where(np.abs(d2V_dx2) > threshold)[0]
transition_points = x[transition_indices]

print(f"Detected {len(transition_points)} potential phase transitions")
if len(transition_points) > 0:
    print(f"Transition points at x ≈ {transition_points[:5]}")  # Show first 5

# ============================================
# DATA PREPARATION FOR SINDY
# ============================================

# Reshape for PySINDy
X = x.reshape(-1, 1)
V_data = V.reshape(-1, 1)
dV_dx_data = dV_dx.reshape(-1, 1)

# Create pseudo-time (using x as independent variable)
t = x.copy()

# Split data for validation
X_train, X_test, V_train, V_test, t_train, t_test = train_test_split(
    X, V_data, t, test_size=0.2, random_state=42
)

# Ensure train and test are sorted
train_idx = np.argsort(t_train)
X_train = X_train[train_idx]
V_train = V_train[train_idx]
t_train = t_train[train_idx]

test_idx = np.argsort(t_test)
X_test = X_test[test_idx]
V_test = V_test[test_idx]
t_test = t_test[test_idx]

# ============================================
# OPTIMIZED LIBRARY CREATION
# ============================================

print("\n" + "="*60)
print("BUILDING PHYSICS-INFORMED LIBRARY")
print("="*60)

# Create streamlined physics-inspired library
library_functions = [
    lambda x: x,                                      # Linear
    lambda x: x**2,                                   # Quadratic
    lambda x: x**3,                                   # Cubic
    lambda x: np.log(x + 0.01),                      # Log term (Nernst)
    lambda x: np.log(1 - x + 0.01),                  # Log(1-x) term
    lambda x: x * np.log(x + 0.01),                  # Entropy term
    lambda x: (1-x) * np.log(1 - x + 0.01),         # Entropy term
    lambda x: np.exp(-5*x),                          # Exponential decay
    lambda x: 1/(x + 0.1),                           # Inverse term
    lambda x: 1/(1 - x + 0.1),                       # Inverse (1-x)
    lambda x: np.tanh(5*(x - 0.5)),                  # Sigmoid-like transition
]

library_function_names = [
    lambda x: x,
    lambda x: x + "²",
    lambda x: x + "³",
    lambda x: "ln(" + x + "+0.01)",
    lambda x: "ln(1-" + x + "+0.01)",
    lambda x: x + "·ln(" + x + ")",
    lambda x: "(1-" + x + ")·ln(1-" + x + ")",
    lambda x: "exp(-5" + x + ")",
    lambda x: "1/(" + x + "+0.1)",
    lambda x: "1/(1-" + x + "+0.1)",
    lambda x: "tanh(5(" + x + "-0.5))",
]

custom_library = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names
)

# Add polynomial library
poly_library = ps.PolynomialLibrary(degree=4, include_bias=True)

# Combine libraries
combined_library = poly_library + custom_library

# ============================================
# MODEL FITTING WITH CROSS-VALIDATION
# ============================================

print("\n" + "="*60)
print("SINDY MODEL FITTING")
print("="*60)

# Try multiple threshold values for STLSQ
thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
best_model = None
best_score = -np.inf
best_threshold = None

for threshold in thresholds:
    try:
        optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5, normalize_columns=True)
        model = ps.SINDy(
            feature_library=combined_library,
            optimizer=optimizer
        )
        
        # Fit on training data
        model.fit(X_train, x_dot=V_train, t=t_train, feature_names=['x'])
        
        # Evaluate on test data
        V_pred = model.predict(X_test)
        score = r2_score(V_test, V_pred)
        
        # Count non-zero coefficients (sparsity)
        n_nonzero = np.count_nonzero(model.coefficients())
        
        # Balance between accuracy and sparsity
        adjusted_score = score - 0.001 * n_nonzero
        
        if adjusted_score > best_score and n_nonzero < 20:  # Limit complexity
            best_score = adjusted_score
            best_model = model
            best_threshold = threshold
            
        print(f"Threshold {threshold:.4f}: R² = {score:.4f}, Terms = {n_nonzero}")
        
    except Exception as e:
        print(f"Threshold {threshold:.4f}: Failed - {str(e)}")

if best_model is not None:
    print(f"\nBest model: threshold = {best_threshold:.4f}, adjusted score = {best_score:.4f}")
    print("\nDiscovered equation for V(x):")
    best_model.print(lhs=['V'])
    
    # Full dataset prediction for visualization
    V_pred_full = best_model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(V_data, V_pred_full))
    mae = mean_absolute_error(V_data, V_pred_full)
    r2 = r2_score(V_data, V_pred_full)
    
    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.4f} V")
    print(f"  MAE:  {mae:.4f} V")
    print(f"  R²:   {r2:.4f}")

# ============================================
# DIFFERENTIAL CAPACITY ANALYSIS
# ============================================

print("\n" + "="*60)
print("DIFFERENTIAL CAPACITY ANALYSIS")
print("="*60)

# Prepare data for differential modeling
X_diff = np.column_stack([x, V])

# Simplified differential library
diff_library = ps.PolynomialLibrary(degree=2, include_bias=True)

try:
    diff_model = ps.SINDy(
        feature_library=diff_library,
        optimizer=ps.STLSQ(threshold=0.1, alpha=1e-2)
    )
    
    diff_model.fit(X_diff, x_dot=dV_dx_data, t=t, feature_names=['x', 'V'])
    print("\nDifferential equation for dV/dx:")
    diff_model.print(lhs=['dV/dx'])
    
    dV_dx_pred = diff_model.predict(X_diff)
    diff_r2 = r2_score(dV_dx_data, dV_dx_pred)
    print(f"Differential model R² = {diff_r2:.4f}")
except Exception as e:
    print(f"Differential modeling failed: {str(e)}")
    dV_dx_pred = None

# ============================================
# PHASE-AWARE MODELING
# ============================================

print("\n" + "="*60)
print("PHASE-AWARE ANALYSIS")
print("="*60)

# Identify distinct voltage plateaus
dV_threshold = 0.1  # V change threshold
plateaus = []
i = 0
while i < len(V) - 10:
    if abs(V[i+10] - V[i]) < dV_threshold:
        start_x = x[i]
        while i < len(V) - 1 and abs(V[i+1] - V[i]) < dV_threshold/10:
            i += 1
        end_x = x[min(i, len(x)-1)]
        if end_x - start_x > 0.05:  # Minimum plateau width
            plateaus.append((start_x, end_x, np.mean(V[np.where((x >= start_x) & (x <= end_x))])))
    i += 1

print(f"Identified {len(plateaus)} voltage plateaus:")
for i, (start, end, v_mean) in enumerate(plateaus[:3]):  # Show first 3
    print(f"  Plateau {i+1}: x ∈ [{start:.3f}, {end:.3f}], V ≈ {v_mean:.3f} V")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Data and Model Fit
ax1 = axes[0, 0]
ax1.scatter(x_raw, V_raw, c='lightblue', alpha=0.5, s=10, label='Raw data')
ax1.plot(x, V, 'b-', linewidth=1, alpha=0.7, label='Smoothed data')
if best_model is not None:
    # Convert to numpy array if needed
    V_pred_plot = np.array(V_pred_full).flatten()
    ax1.plot(x, V_pred_plot, 'r-', linewidth=2, label=f'SINDy (R²={r2:.3f})')
ax1.set_xlabel('Lithium ratio x')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('Equilibrium Potential Model')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
if best_model is not None:
    V_pred_plot = np.array(V_pred_full).flatten()
    residuals = V - V_pred_plot
    ax2.scatter(x, residuals, c='green', alpha=0.6, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(x, -rmse, rmse, alpha=0.2, color='red', label=f'±RMSE ({rmse:.3f}V)')
    ax2.set_xlabel('Lithium ratio x')
    ax2.set_ylabel('Residuals (V)')
    ax2.set_title('Model Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Differential Capacity
ax3 = axes[0, 2]
ax3.plot(x, -dV_dx, 'purple', linewidth=2, label='−dV/dx (capacity)')
if dV_dx_pred is not None:
    dV_dx_pred_plot = np.array(dV_dx_pred).flatten()
    ax3.plot(x, -dV_dx_pred_plot, 'orange', linewidth=1, label='Model −dV/dx')
ax3.set_xlabel('Lithium ratio x')
ax3.set_ylabel('−dV/dx')
ax3.set_title('Differential Capacity')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Phase Transitions
ax4 = axes[1, 0]
ax4.plot(x, V, 'b-', linewidth=2)
for trans_x in transition_points[:10]:  # Show first 10 transitions
    ax4.axvline(x=trans_x, color='red', alpha=0.3, linestyle='--')
for start, end, v_mean in plateaus:
    ax4.axhspan(v_mean - 0.01, v_mean + 0.01, 
                xmin=(start-x.min())/(x.max()-x.min()),
                xmax=(end-x.min())/(x.max()-x.min()),
                alpha=0.2, color='green')
ax4.set_xlabel('Lithium ratio x')
ax4.set_ylabel('Voltage (V)')
ax4.set_title('Phase Transitions & Plateaus')
ax4.grid(True, alpha=0.3)

# Plot 5: Feature Importance
ax5 = axes[1, 1]
if best_model is not None:
    coefficients = best_model.coefficients()[0]
    feature_names = best_model.get_feature_names()
    
    # Get non-zero coefficients
    nonzero_idx = np.where(np.abs(coefficients) > 1e-10)[0]
    nonzero_coefs = coefficients[nonzero_idx]
    nonzero_names = [feature_names[i] for i in nonzero_idx]
    
    # Sort by magnitude
    sort_idx = np.argsort(np.abs(nonzero_coefs))[::-1][:15]  # Top 15
    
    colors = ['green' if c > 0 else 'red' for c in nonzero_coefs[sort_idx]]
    ax5.barh(range(len(sort_idx)), np.abs(nonzero_coefs[sort_idx]), color=colors, alpha=0.7)
    ax5.set_yticks(range(len(sort_idx)))
    ax5.set_yticklabels([nonzero_names[i] for i in sort_idx], fontsize=8)
    ax5.set_xlabel('|Coefficient|')
    ax5.set_title('Important Features')
    ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Second Derivative (Phase Detection)
ax6 = axes[1, 2]
ax6.plot(x, d2V_dx2, 'k-', linewidth=1, label='d²V/dx²')
ax6.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax6.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
ax6.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
ax6.set_xlabel('Lithium ratio x')
ax6.set_ylabel('d²V/dx²')
ax6.set_title('Second Derivative Analysis')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphite_sindy_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PHYSICAL INTERPRETATION
# ============================================

print("\n" + "="*60)
print("PHYSICAL INTERPRETATION")
print("="*60)

if best_model is not None:
    coefficients = best_model.coefficients()[0]
    feature_names = best_model.get_feature_names()
    
    # Categorize terms
    print("\nDominant physical contributions:")
    print("-" * 40)
    
    # Find significant terms
    significant = [(name, coef) for name, coef in zip(feature_names, coefficients) 
                   if abs(coef) > 0.001]
    
    # Categorize by type
    entropic_terms = [(n, c) for n, c in significant if 'ln' in n]
    polynomial_terms = [(n, c) for n, c in significant if any(p in n for p in ['x', '²', '³']) and 'ln' not in n and 'exp' not in n]
    exponential_terms = [(n, c) for n, c in significant if 'exp' in n]
    transition_terms = [(n, c) for n, c in significant if 'tanh' in n]
    
    if entropic_terms:
        print("\n1. ENTROPIC/NERNST CONTRIBUTIONS:")
        print("   (Related to configurational entropy and electrochemical potential)")
        for name, coef in entropic_terms[:3]:
            print(f"   • {name}: {coef:+.4f}")
    
    if polynomial_terms:
        print("\n2. ACTIVITY/INTERACTION TERMS:")
        print("   (Related to lithium-lithium interactions and non-ideal behavior)")
        for name, coef in polynomial_terms[:5]:
            print(f"   • {name}: {coef:+.4f}")
    
    if exponential_terms:
        print("\n3. PHASE TRANSITION TERMS:")
        print("   (Rapid changes in structure/ordering)")
        for name, coef in exponential_terms[:3]:
            print(f"   • {name}: {coef:+.4f}")
    
    if transition_terms:
        print("\n4. SIGMOID TRANSITION TERMS:")
        print("   (Smooth transitions between phases)")
        for name, coef in transition_terms[:3]:
            print(f"   • {name}: {coef:+.4f}")
    
    # Physical insights
    print("\n" + "="*60)
    print("KEY PHYSICAL INSIGHTS:")
    print("="*60)
    
    if len(plateaus) > 0:
        print(f"\n✓ {len(plateaus)} distinct voltage plateaus identified")
        print("  → Indicates staged lithium intercalation")
    
    if len(transition_points) > 0:
        print(f"\n✓ {len(transition_points)} phase transitions detected")
        print("  → Structural changes during lithiation/delithiation")
    
    if entropic_terms:
        print("\n✓ Logarithmic terms present in model")
        print("  → Configurational entropy is significant")
        print("  → System follows modified Nernst equation")
    
    if polynomial_terms:
        print("\n✓ Polynomial terms indicate non-ideal behavior")
        print("  → Lithium-lithium interactions are important")
        print("  → Activity coefficients deviate from unity")
    
    # Thermodynamic interpretation
    print("\n" + "="*60)
    print("THERMODYNAMIC INTERPRETATION:")
    print("="*60)
    
    # Estimate entropy contribution
    entropy_contribution = sum(abs(c) for n, c in entropic_terms) if entropic_terms else 0
    interaction_contribution = sum(abs(c) for n, c in polynomial_terms) if polynomial_terms else 0
    total_contribution = entropy_contribution + interaction_contribution
    
    if total_contribution > 0:
        print(f"\nRelative contributions to voltage:")
        print(f"  • Entropic effects: {100*entropy_contribution/total_contribution:.1f}%")
        print(f"  • Interaction effects: {100*interaction_contribution/total_contribution:.1f}%")
    
    print("\n" + "="*60)
    print("MODEL EQUATION (simplified):")
    print("="*60)
    
    # Print simplified equation with only significant terms
    equation_parts = []
    for name, coef in significant[:8]:  # Top 8 terms
        if abs(coef) > 0.01:
            equation_parts.append(f"{coef:+.3f}·{name}")
    
    if equation_parts:
        print(f"\nV(x) ≈ {' '.join(equation_parts)}")

print("\n" + "="*60)
print("Analysis complete! Results saved to 'graphite_sindy_analysis.png'")
print("="*60)