import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# =============================================================================
# PART A: STANDARDIZED PCA FROM SCRATCH
# =============================================================================

# Step 1: Load the dataset into a Pandas DataFrame
data = {
    "Student": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14]
}
df = pd.DataFrame(data)
subjects = ["Mathematics", "Statistics", "Programming", "Algorithms"]
n_students = len(df)
n_subjects = len(subjects)

# Step 2: Dimensions and manual descriptive statistics (no df.describe())
print("--- Part A: Standardized PCA from Scratch ---")
print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")

print("Manual Descriptive Statistics:")
print(f"{'Subject':14s} {'Mean':>8s} {'Sample Std':>12s} {'Pop Std':>10s} {'Min':>6s} {'Median':>8s} {'Max':>6s}")
for sub in subjects:
    vals = df[sub].tolist()
    # Mean
    mean_val = sum(vals) / n_students
    # Variances
    sq_diff = sum((x - mean_val) ** 2 for x in vals)
    sample_std = math.sqrt(sq_diff / (n_students - 1))
    pop_std = math.sqrt(sq_diff / n_students)
    # Min and Max
    min_val = min(vals)
    max_val = max(vals)
    # Median using sorted list
    sorted_v = sorted(vals)
    median_val = (sorted_v[2] + sorted_v[3]) / 2.0
    print(f"{sub:14s} {mean_val:8.4f} {sample_std:12.4f} {pop_std:10.4f} {min_val:6.1f} {median_val:8.1f} {max_val:6.1f}")
print()

# Step 3: Calculate mean and population standard deviation manually
means = []
stds = []
for sub in subjects:
    vals = df[sub].tolist()
    m = sum(vals) / n_students
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / n_students)
    means.append(m)
    stds.append(s)

# Step 4: Standardize each subject manually: z = (x - mean) / std
Z = []
for i in range(n_students):
    row = []
    for j in range(n_subjects):
        val = df[subjects[j]].iloc[i]
        z_score = (val - means[j]) / stds[j]
        row.append(z_score)
    Z.append(row)

print("Standardized Data Matrix (Z):")
z_df = pd.DataFrame(Z, index=df["Student"], columns=subjects)
print(z_df.round(4))
print()

# Step 5: Correlation matrix of standardized data (permitted routine)
corr_matrix = np.corrcoef(np.array(Z), rowvar=False)
print("Correlation Matrix:")
corr_df = pd.DataFrame(corr_matrix, index=subjects, columns=subjects)
print(corr_df.round(4))
print()

# Step 6: Sample covariance matrix of standardized data (manual, divisor n - 1)
# Because data is standardized, means of columns are 0.
cov_matrix = []
for j in range(n_subjects):
    cov_row = []
    mean_j = sum(Z[i][j] for i in range(n_students)) / n_students
    for k in range(n_subjects):
        mean_k = sum(Z[i][k] for i in range(n_students)) / n_students
        s = sum((Z[i][j] - mean_j) * (Z[i][k] - mean_k) for i in range(n_students))
        cov_row.append(s / (n_students - 1))
    cov_matrix.append(cov_row)

print("Sample Covariance Matrix of Standardized Data (ddof=1):")
cov_df = pd.DataFrame(cov_matrix, index=subjects, columns=subjects)
print(cov_df.round(4))
print()

# Step 7: Eigenvalues and eigenvectors (permitted routine)
cov_np = np.array(cov_matrix)
eigvals_raw, eigvecs_raw = np.linalg.eigh(cov_np)

# Step 8: Sort eigenvalues and eigenvectors in descending order manually
# Selection sort
indices = list(range(n_subjects))
for i in range(n_subjects):
    max_idx = i
    for j in range(i + 1, n_subjects):
        if eigvals_raw[indices[j]] > eigvals_raw[indices[max_idx]]:
            max_idx = j
    indices[i], indices[max_idx] = indices[max_idx], indices[i]

sorted_eigvals = [eigvals_raw[idx] for idx in indices]
sorted_eigvecs = []
for r in range(n_subjects):
    sorted_eigvecs.append([eigvecs_raw[r, idx] for idx in indices])

print("Sorted Eigenvalues:")
for i, val in enumerate(sorted_eigvals):
    print(f"  Component {i+1}: {val:.6f}")
print()

# Step 9: Select the first two eigenvectors (W has shape 4 x 2)
W = []
for r in range(n_subjects):
    W.append([sorted_eigvecs[r][0], sorted_eigvecs[r][1]])

print("Selected First Two Eigenvectors (PC1 and PC2):")
w_df = pd.DataFrame(W, index=subjects, columns=["PC1", "PC2"])
print(w_df.round(6))
print()

# Step 10: Project standardized data onto the two components using manual loops
coords_scratch = []
for i in range(n_students):
    student_coords = []
    for c in range(2):
        # dot product of student standardized row with eigenvector column
        val = sum(Z[i][j] * W[j][c] for j in range(n_subjects))
        student_coords.append(val)
    coords_scratch.append(student_coords)

print("Transformed Coordinates (From Scratch):")
coords_scratch_df = pd.DataFrame(coords_scratch, index=df["Student"], columns=["PC1", "PC2"])
print(coords_scratch_df.round(6))
print()

# Step 11: Explained variance ratio and cumulative variance manually
total_variance = sum(sorted_eigvals)
var_ratios = [ev / total_variance for ev in sorted_eigvals]
cum_var = []
running_sum = 0.0
for r in var_ratios:
    running_sum += r
    cum_var.append(running_sum)

print("Explained Variance Ratios (From Scratch):")
for i in range(len(var_ratios)):
    print(f"  PC{i+1}: {var_ratios[i]:.6f} ({var_ratios[i]*100:.2f}%), Cumulative: {cum_var[i]*100:.2f}%")
print()


# =============================================================================
# PART B: STANDARDIZED PCA USING SCIKIT-LEARN
# =============================================================================
print("--- Part B: Standardized PCA Using Scikit-Learn ---")

# Step 10 (PDF): Standardize data using StandardScaler
X_raw = df[subjects].values
scaler = StandardScaler()
X_std_sk = scaler.fit_transform(X_raw)

# Step 11 (PDF): Apply PCA with two components
pca = PCA(n_components=2)
coords_sklearn = pca.fit_transform(X_std_sk)

# Step 12 (PDF): Display transformed coordinates
print("Transformed Coordinates (Scikit-Learn):")
coords_sk_df = pd.DataFrame(coords_sklearn, index=df["Student"], columns=["PC1", "PC2"])
print(coords_sk_df.round(6))
print()

# Step 13 (PDF): Explained variance ratio and cumulative explained variance
cum_var_sk = np.cumsum(pca.explained_variance_ratio_)
print("Scikit-Learn Explained Variance:")
for i in range(2):
    print(f"  PC{i+1}: Variance = {pca.explained_variance_[i]:.6f}, "
          f"Ratio = {pca.explained_variance_ratio_[i]:.6f} ({pca.explained_variance_ratio_[i]*100:.2f}%), "
          f"Cumulative = {cum_var_sk[i]*100:.2f}%")
print()


# =============================================================================
# PART C: COMPARISON AND INTERPRETATION
# =============================================================================
print("--- Part C: Comparison and Interpretation ---")

# Step 14: Compare coordinates and handle sign changes
# Compute sign alignment using manual dot product of eigenvectors
sign_factors = []
for c in range(2):
    dot = sum(W[j][c] * pca.components_[c, j] for j in range(n_subjects))
    sign_factors.append(1.0 if dot >= 0 else -1.0)

print(f"Sign adjustments needed to align Scratch with Scikit-learn: PC1: {sign_factors[0]:.0f}, PC2: {sign_factors[1]:.0f}")

aligned_coords_scratch = []
for i in range(n_students):
    aligned_coords_scratch.append([
        coords_scratch[i][0] * sign_factors[0],
        coords_scratch[i][1] * sign_factors[1]
    ])

print("\nCoordinate Comparison Table:")
comp_df = pd.DataFrame({
    "Student": df["Student"],
    "Scratch PC1": [r[0] for r in coords_scratch],
    "Sklearn PC1": coords_sklearn[:, 0],
    "Aligned PC1": [r[0] for r in aligned_coords_scratch],
    "Scratch PC2": [r[1] for r in coords_scratch],
    "Sklearn PC2": coords_sklearn[:, 1],
    "Aligned PC2": [r[1] for r in aligned_coords_scratch]
})
print(comp_df.round(6).to_string(index=False))

# Check maximum differences manually
max_diff_pc1 = max(abs(aligned_coords_scratch[i][0] - coords_sklearn[i, 0]) for i in range(n_students))
max_diff_pc2 = max(abs(aligned_coords_scratch[i][1] - coords_sklearn[i, 1]) for i in range(n_students))
print(f"\nMax difference after sign alignment -> PC1: {max_diff_pc1:.2e}, PC2: {max_diff_pc2:.2e}")

# Step 15: Compare eigenvalues and explained variance ratios
print("\nEigenvalue and Variance Ratio Comparison:")
eig_comp = pd.DataFrame({
    "Component": ["PC1", "PC2"],
    "Scratch Eigenvalue": sorted_eigvals[:2],
    "Sklearn Eigenvalue": pca.explained_variance_,
    "Scratch Ratio": var_ratios[:2],
    "Sklearn Ratio": pca.explained_variance_ratio_
})
print(eig_comp.round(6).to_string(index=False))
print()

# Step 16: Plot the students in 2D using both implementations
students = df["Student"].tolist()

# Plot 1: Scratch implementation
plt.figure(figsize=(7, 5))
x_sc = [r[0] for r in coords_scratch]
y_sc = [r[1] for r in coords_scratch]
plt.scatter(x_sc, y_sc, color="#1f77b4", s=100)
for i, s in enumerate(students):
    plt.annotate(f"  {s}", (x_sc[i], y_sc[i]), fontsize=11, fontweight="bold")
plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
plt.axvline(0, color="gray", linestyle="--", alpha=0.6)
plt.title("2D Projection: Scratch PCA", fontsize=12, fontweight="bold")
plt.xlabel(f"PC1 ({var_ratios[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({var_ratios[1]*100:.2f}%)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("results/pca_scratch.png", dpi=300)
plt.close()

# Plot 2: Scikit-learn implementation
plt.figure(figsize=(7, 5))
x_sk = coords_sklearn[:, 0]
y_sk = coords_sklearn[:, 1]
plt.scatter(x_sk, y_sk, color="#2ca02c", s=100)
for i, s in enumerate(students):
    plt.annotate(f"  {s}", (x_sk[i], y_sk[i]), fontsize=11, fontweight="bold")
plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
plt.axvline(0, color="gray", linestyle="--", alpha=0.6)
plt.title("2D Projection: Scikit-Learn PCA", fontsize=12, fontweight="bold")
plt.xlabel(f"PC1 ({var_ratios[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({var_ratios[1]*100:.2f}%)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("results/pca_sklearn.png", dpi=300)
plt.close()

# Plot 3: Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(x_sc, y_sc, color="#1f77b4", s=100)
for i, s in enumerate(students):
    axes[0].annotate(f"  {s}", (x_sc[i], y_sc[i]), fontsize=11, fontweight="bold")
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.6)
axes[0].axvline(0, color="gray", linestyle="--", alpha=0.6)
axes[0].set_title("Scratch PCA (NumPy)", fontsize=12, fontweight="bold")
axes[0].set_xlabel(f"PC1 ({var_ratios[0]*100:.2f}%)")
axes[0].set_ylabel(f"PC2 ({var_ratios[1]*100:.2f}%)")
axes[0].grid(True, linestyle=":", alpha=0.6)

axes[1].scatter(x_sk, y_sk, color="#2ca02c", s=100)
for i, s in enumerate(students):
    axes[1].annotate(f"  {s}", (x_sk[i], y_sk[i]), fontsize=11, fontweight="bold")
axes[1].axhline(0, color="gray", linestyle="--", alpha=0.6)
axes[1].axvline(0, color="gray", linestyle="--", alpha=0.6)
axes[1].set_title("Scikit-Learn PCA", fontsize=12, fontweight="bold")
axes[1].set_xlabel(f"PC1 ({var_ratios[0]*100:.2f}%)")
axes[1].set_ylabel(f"PC2 ({var_ratios[1]*100:.2f}%)")
axes[1].grid(True, linestyle=":", alpha=0.6)

fig.suptitle("Comparison of Student Projections (Notice horizontal mirror flip along PC1)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/pca_comparison.png", dpi=300)
plt.close()

print("Plots saved in results/ directory:")
print("  - results/pca_scratch.png")
print("  - results/pca_sklearn.png")
print("  - results/pca_comparison.png\n")

# Step 17: Explain sign difference
print("Step 17 Explanation:")
print("Eigenvectors are defined up to an arbitrary sign factor because if C*v = lambda*v,")
print("then C*(-v) = lambda*(-v). NumPy's eigh chose the negative direction for PC1 while")
print("Scikit-Learn chose the positive direction. This creates a mirror reflection along PC1")
print("but does not change any distances between students or the amount of variance explained.\n")

# Step 18: Total variance retained by first two components
total_retained = (var_ratios[0] + var_ratios[1]) * 100.0
print(f"Step 18: Total variance retained by the first two components = {total_retained:.2f}%\n")

# Interpretation
print("Brief Interpretation of Findings:")
print("1. PC1 (93.49% of variance): All four subjects have positive weights of similar size")
print("   (~0.48 to ~0.52). PC1 represents overall academic performance. Students with high")
print("   grades in all subjects (S4, S2) have high positive scores on PC1, while students with")
print("   low grades in all subjects (S3, S5) have negative scores.")
print("2. PC2 (6.01% of variance): Contrasts Statistics (+0.78) against Programming (-0.59) and")
print("   Algorithms (-0.19). It reflects student profile: S6 scored highest in Statistics (15)")
print("   relative to Programming (13) and has the highest PC2 score (+0.79), whereas S1 had higher")
print("   grades in Programming (14) than Statistics (11) and has a negative PC2 score (-0.67).")
print("3. Retaining 2 components preserves 99.50% of the total variance, making this 2D summary")
print("   extremely accurate.")
