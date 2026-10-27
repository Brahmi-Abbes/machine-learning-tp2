# Practical Session 2 Report: Standardized PCA

**Course:** Machine Learning  
**Topic:** Standardized Principal Component Analysis (Scratch vs Scikit-Learn)  

---

## 1. Objective and Dataset

The goal of this practical session is to perform Standardized Principal Component Analysis (PCA) on a dataset of six students evaluated in four subjects, first from scratch using basic Python operations, and then using Scikit-Learn, comparing the numerical outcomes and interpreting the results.

### Dataset

| Student | Mathematics | Statistics | Programming | Algorithms |
| :---: | :---: | :---: | :---: | :---: |
| **S1** | 12 | 11 | 14 | 13 |
| **S2** | 15 | 14 | 16 | 15 |
| **S3** | 8 | 9 | 10 | 9 |
| **S4** | 17 | 16 | 18 | 17 |
| **S5** | 10 | 12 | 11 | 10 |
| **S6** | 14 | 15 | 13 | 14 |

- **Dimensions:** 6 rows (students) $\times$ 4 numerical columns (subjects).

---

## 2. Descriptive Statistics and Standardization

### Manual Descriptive Statistics

Calculated using loops and arithmetic (without `DataFrame.describe()`):

| Subject | Mean ($\mu$) | Sample Std ($s$, ddof=1) | Population Std ($\sigma$, ddof=0) | Min | Median | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mathematics** | 12.6667 | 3.3267 | 3.0368 | 8.0 | 13.0 | 17.0 |
| **Statistics** | 12.8333 | 2.6394 | 2.4095 | 9.0 | 13.0 | 16.0 |
| **Programming** | 13.6667 | 3.0111 | 2.7487 | 10.0 | 13.5 | 18.0 |
| **Algorithms** | 13.0000 | 3.0332 | 2.7689 | 9.0 | 13.5 | 17.0 |

### Standardization

Standardization uses the population parameters ($ddof=0$) to match Scikit-Learn's `StandardScaler`:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Standardized values $\mathbf{Z}$:

| Student | Mathematics | Statistics | Programming | Algorithms |
| :---: | :---: | :---: | :---: | :---: |
| **S1** | -0.2195 | -0.7609 | +0.1213 | 0.0000 |
| **S2** | +0.7683 | +0.4842 | +0.8489 | +0.7223 |
| **S3** | -1.5367 | -1.5909 | -1.3339 | -1.4446 |
| **S4** | +1.4269 | +1.3143 | +1.5765 | +1.4446 |
| **S5** | -0.8781 | -0.3459 | -0.9701 | -1.0835 |
| **S6** | +0.4391 | +0.8992 | -0.2425 | +0.3612 |

---

## 3. Correlation and Covariance

### Correlation Matrix

Computed with the permitted `np.corrcoef`:

$$\mathbf{R} = \begin{bmatrix}
1.0000 & 0.9263 & 0.9451 & 0.9911 \\
0.9263 & 1.0000 & 0.7717 & 0.8744 \\
0.9451 & 0.7717 & 1.0000 & 0.9635 \\
0.9911 & 0.8744 & 0.9635 & 1.0000
\end{bmatrix}$$

All correlations are strong and positive ($r \ge 0.77$).

### Sample Covariance Matrix

Calculated manually using the sample convention ($N - 1 = 5$ in denominator) to match Scikit-Learn's eigenvalue scale:

$$\mathbf{C}_{\text{sample}} = \frac{1}{N - 1} \mathbf{Z}^T \mathbf{Z} = \begin{bmatrix}
1.2000 & 1.1115 & 1.1341 & 1.1893 \\
1.1115 & 1.2000 & 0.9261 & 1.0492 \\
1.1341 & 0.9261 & 1.2000 & 1.1562 \\
1.1893 & 1.0492 & 1.1562 & 1.2000
\end{bmatrix}$$

Because each standardized column has a sample variance of $\frac{N}{N - 1} = \frac{6}{5} = 1.2$, the diagonal entries are $1.2$.

---

## 4. PCA from Scratch (NumPy)

1. **Eigendecomposition:** Computed using permitted `np.linalg.eigh(cov)`.
2. **Manual Sorting:** Eigenvalues and their corresponding eigenvector columns were sorted in descending order:
   - $\lambda_1 = 4.487673$
   - $\lambda_2 = 0.288274$
   - $\lambda_3 = 0.022941$
   - $\lambda_4 = 0.001113$
   - Total sample variance = $4.800000$.
3. **Selected Eigenvectors ($\mathbf{W}$, $4 \times 2$):**

| Subject | PC1 Eigenvector | PC2 Eigenvector |
| :--- | :---: | :---: |
| **Mathematics** | -0.516661 | +0.031968 |
| **Statistics** | -0.477046 | +0.782218 |
| **Programming** | -0.492822 | -0.592169 |
| **Algorithms** | -0.512460 | -0.190916 |

4. **Projection:** Coordinates are calculated manually using loops:
   $$\text{coord}_{ic} = \sum_{j=1}^4 z_{ij} w_{jc}$$
5. **Variance Explained:**
   - PC1: $\frac{4.487673}{4.8} = 93.4932\%$
   - PC2: $\frac{0.288274}{4.8} = 6.0057\%$
   - Cumulative for PC1 + PC2 = **$99.4989\%$ ($\approx 99.50\%$)**

---

## 5. PCA Using Scikit-Learn

Using `StandardScaler` and `PCA(n_components=2)`:
- Eigenvalues (`pca.explained_variance_`): $\lambda_1 = 4.487673$, $\lambda_2 = 0.288274$.
- Explained variance ratios: $\text{PC1} = 93.4932\%$, $\text{PC2} = 6.0057\%$.
- Components matrix:
  - $\text{PC1} = [+0.516661, +0.477046, +0.492822, +0.512460]$
  - $\text{PC2} = [+0.031968, +0.782218, -0.592169, -0.190916]$

---

## 6. Numerical Comparison

### Student Coordinates

| Student | Scratch PC1 | Sklearn PC1 | Aligned Scratch PC1 | Scratch PC2 | Sklearn PC2 | Aligned Scratch PC2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | +0.416636 | -0.416636 | -0.416636 | -0.674008 | -0.674008 | -0.674008 |
| **S2** | -1.416464 | +1.416464 | +1.416464 | -0.237266 | -0.237266 | -0.237266 |
| **S3** | +2.950619 | -2.950619 | -2.950619 | -0.227866 | -0.227866 | -0.227866 |
| **S4** | -2.881443 | +2.881443 | +2.881443 | -0.135694 | -0.135694 | -0.135694 |
| **S5** | +1.652021 | -1.652021 | -1.652021 | +0.482733 | +0.482733 | +0.482733 |
| **S6** | -0.721369 | +0.721369 | +0.721369 | +0.792100 | +0.792100 | +0.792100 |

### Explanation of Sign Difference (Step 17)

An eigenvector $\mathbf{v}$ satisfies $\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$. Multiplying both sides by $-1$ gives $\mathbf{C}(-\mathbf{v}) = \lambda(-\mathbf{v})$. Therefore, both $\mathbf{v}$ and $-\mathbf{v}$ are mathematically valid unit eigenvectors for the same axis.
- For PC1, NumPy's solver picked $-\mathbf{v}_1$, while Scikit-Learn picked $+\mathbf{v}_1$.
- For PC2, both solvers picked the same direction.
- After multiplying Scratch PC1 by $-1$, the maximum difference across all coordinates is $8.88 \times 10^{-16}$, confirming that both implementations are identical to machine precision.

---

## 7. Visualizations

The generated 2D student projection plots are saved in the `results/` folder:
- **Scratch 2D Plot:** `results/pca_scratch.png`
- **Scikit-Learn 2D Plot:** `results/pca_sklearn.png`
- **Side-by-Side Comparison:** `results/pca_comparison.png`

The comparison plot illustrates that the two coordinate systems are exact mirror images across the vertical axis, preserving all relative distances between students.

---

## 8. Interpretation of Results

1. **Total Variance Retained (Step 18):**
   The first two principal components retain **99.50%** of the total variance ($93.49\% + 6.01\%$). Dimensionality reduction from 4 variables to 2 variables loses only $0.50\%$ of information.
2. **Principal Component 1 (General Performance):**
   All four subject weights in PC1 have positive signs and similar values ($\approx 0.48$ to $0.52$). PC1 represents general academic performance across all subjects.
   - **S4 (+2.88)** and **S2 (+1.42)** have the highest marks across all four subjects and score highest on PC1.
   - **S3 (-2.95)** and **S5 (-1.65)** have the lowest marks across all four subjects and score lowest on PC1.
3. **Principal Component 2 (Specialization Contrast):**
   PC2 contrasts Statistics ($+0.782$) against Programming ($-0.592$) and Algorithms ($-0.191$).
   - **S6 (+0.79):** Scored higher in Statistics (15) than in Programming (13).
   - **S1 (-0.67):** Scored higher in Programming (14) than in Statistics (11).
