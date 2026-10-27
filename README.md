# Machine Learning — Practical Session 2: Standardized PCA

**Institution:** National Higher School of Advanced Technologies  
**Academic Year:** 2026–2027 | First-Year Master's Degree  
**Subject:** Machine Learning — Python for Standardized PCA  

---

## 1. Project Overview & Implementation Constraints

This repository contains the complete implementation and comparison of **Standardized Principal Component Analysis (PCA)** under strict pedagogical rules:

### Permitted Predefined Mathematical Routines:
1. `np.linalg.eigh`: Used to calculate eigenvalues and eigenvectors of the real symmetric covariance matrix.
2. `np.corrcoef`: Used to calculate the Pearson correlation matrix.
3. `sklearn.preprocessing.StandardScaler` and `sklearn.decomposition.PCA(n_components=2)`: Exclusively used in **Part B** to provide the library comparison benchmark required by the assignment.

### All Other Calculations Implemented Manually (No Predefined Routines):
- **Descriptive Statistics:** Manual count, mean, sample std ($ddof=1$), population std ($ddof=0$), min, median, max using basic loops and arithmetic (no `DataFrame.describe()`).
- **Standardization:** $z = \frac{x - \mu}{\sigma}$ using population parameters computed manually.
- **Sample Covariance Matrix:** Computed manually with the sample convention ($N - 1 = 5$ divisor) using explicit nested loops.
- **Sorting:** Manual selection sort to order eigenvalues and reorder eigenvector columns in descending order (no `np.argsort`).
- **Selection of Principal Axes:** Manual 2-component column extraction.
- **Projection:** Manual nested loops ($Y[i][c] = \sum_j Z[i][j] \times W[j][c]$) without matrix multiplication libraries or `@` operator.
- **Variance Ratios:** Manual loop summation and ratio calculation.
- **Sign Alignment & Validation:** Manual dot product and coordinate difference check.

---

## 2. Directory Structure

```text
TP2/
│
├── Work session 2.pdf          # Practical session assignment specification
├── main.py                     # Complete Python implementation (Parts A, B, and C)
├── requirements.txt            # Minimal required Python dependencies
├── README.md                   # Setup guide, execution steps, and explanation
├── report.md                   # Full practical session report with numerical results & analysis
│
└── results/                    # Generated high-resolution plots (created on run)
    ├── pca_scratch.png         # 2D projection from manual scratch implementation
    ├── pca_sklearn.png         # 2D projection from Scikit-Learn
    └── pca_comparison.png      # Side-by-side comparison (Manual vs Scikit-Learn)
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.9+ (tested on Python 3.14.3)
- Pip package manager

### Step-by-Step Installation

1. Open PowerShell or terminal in the project directory:
   ```powershell
   cd "c:\Users\BAB AL SAFA\Desktop\TPs\ML\TP2"
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 4. Running the Program

Run `main.py`:

```powershell
python main.py
```

The script will:
- Display dimensions and manual descriptive statistics.
- Perform manual standardization and validate that standardized means $\approx 0$ and population stds $\approx 1$.
- Compute and display the correlation matrix (using `np.corrcoef`).
- Compute sample covariance manually ($ddof=1$) and calculate eigenvalues/eigenvectors using `np.linalg.eigh`.
- Sort components manually and project students manually onto PC1 and PC2.
- Execute Scikit-Learn PCA as the reference comparison.
- Manually check sign alignment, calculate coordinate differences (within $8.88 \times 10^{-16}$), and print variance retained ($99.50\%$).
- Save all plots to `results/`.
- Print an academic interpretation of student profiles.
