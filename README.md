# 🧬 Evolutionary Feature Selection: Metaheuristics Benchmark

![Status](https://img.shields.io/badge/status-research-purple)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Library](https://img.shields.io/badge/DEAP-Evolutionary-orange)
![Scikit-Learn](https://img.shields.io/badge/sklearn-wrapper-green)
![License](https://img.shields.io/badge/license-MIT-green)

This project implements a **Genetic Algorithm (GA)** based Wrapper approach for feature selection in high-dimensional classification problems.

The main objective is to solve the combinatorial problem of finding the optimal subset of features that maximizes model accuracy while simultaneously reducing dimensionality under **strict limits (Hard Constraints)**. To validate the proposal, a rigorous comparative study is conducted against other classical and modern metaheuristics. Additionally, the analysis is extended to a **multi-objective framework (NSGA-II)** to explore the trade-off between Precision and Recall.

---

## 📋 Table of Contents
- [Architecture and Tech Stack](#-arquitectura-y-tech-stack)
- [Project Structure](#-estructura-del-proyecto)
- [Implemented Algorithms](#-algoritmos-implementados)
- [Installation & Usage](#-instalación-y-uso)
- [Experimental Methodology](#-metodología-experimental)
- [Multi-objective Analysis](#-análisis-multiobjetivo)
- [Results and Statistics](#-resultados-y-estadística)
- [Authors](#-autor)

---

## 🛠 Architecture & Tech Stack

The code follows a decoupled modular architecture, separating the optimization logic (metaheuristics) from the model evaluation logic.

* **Lenguaje:** Python 3.12+
* **Evolutionary Computation:** [DEAP](https://deap.readthedocs.io/) (Distributed Evolutionary Algorithms in Python).
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Wrapper Model: Decision Tree).
* **Data Manipulation:** Pandas and NumPy.
* **Visualization:** Seaborn y Matplotlib.
* **Statistics:** SciPy (Friedman and Wilcoxon tests).
* **Dependency Management:** [uv](https://github.com/astral-sh/uv) (o standard pip).

---

## 📂 Project Structure

The repository is organized to ensure experiment reproducibility:

```text
.
├── csv/                       # 📂 Generated CSV results and tables
│   ├── resultados_comparativa_final.csv    # Raw results (30 runs per algorithm/dataset)
│   ├── tabla_resumen_paper_CI.csv          # Summary table with confidence intervals
│   ├── resultados_multiobjetivo.csv        # Pareto front solutions
│   └── tabla_resumen_paper.csv             # Summary statistics table
├── images/                    # 📊 Generated plots and visualizations
│   ├── boxplot_*.png              # Boxplots for each dataset
│   ├── pareto_*.png               # Pareto fronts (Precision vs Recall)
│   └── convergence_combined.png   # GA convergence curves
├── scripts/                   # 🧠 Source code
│   ├── algorithms.py              # Single-objective metaheuristics (GA, SA, Tabu, PSO, GWO)
│   ├── algorithms_mo.py           # Multi-objective logic (NSGA-II)
│   ├── analysis.py                # Statistical analysis, tests, and boxplots
│   ├── data_loader.py             # Dataset loading from OpenML
│   ├── evaluator.py               # Evaluator Class (Wrapper + Hard Constraints)
│   ├── generar_tabla_paper.py     # Generate tables with confidence intervals
│   ├── main_experiment.py         # Main script (Single-Objective Benchmark - 30 runs)
│   ├── main_multiobjective.py     # NSGA-II Pareto front analysis
│   ├── plot_convergence.py        # Generate GA convergence plots
│   └── tuning_optuna.py           # Hyperparameter tuning with Optuna
├── pyproject.toml             # ⚙️ Project dependencies
├── uv.lock                    # 🔒 Lock file for dependencies
└── README.md                  # 📄 Documentation

```

---

## 🧬 Implemented Algorithms

Five algorithms have been implemented and tuned for the comparison, all sharing the same evaluation function and constraints:

1. Genetic Algorithm (GA) - Proposed: Population-based evolution with cardinality constraints.

2. Simulated Annealing (SA): Trajectory-based metaheuristic inspired by physical annealing.

3. Tabu Search (TS): Local search with short-term memory to avoid cycles.

4. Particle Swarm Optimization (PSO): Binary swarm intelligence using a sigmoid transfer function.

5. Grey Wolf Optimizer (GWO): Modern bio-inspired algorithm based on the social hierarchy of wolves.

---

## 💻 Installation & Usage

### 1. Prerequisites

Ensure you have Python 3.12+ installed. Install dependencies:

```bash
# Option A: Using uv (recommended)
uv sync

# Option B: Using pip
pip install numpy pandas scikit-learn deap seaborn matplotlib scipy
```

### 2. Run Comparative Study (Single-Objective)

This script runs the 5 algorithms across the 5 datasets (**30 runs each**) and saves the results to a CSV file.

```bash
python scripts/main_experiment.py
```

**Output:** Generates `csv/resultados_comparativa_final.csv` (750 total runs: 5 datasets × 5 algorithms × 30 runs).

**Execution time:** ~8-12 hours (depending on hardware).

### 3. Generate Tables with Confidence Intervals

Once the experiment is finished, generate summary tables for the paper:

```bash
python scripts/generar_tabla_paper.py
```

**Output:** Generates `csv/tabla_resumen_paper_CI.csv` with means and 95% confidence intervals.

### 4. Generate Statistical Analysis

Run the analysis to generate boxplots and significance tests:

```bash
python scripts/analysis.py
```

**Output:** 
- Boxplots saved to `images/boxplot_*.png`
- Friedman and Wilcoxon test results printed to console

### 5. Run Multi-objective Analysis (NSGA-II)

To generate the Pareto Fronts (Precision vs Recall):

```bash
python scripts/main_multiobjective.py
```

**Output:** 
- Pareto front plots: `images/pareto_*.png`
- Solutions data: `csv/resultados_multiobjetivo.csv`

### 6. Generate Convergence Plots

To visualize GA convergence curves:

```bash
python scripts/plot_convergence.py
```

**Output:** `images/convergence_combined.png`

---

## ⚙️ Experimental Methodology

The study is conducted on **5 medical/biological datasets** from OpenML:
- **Breast Cancer** (Wisconsin Diagnostic)
- **Wine** (Chemical analysis)
- **Ionosphere** (Radar returns)
- **Lymphography** (Lymph node diagnosis)
- **Zoo** (Animal classification)

### Hard Constraints
Unlike standard approaches that use soft penalties, this project implements a **"death penalty"** mechanism. If an individual selects fewer than $k_{min}$ or more than $k_{max}$ features, its fitness is immediately reduced to 0.0, forcing the algorithm to search for compact solutions.

### Validation
The fitness of each solution is calculated using:
- **Model:** Decision Tree Classifier (scikit-learn)
- **Metric:** Precision (weighted for single-objective, macro for multi-objective)
- **Validation:** Stratified k-fold Cross-Validation (k=5 or dynamically adjusted)
- **Penalty coefficient (α):** 0.001 for single-objective (parsimony pressure)

### Statistical Robustness
- **30 independent runs** per algorithm per dataset
- **Different random seeds** for each run (42 + run_id)
- **95% Confidence Intervals** using t-distribution

---

## 🎯 Multi-objective Analysis

The **NSGA-II** algorithm is employed to simultaneously optimize two conflicting objectives:

1. **Maximize Precision (macro):** Minimize false positives across all classes equally
2. **Maximize Recall (macro):** Minimize false negatives across all classes equally

This results in a **Pareto Front** of non-dominated solutions, offering the human expert different trade-off options depending on whether they prefer to minimize false positives or false negatives.

### NSGA-II Configuration
- **Population size (μ):** 300
- **Generations:** 150
- **Crossover probability:** 0.6
- **Mutation probability:** 0.4
- **Selection:** NSGA-II (non-dominated sorting + crowding distance)
- **Penalty coefficient:** None (α = 0 for multi-objective)

---

## 📊 Results & Statistics

The project validates the superiority or equivalence of the algorithms using non-parametric tests:

### Statistical Tests
- **Friedman Test:** Detects global differences in algorithm rankings across all datasets
- **Wilcoxon Signed-Rank Test (Post-hoc):** Pairwise comparisons to determine significant differences between specific algorithms

### Outputs
- **Tables:** Summary statistics with means ± 95% confidence intervals (`csv/tabla_resumen_paper_CI.csv`)
- **Boxplots:** Distribution visualization for each dataset (`images/boxplot_*.png`)
- **Pareto Fronts:** Multi-objective trade-off curves (`images/pareto_*.png`)
- **Convergence Curves:** GA evolution over generations (`images/convergence_combined.png`)

### Key Metrics
- **Best Precision:** Main fitness metric (weighted average for single-objective)
- **Number of Features:** Parsimony measure (fewer features preferred)
- **Execution Time:** Computational efficiency tracking

---

## ✒️ Authors

**Juan Pedro García Sanz (jpgarciasanz@al.uloyola.es)**

**Adolfo Peña Marín (apenamarin@al.uloyola.es)**