# 🧬 Evolutionary Feature Selection: Metaheuristics Benchmark

![Status](https://img.shields.io/badge/status-research-purple)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Library](https://img.shields.io/badge/DEAP-Evolutionary-orange)
![Scikit-Learn](https://img.shields.io/badge/sklearn-wrapper-green)
![License](https://img.shields.io/badge/license-MIT-green)

This project implements a **Genetic Algorithm (GA)** based Wrapper approach for feature selection in high-dimensional classification problems.

The main objective is to solve the combinatorial problem of finding the optimal subset of features that maximizes model accuracy while simultaneously reducing dimensionality under **strict limits (Hard Constraints)**. To validate the proposal, a rigorous comparative study is conducted against other classical and modern metaheuristics. Additionally, the analysis is extended to a **multi-objective framework (NSGA-II)** to explore the trade-off between Precision and Recall.

---

## 📋 Tabla de Contenidos
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
├── algorithms.py          # 🧠 Metaheuristics logic (GA, SA, TS, PSO, GWO)
├── algorithms_mo.py       # 🧬 Multi-objective logic (NSGA-II)
├── analysis.py            # 📊 Script for statistical analysis and visualization
├── data_loader.py         # 📥 Dataset ingestion and cleaning (UCI)
├── evaluator.py           # ⚖️ Evaluator Class (Wrapper + Constraints)
├── main_experiment.py     # 🚀 Main script (Single-Objective Benchmark)
├── main_multiobjective.py # 🎯 Secondary script (Pareto Analysis)
├── pyproject.toml         # ⚙️ Project dependencies
└── README.md              # 📄 Documentation

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

1. Prerequisites

Ensure you have Python 3.12 installed. Install dependencies:

```bash

# Option A: pip
pip install numpy pandas scikit-learn deap seaborn matplotlib scipy

# Option B: uv (recommended)
uv sync
```

2. Run Comparative Study (Single-Objective)

This script runs the 5 algorithms across the 5 datasets (10 runs each) and saves the results to a CSV file.

```bash

python main_experiment.py

```
Output: Generates resultados_comparativa_final.csv.

3. Generate Statistical Analysis

Once the experiment is finished, run the analysis to generate Boxplots and calculate p-values.

```bash

python analysis.py

```

Output: Displays distribution plots and prints significance tables (Friedman/Wilcoxon) to the console.

4. Run Multi-objective Analysis

To generate the Pareto Fronts (Precision vs Recall):

```bash

python main_multiobjective.py

```

Output: Generates pareto_{dataset}.png images in the root folder.

---

## ⚙️ Experimental Methodology

The study is conducted on 5 medical/biological datasets from the UCI repository (Breast Cancer, Wine, Ionosphere, Lymphography, Zoo).

**Hard Constraints:** Unlike standard approaches that use soft penalties, this project implements a **"death penalty"** mechanism. If an individual selects fewer than $K_{min}$ or more than $K_{max}$ features, its fitness is immediately reduced to 0.0, forcing the algorithm to search for compact solutions.

**Validation:** The fitness of each solution is calculated using the mean precision of a Decision Tree with Cross-Validation ($k=5$ folds).

---

## 🎯 Multi-objective Analysis

The NSGA-II algorithm is employed to simultaneously optimize two conflicting objectives:

1. Maximize Precision.

2. Maximize Sensitivity (Recall).

This results in a set of non-dominated solutions (Pareto Front), offering the human expert different options depending on whether they prefer to minimize false positives or false negatives.

---

## 📊 Results & Statistics

The project validates the superiority or equivalence of the algorithms using non-parametric tests:

* Friedman Test: To detect global differences in the algorithms' rankings.

* Wilcoxon Test (Post-hoc): To compare algorithm pairs and determine if the proposed method significantly outperforms classical methods.

(Insert an image of the boxplots generated by analysis.py here)

---

## ✒️ Authors

**Juan Pedro García Sanz**

**Adolfo Peña Marín**