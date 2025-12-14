import pandas as pd  
import numpy as np  
from scipy import stats  


def calcular_margen_ci(serie, confianza=0.95):
    """Calculate margin h for Mean ± h (confidence interval)."""
    a = 1.0 * np.array(serie)  # convert to float array
    n = len(a)  # sample size
    if n < 2:
        return 0.0  # safe return if no data

    se = stats.sem(a)  # standard error of mean
    h = se * stats.t.ppf((1 + confianza) / 2., n - 1)  # t * se
    return h  # margin


# load results file
archivo = "csv/resultados_comparativa_final.csv"
try:
    df = pd.read_csv(archivo)  # read CSV
except FileNotFoundError:
    print(f"❌ Cannot find '{archivo}'. Make sure you've run the experiment.")
    exit()

print("📊 Generating table with 95% Confidence Intervals...")  # inform

# group by Dataset and Algorithm and calculate mean and margin (CI)
resumen = df.groupby(['Dataset', 'Algorithm']).agg({
    'Best_Precision': ['mean', calcular_margen_ci],
    'N_Features': ['mean', calcular_margen_ci],
}).reset_index()

# format columns with "Mean ± CI"
resumen['Precision (Mean ± 95% CI)'] = resumen.apply(
    lambda x: f"{x[('Best_Precision', 'mean')]:.4f} ± {x[('Best_Precision', 'calcular_margen_ci')]:.4f}", axis=1
)

resumen['Features (Mean ± 95% CI)'] = resumen.apply(
    lambda x: f"{x[('N_Features', 'mean')]:.1f} ± {x[('N_Features', 'calcular_margen_ci')]:.1f}", axis=1
)

# final columns to export
tabla_final = resumen[['Dataset', 'Algorithm', 'Precision (Mean ± 95% CI)', 'Features (Mean ± 95% CI)']]

# export CSV
nombre_salida = "csv/tabla_resumen_paper_CI.csv"
tabla_final.to_csv(nombre_salida, index=False)  # save

print("\n" + "=" * 60)
print(tabla_final.to_string())  # show table
print("=" * 60)
print(f"✅ Table saved at: {nombre_salida}")  # inform path