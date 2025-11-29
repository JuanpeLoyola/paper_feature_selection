import pandas as pd
import numpy as np
from scipy import stats

def calcular_margen_ci(serie, confianza=0.95):
    """
    Calcula el margen de error (h) para un intervalo de confianza.
    El intervalo es: Media ± h
    """
    a = 1.0 * np.array(serie)
    n = len(a)
    if n < 2: return 0.0 # Seguridad por si falla algo
    
    se = stats.sem(a) # Error estándar
    # Usamos t-student porque n < 30
    h = se * stats.t.ppf((1 + confianza) / 2., n-1)
    return h

# 1. Cargar resultados
archivo = "resultados_comparativa_final.csv"
try:
    df = pd.read_csv(archivo)
except FileNotFoundError:
    print(f"❌ No encuentro '{archivo}'. Asegúrate de haber corrido el experimento.")
    exit()

print("📊 Generando tabla con Intervalos de Confianza (95%)...")

# 2. Agrupar y calcular Media y Margen de Error (CI)
resumen = df.groupby(['Dataset', 'Algorithm']).agg({
    'Best_Precision': ['mean', calcular_margen_ci],
    'N_Features': ['mean', calcular_margen_ci]
}).reset_index()

# 3. Formatear estilo Paper: "Media ± CI"
# Ejemplo: "0.950 ± 0.004"
resumen['Precision (Mean ± 95% CI)'] = resumen.apply(
    lambda x: f"{x[('Best_Precision', 'mean')]:.4f} ± {x[('Best_Precision', 'calcular_margen_ci')]:.4f}", axis=1
)

# Ejemplo: "5.2 ± 1.1"
resumen['Features (Mean ± 95% CI)'] = resumen.apply(
    lambda x: f"{x[('N_Features', 'mean')]:.1f} ± {x[('N_Features', 'calcular_margen_ci')]:.1f}", axis=1
)

# 4. Seleccionar columnas finales para el paper
tabla_final = resumen[['Dataset', 'Algorithm', 'Precision (Mean ± 95% CI)', 'Features (Mean ± 95% CI)']]

# 5. Exportar
nombre_salida = "tabla_resumen_paper_CI.csv"
tabla_final.to_csv(nombre_salida, index=False)

print("\n" + "="*60)
print(tabla_final.to_string())
print("="*60)
print(f"✅ Tabla guardada en: {nombre_salida}")
print("   -> Copia estos valores a tu Tabla 3 en el LaTeX.")