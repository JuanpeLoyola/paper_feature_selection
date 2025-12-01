import pandas as pd  
import numpy as np  
from scipy import stats  


def calcular_margen_ci(serie, confianza=0.95):
    """Calcular margen h para Media ± h (IC de confianza)."""
    a = 1.0 * np.array(serie)  # convertir a array float
    n = len(a)  # tamaño de la muestra
    if n < 2:
        return 0.0  # retorno seguro si no hay datos

    se = stats.sem(a)  # error estándar de la media
    h = se * stats.t.ppf((1 + confianza) / 2., n - 1)  # t * se
    return h  # margen


# cargar archivo de resultados
archivo = "resultados_comparativa_final.csv"
try:
    df = pd.read_csv(archivo)  # leer CSV
except FileNotFoundError:
    print(f"❌ No encuentro '{archivo}'. Asegúrate de haber corrido el experimento.")
    exit()

print("📊 Generando tabla con Intervalos de Confianza (95%)...")  # informar

# agrupar por Dataset y Algorithm y calcular mean y margen (CI)
resumen = df.groupby(['Dataset', 'Algorithm']).agg({
    'Best_Precision': ['mean', calcular_margen_ci],
    'N_Features': ['mean', calcular_margen_ci],
}).reset_index()

# formatear columnas con "Media ± CI"
resumen['Precision (Mean ± 95% CI)'] = resumen.apply(
    lambda x: f"{x[('Best_Precision', 'mean')]:.4f} ± {x[('Best_Precision', 'calcular_margen_ci')]:.4f}", axis=1
)

resumen['Features (Mean ± 95% CI)'] = resumen.apply(
    lambda x: f"{x[('N_Features', 'mean')]:.1f} ± {x[('N_Features', 'calcular_margen_ci')]:.1f}", axis=1
)

# columnas finales para exportar
tabla_final = resumen[['Dataset', 'Algorithm', 'Precision (Mean ± 95% CI)', 'Features (Mean ± 95% CI)']]

# exportar CSV
nombre_salida = "tabla_resumen_paper_CI.csv"
tabla_final.to_csv(nombre_salida, index=False)  # guardar

print("\n" + "=" * 60)
print(tabla_final.to_string())  # mostrar tabla
print("=" * 60)
print(f"✅ Tabla guardada en: {nombre_salida}")  # informar ruta