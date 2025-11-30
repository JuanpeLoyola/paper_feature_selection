import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import warnings
import os

warnings.filterwarnings("ignore")  # Silencia warnings para salidas limpias


def analizar_resultados(archivo_csv):
    """Carga un CSV de resultados y produce gráficos + tests estadísticos.

    Entrada:
      - archivo_csv: ruta al CSV con columnas mínimas ['Dataset','Algorithm','Run_ID','Best_Precision']

    Salidas (por pantalla/figuras):
      - Boxplots por dataset comparando algoritmos
      - Resultado del test de Friedman por dataset
      - Comparaciones por pares (Wilcoxon) si Friedman es significativo
    """

    # 1) Cargar el CSV y validar existencia
    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print("❌ Archivo CSV no encontrado. Ejecuta main_experiment.py primero.")
        return

    # 2) Extraer listas únicas de datasets y algoritmos del CSV
    datasets = df['Dataset'].unique()
    algoritmos = df['Algorithm'].unique()
    print(f"📊 Analizando: {len(datasets)} datasets, {len(algoritmos)} algoritmos\n")

    # 3) Visualización: para cada dataset, mostrar boxplot + swarm de precisiones
    #    - Boxplot: resume la distribución por algoritmo
    #    - Swarmplot: muestra cada punto de ejecución encima del boxplot

    CARPETA_IMG = "imagenes"
    os.makedirs(CARPETA_IMG, exist_ok=True)

    for ds in datasets:
        plt.figure(figsize=(12, 6))
        data_subset = df[df['Dataset'] == ds]

        sns.boxplot(x='Algorithm', y='Best_Precision', data=data_subset, palette="Set3")
        sns.swarmplot(x='Algorithm', y='Best_Precision', data=data_subset, color=".25", size=3)

        plt.title(f'Precision Comparison - Dataset: {ds}')
        plt.ylabel('Precision Weighted')
        #plt.ylim(0, 1.05)
        # --- CAMBIO AQUÍ ---
        # En lugar de solo mostrar, guardamos la imagen
        nombre_archivo = f"{CARPETA_IMG}/boxplot_{ds}.png" # <--- CAMBIO AQUÍ
        plt.savefig(nombre_archivo, dpi=300)
        print(f"   📊 Gráfico guardado: {nombre_archivo}")
        plt.close()

    # 4) Tests estadísticos: primero Friedman (global), luego Wilcoxon por pares si procede
    print("\n" + "="*50)
    print("🧪 Tests estadísticos (Friedman + Wilcoxon)")
    print("="*50)

    for ds in datasets:
        print(f"\nDataset: {ds.upper()}")
        df_ds = df[df['Dataset'] == ds]

        # Construir tabla pivote: filas = ejecuciones (Run_ID), columnas = algoritmo
        # Esto permite pasar los vectores de precisión al test de Friedman
        tabla = df_ds.pivot(index='Run_ID', columns='Algorithm', values='Best_Precision')

        # Friedman: test no paramétrico para comparar más de 2 algoritmos en medidas repetidas
        vectores = [tabla[algo] for algo in algoritmos]
        _, p_value = friedmanchisquare(*vectores)

        print(f"  Friedman p-value: {p_value:.2e} ", end="")
        if p_value < 0.05:
            # Si Friedman detecta diferencias globales, hacemos post-hoc por pares
            print("✅ (Diferencias significativas)")
            print("    Comparaciones por pares (Wilcoxon):")
            for a1, a2 in itertools.combinations(algoritmos, 2):
                # Wilcoxon: test para pares emparejados (requiere mismas Run_IDs)
                _, w_p = wilcoxon(tabla[a1], tabla[a2])
                sig = "⭐" if w_p < 0.05 else "  "
                print(f"    {sig} {a1} vs {a2:<15} | p={w_p:.4f}")
        else:
            print("❌ (Sin diferencias significativas)")


if __name__ == "__main__":
    # Nombre del archivo resultante por defecto desde main_experiment.py
    analizar_resultados("/home/juanpe/master/practicas/paper_feature_selection/csv/resultados_comparativa_final.csv")