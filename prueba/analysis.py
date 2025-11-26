import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import warnings

warnings.filterwarnings("ignore")


def analizar_resultados(archivo_csv):
    """Genera visualizaciones y tests estadísticos de los resultados."""
    # Cargar datos
    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print("❌ Archivo CSV no encontrado. Ejecuta main_experiment.py primero.")
        return

    datasets = df['Dataset'].unique()
    algoritmos = df['Algorithm'].unique()
    print(f"📊 Analizando: {len(datasets)} datasets, {len(algoritmos)} algoritmos\n")
    
    # Visualización: boxplots por dataset
    for ds in datasets:
        plt.figure(figsize=(12, 6))
        data_subset = df[df['Dataset'] == ds]
        
        sns.boxplot(x='Algorithm', y='Best_Precision', data=data_subset, palette="Set3")
        sns.swarmplot(x='Algorithm', y='Best_Precision', data=data_subset, color=".25", size=3)
        
        plt.title(f'Comparativa de Precisión - Dataset: {ds}')
        plt.ylabel('Precision Weighted')
        plt.ylim(0, 1.05)
        plt.show()

    # Tests estadísticos: Friedman + Wilcoxon
    print("\n" + "="*50)
    print("🧪 Tests estadísticos (Friedman + Wilcoxon)")
    print("="*50)
    
    for ds in datasets:
        print(f"\nDataset: {ds.upper()}")
        df_ds = df[df['Dataset'] == ds]
        
        # Tabla pivote: filas=ejecuciones, columnas=algoritmos
        tabla = df_ds.pivot(index='Run_ID', columns='Algorithm', values='Best_Precision')
        
        # Test de Friedman (diferencias globales)
        vectores = [tabla[algo] for algo in algoritmos]
        stat, p_value = friedmanchisquare(*vectores)
        
        print(f"  Friedman p-value: {p_value:.2e} ", end="")
        if p_value < 0.05:
            print("✅ (Diferencias significativas)")
            
            # Post-hoc: Wilcoxon por pares
            print("    Comparaciones por pares (Wilcoxon):")
            for a1, a2 in itertools.combinations(algoritmos, 2):
                w_stat, w_p = wilcoxon(tabla[a1], tabla[a2])
                sig = "⭐" if w_p < 0.05 else "  "
                print(f"    {sig} {a1} vs {a2:<15} | p={w_p:.4f}")
        else:
            print("❌ (Sin diferencias significativas)")


if __name__ == "__main__":
    analizar_resultados("resultados_comparativa_final.csv")