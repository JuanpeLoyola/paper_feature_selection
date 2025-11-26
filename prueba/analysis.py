import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import warnings

# Este archivo se ejecuta después de tener el CSV de resultados. Genera gráficos y tests estadísticos.

warnings.filterwarnings("ignore")

def analizar_resultados(archivo_csv):
    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print("❌ No se encuentra el archivo CSV. Ejecuta main_experiment.py primero.")
        return

    datasets = df['Dataset'].unique()
    algoritmos = df['Algorithm'].unique()
    
    print(f"📊 ANALIZANDO: {len(datasets)} Datasets x {len(algoritmos)} Algoritmos")
    
    # 1. GRAFICAR BOXPLOTS
    for ds in datasets:
        plt.figure(figsize=(12, 6))
        data_subset = df[df['Dataset'] == ds]
        
        sns.boxplot(x='Algorithm', y='Best_Precision', data=data_subset, palette="Set3")
        sns.swarmplot(x='Algorithm', y='Best_Precision', data=data_subset, color=".25", size=3)
        
        plt.title(f'Comparativa de Precisión - Dataset: {ds}')
        plt.ylabel('Precision Weighted')
        plt.ylim(0, 1.05)
        plt.show()

    # 2. TESTS ESTADÍSTICOS
    print("\n" + "="*50)
    print("🧪 TESTS ESTADÍSTICOS (Friedman + Wilcoxon)")
    print("="*50)
    
    for ds in datasets:
        print(f"\ndataset: {ds.upper()}")
        df_ds = df[df['Dataset'] == ds]
        
        # Crear tabla pivote: Filas=Run_ID, Cols=Algoritmo
        tabla = df_ds.pivot(index='Run_ID', columns='Algorithm', values='Best_Precision')
        
        # Test de Friedman
        vectores = [tabla[algo] for algo in algoritmos]
        stat, p_value = friedmanchisquare(*vectores)
        
        print(f"  > Friedman p-value: {p_value:.2e} ", end="")
        if p_value < 0.05:
            print("✅ (Diferencias significativas)")
            
            # Post-hoc: Wilcoxon
            pairs = list(itertools.combinations(algoritmos, 2))
            print("    --- Comparaciones por pares (Wilcoxon) ---")
            for a1, a2 in pairs:
                w_stat, w_p = wilcoxon(tabla[a1], tabla[a2])
                sig = "⭐ SI" if w_p < 0.05 else "   NO"
                print(f"    {a1} vs {a2:<15} | p={w_p:.4f} {sig}")
        else:
            print("❌ (Sin diferencias significativas)")

if __name__ == "__main__":
    # Asegúrate de que este nombre coincida con el que genera main_experiment.py
    analizar_resultados("resultados_comparativa.csv")