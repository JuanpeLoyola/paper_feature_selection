import pandas as pd  # manejo de datos tabulares
import seaborn as sns  # visualización estadística
import matplotlib.pyplot as plt  # backend de plots
from scipy.stats import friedmanchisquare, wilcoxon  # tests estadísticos
import itertools  # combinaciones por pares
import warnings  # control de warnings
import os  # operaciones de sistema de ficheros

warnings.filterwarnings("ignore")  # ignorar warnings


def analizar_resultados(archivo_csv):
    """Leer CSV de resultados y generar gráficos y tests."""
    try:
        df = pd.read_csv(archivo_csv)  # leer CSV
    except FileNotFoundError:
        print("❌ Archivo CSV no encontrado. Ejecuta main_experiment.py primero.")  # avisar si falta
        return  # salir si no existe el archivo

    datasets = df['Dataset'].unique()  # lista de datasets únicos
    algoritmos = df['Algorithm'].unique()  # lista de algoritmos únicos
    print(f"📊 Analizando: {len(datasets)} datasets, {len(algoritmos)} algoritmos\n")  # resumen

    CARPETA_IMG = "imagenes"  # carpeta para guardar imágenes
    os.makedirs(CARPETA_IMG, exist_ok=True)  # crear carpeta si hace falta

    for ds in datasets:
        plt.figure(figsize=(12, 6))  # nueva figura
        data_subset = df[df['Dataset'] == ds]  # filtrar por dataset

        sns.boxplot(x='Algorithm', y='Best_Precision', data=data_subset, palette="Set3")  # boxplot
        sns.swarmplot(x='Algorithm', y='Best_Precision', data=data_subset, color=".25", size=3)  # puntos

        plt.title(f'Precision Comparison - Dataset: {ds}')  # título
        plt.ylabel('Precision Weighted')  # etiqueta Y
        nombre_archivo = f"{CARPETA_IMG}/boxplot_{ds}.png"  # ruta salida
        plt.savefig(nombre_archivo, dpi=300)  # guardar figura
        print(f"   📊 Gráfico guardado: {nombre_archivo}")  # informar
        plt.close()  # cerrar figura

    print("\n" + "="*50)  # separador
    print("🧪 Tests estadísticos (Friedman + Wilcoxon)")  # encabezado tests
    print("="*50)  # separador

    for ds in datasets:
        print(f"\nDataset: {ds.upper()}")  # imprimir dataset
        df_ds = df[df['Dataset'] == ds]  # filtrar por dataset

        tabla = df_ds.pivot(index='Run_ID', columns='Algorithm', values='Best_Precision')  # pivot table

        vectores = [tabla[algo] for algo in algoritmos]  # vectores por algoritmo
        _, p_value = friedmanchisquare(*vectores)  # test de Friedman

        print(f"  Friedman p-value: {p_value:.2e} ", end="")  # mostrar p-value
        if p_value < 0.05:
            print("✅ (Diferencias significativas)")  # significativo
            print("    Comparaciones por pares (Wilcoxon):")  # post-hoc
            for a1, a2 in itertools.combinations(algoritmos, 2):
                _, w_p = wilcoxon(tabla[a1], tabla[a2])  # Wilcoxon par a par
                sig = "⭐" if w_p < 0.05 else "  "  # marcar significancia
                print(f"    {sig} {a1} vs {a2:<15} | p={w_p:.4f}")  # imprimir comparación
        else:
            print("❌ (Sin diferencias significativas)")  # no significativo


if __name__ == "__main__":
    analizar_resultados("/home/juanpe/master/practicas/paper_feature_selection/csv/resultados_comparativa_final.csv")  # ejecución por defecto