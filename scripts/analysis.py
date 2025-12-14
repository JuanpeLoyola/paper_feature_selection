import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import warnings
import os

warnings.filterwarnings("ignore")


def analizar_resultados(archivo_csv):
    """Read results CSV and generate plots and tests."""
    try:
        df = pd.read_csv(archivo_csv)  # read CSV
    except FileNotFoundError:
        print("❌ CSV file not found. Run main_experiment.py first.")  # warn if missing
        return  # exit if file doesn't exist

    datasets = df['Dataset'].unique()  # list of unique datasets
    algoritmos = df['Algorithm'].unique()  # list of unique algorithms
    print(f"📊 Analyzing: {len(datasets)} datasets, {len(algoritmos)} algorithms\n")  # summary

    CARPETA_IMG = "images"  # folder to save images
    os.makedirs(CARPETA_IMG, exist_ok=True)  # create folder if needed

    for ds in datasets:
        plt.figure(figsize=(12, 6))  # new figure
        data_subset = df[df['Dataset'] == ds]  # filter by dataset

        sns.boxplot(x='Algorithm', y='Best_Precision', data=data_subset, palette="Set3")  # boxplot
        sns.swarmplot(x='Algorithm', y='Best_Precision', data=data_subset, color=".25", size=3)  # points

        # plt.title(f'Precision Comparison - Dataset: {ds}')  # title
        plt.ylabel('Precision Weighted')  # Y label
        nombre_archivo = f"{CARPETA_IMG}/boxplot_{ds}.png"  # output path
        plt.savefig(nombre_archivo, dpi=300)  # save figure
        print(f"   📊 Plot saved: {nombre_archivo}")  # inform
        plt.close()  # close figure

    print("\n" + "="*50)  # separator
    print("🧪 Statistical tests (Friedman + Wilcoxon)")  # test header
    print("="*50)  # separator

    for ds in datasets:
        print(f"\nDataset: {ds.upper()}")  # print dataset
        df_ds = df[df['Dataset'] == ds]  # filter by dataset

        tabla = df_ds.pivot(index='Run_ID', columns='Algorithm', values='Best_Precision')  # pivot table

        vectores = [tabla[algo] for algo in algoritmos]  # vectors per algorithm
        _, p_value = friedmanchisquare(*vectores)  # Friedman test

        print(f"  Friedman p-value: {p_value:.2e} ", end="")  # show p-value
        if p_value < 0.05:
            print("✅ (Significant differences)")  # significant
            print("    Pairwise comparisons (Wilcoxon):")  # post-hoc
            for a1, a2 in itertools.combinations(algoritmos, 2):
                _, w_p = wilcoxon(tabla[a1], tabla[a2])  # Wilcoxon pairwise
                sig = "⭐" if w_p < 0.05 else "  "  # mark significance
                print(f"    {sig} {a1} vs {a2:<15} | p={w_p:.4f}")  # print comparison
        else:
            print("❌ (No significant differences)")  # not significant


if __name__ == "__main__":
    analizar_resultados("/home/juanpe/master/practicas/paper_feature_selection/csv/resultados_comparativa_final.csv")  # default execution