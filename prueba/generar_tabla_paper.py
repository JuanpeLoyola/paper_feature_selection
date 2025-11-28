import pandas as pd

# Cargar resultados
df = pd.read_csv("resultados_comparativa_final.csv")

# Agrupar por Dataset y Algoritmo
resumen = df.groupby(['Dataset', 'Algorithm']).agg({
    'Best_Precision': ['mean', 'std'],
    'N_Features': ['mean', 'std']
}).reset_index()

# Formatear para que quede bonito (Media ± Desviación)
resumen['Precision (Mean ± Std)'] = resumen.apply(
    lambda x: f"{x[('Best_Precision', 'mean')]:.4f} ± {x[('Best_Precision', 'std')]:.3f}", axis=1
)
resumen['Features (Mean ± Std)'] = resumen.apply(
    lambda x: f"{x[('N_Features', 'mean')]:.1f} ± {x[('N_Features', 'std')]:.1f}", axis=1
)

# Seleccionar columnas finales
tabla_final = resumen[['Dataset', 'Algorithm', 'Precision (Mean ± Std)', 'Features (Mean ± Std)']]

print(tabla_final.to_string())
# Opcional: Guardar en CSV para abrir en Excel y copiar
tabla_final.to_csv("tabla_resumen_paper.csv", index=False)