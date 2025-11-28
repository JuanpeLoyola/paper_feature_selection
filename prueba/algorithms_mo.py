import random
import numpy as np
from deap import base, creator, tools, algorithms

# Configuración DEAP para MULTIOBJETIVO
# Weights = (1.0, 1.0) -> Maximizar Precisión y Maximizar Recall
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
if not hasattr(creator, "IndividuoMO"):
    creator.create("IndividuoMO", list, fitness=creator.FitnessMulti)

def run_nsga2(evaluator, n_feats, params):
    # 1. Extraer parámetros del diccionario
    pop_size = params.get('pop_size', 100)
    n_gen = params.get('n_gen', 50)
    p_cx = params.get('p_cruce', 0.8)
    p_mut = params.get('p_mutacion', 0.2)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividuoMO, toolbox.attr_bool, n_feats)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Importante: Usamos el método específico multiobjetivo
    toolbox.register("evaluate", evaluator.evaluar_multiobjetivo)
    
    # Operadores NSGA-II
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)
    toolbox.register("select", tools.selNSGA2) # Selección clave para Pareto
    
    pop = toolbox.population(n=pop_size)
    pareto_front = tools.ParetoFront()
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Ejecución (Pasando ngen como entero explícito)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen,
                                             stats=stats, halloffame=pareto_front, verbose=False)
    
    # Retornamos TODO el frente de Pareto
    return pareto_front, logbook