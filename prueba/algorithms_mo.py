import random
import numpy as np
from deap import base, creator, tools, algorithms

# Configuración DEAP para MULTIOBJETIVO
# Weights = (1.0, 1.0) -> Maximizar Precisión y Maximizar Recall
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("IndividuoMO", list, fitness=creator.FitnessMulti)

def run_nsga2(evaluator, n_feats, params):
    # Extraer parámetros
    pop_size = params['pop_size']
    n_gen = params['n_gen']
    p_cx = params['p_cruce']
    p_mut = params['p_mutacion']
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividuoMO, toolbox.attr_bool, n_feats)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Importante: Usamos el nuevo método del evaluador
    toolbox.register("evaluate", evaluator.evaluar_multiobjetivo)
    
    # Operadores NSGA-II
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)
    toolbox.register("select", tools.selNSGA2) # Selección clave
    
    pop = toolbox.population(n=pop_size)
    pareto_front = tools.ParetoFront()
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Ejecución
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen,
                                             stats=stats, halloffame=pareto_front, verbose=False)
    
    # Retornamos TODO el frente de Pareto (no solo un individuo)
    return pareto_front, logbook