import random  
import numpy as np  
from deap import base, creator, tools, algorithms  

# crear clase de fitness multiobjetivo si no existe
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # optimizar dos objetivos (max, max)
# crear clase de individuo para multiobjetivo si no existe
if not hasattr(creator, "IndividuoMO"):
    creator.create("IndividuoMO", list, fitness=creator.FitnessMulti)  # individuo basado en lista

def run_nsga2(evaluator, n_feats, params):
    pop_size = params.get('pop_size', 100)  # tamaño de la población
    n_gen = params.get('n_gen', 50)  # número de generaciones
    p_cx = params.get('p_cruce', 0.8)  # probabilidad de cruce
    p_mut = params.get('p_mutacion', 0.2)  # probabilidad de mutación

    toolbox = base.Toolbox()  # contenedor de operadores
    toolbox.register("attr_bool", random.randint, 0, 1)  # gen de atributo binario
    toolbox.register("individual", tools.initRepeat, creator.IndividuoMO, toolbox.attr_bool, n_feats)  # individuo
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # población

    toolbox.register("evaluate", evaluator.evaluar_multiobjetivo)  # función de evaluación multiobjetivo

    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # cruce uniforme
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)  # mutación por flip bit
    toolbox.register("select", tools.selNSGA2)  # selección NSGA-II

    pop = toolbox.population(n=pop_size)  # crear población inicial
    pareto_front = tools.ParetoFront()  # hall of fame para Pareto

    stats = tools.Statistics(lambda ind: ind.fitness.values)  # recopilar fitness
    stats.register("avg", np.mean, axis=0)  # media por objetivo
    stats.register("max", np.max, axis=0)  # máximo por objetivo

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen,
                                             stats=stats, halloffame=pareto_front, verbose=False)  # ejecutar NSGA-II

    return pareto_front, logbook  # devolver frente de Pareto y log