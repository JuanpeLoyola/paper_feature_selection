import random  
import numpy as np  
from deap import base, creator, tools, algorithms  

# create multi-objective fitness class if not exists
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # optimize two objectives (max, max)
# create multi-objective individual class if not exists
if not hasattr(creator, "IndividuoMO"):
    creator.create("IndividuoMO", list, fitness=creator.FitnessMulti)  # list-based individual

def run_nsga2(evaluator, n_feats, params):
    pop_size = params.get('pop_size', 100)  # population size
    n_gen = params.get('n_gen', 50)  # number of generations
    p_cx = params.get('p_cruce', 0.8)  # crossover probability
    p_mut = params.get('p_mutacion', 0.2)  # mutation probability

    toolbox = base.Toolbox()  # operator container
    toolbox.register("attr_bool", random.randint, 0, 1)  # binary attribute gene
    toolbox.register("individual", tools.initRepeat, creator.IndividuoMO, toolbox.attr_bool, n_feats)  # individual
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # population

    toolbox.register("evaluate", evaluator.evaluar_multiobjetivo)  # multi-objective evaluation function

    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)  # flip bit mutation
    toolbox.register("select", tools.selNSGA2)  # NSGA-II selection

    pop = toolbox.population(n=pop_size)  # create initial population
    pareto_front = tools.ParetoFront()  # Pareto hall of fame

    stats = tools.Statistics(lambda ind: ind.fitness.values)  # collect fitness
    stats.register("avg", np.mean, axis=0)  # average per objective
    stats.register("max", np.max, axis=0)  # maximum per objective

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen,
                                             stats=stats, halloffame=pareto_front, verbose=False)  # run NSGA-II

    return pareto_front, logbook  # return Pareto front and log