import random  
import numpy as np  
from deap import base, creator, tools, algorithms  

# create fitness class (maximization) if not exists
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # fitness to maximize
# create individual class if not exists
if not hasattr(creator, "Individuo"):
    creator.create("Individuo", list, fitness=creator.FitnessMax)  # list-based individual


def run_ga(evaluator, n_feats, params):
    """Genetic Algorithm with DEAP."""
    pop_size = params.get('pop_size', 50)  # population size
    n_gen = params.get('n_gen', 50)  # generations
    p_cx = params.get('p_cruce', 0.5)  # crossover probability
    p_mut = params.get('p_mutacion', 0.2)  # mutation probability
    tam_torneo = params.get('tam_torneo', 3)  # tournament size

    toolbox = base.Toolbox()  # create toolbox
    toolbox.register("attr_bool", random.randint, 0, 1)  # random bit 0/1
    toolbox.register("individual", tools.initRepeat, creator.Individuo, toolbox.attr_bool, n_feats)  # individual
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # population

    toolbox.register("evaluate", evaluator.evaluar)  # evaluation function

    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)  # flip bit mutation
    toolbox.register("select", tools.selTournament, tournsize=tam_torneo)  # tournament selection

    pop = toolbox.population(n=pop_size)  # initialize population
    hof = tools.HallOfFame(1)  # save best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # collect fitness
    stats.register("avg", np.mean)  # register average
    stats.register("max", np.max)  # register maximum

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen,
                             stats=stats, halloffame=hof, verbose=False)  # run algorithm

    return hof[0], hof[0].fitness.values[0], log  # return best individual, its fitness and log


def run_sa(evaluator, n_feats, params):
    """Simulated Annealing (geometric cooling)."""
    max_iter = params.get('max_iter', 1000)  # iterations
    T = params.get('temp_init', 1.0)  # initial temperature
    alpha = params.get('alpha', 0.99)  # cooling factor

    current_sol = [random.randint(0, 1) for _ in range(n_feats)]  # random initial solution
    current_fit = evaluator.evaluar(current_sol)[0]  # current fitness
    best_sol = list(current_sol)  # best solution found
    best_fit = current_fit  # best fitness

    for _ in range(max_iter):
        neighbor = list(current_sol)  # copy solution
        idx = random.randint(0, n_feats - 1)  # index to mutate
        neighbor[idx] = 1 - neighbor[idx]  # flip bit in neighbor

        neighbor_fit = evaluator.evaluar(neighbor)[0]  # evaluate neighbor

        delta = neighbor_fit - current_fit  # fitness difference
        if delta > 0 or random.random() < np.exp(delta / T):  # Metropolis criterion
            current_sol = neighbor  # accept neighbor
            current_fit = neighbor_fit  # update current fitness

            if current_fit > best_fit:  # update best if improved
                best_fit = current_fit
                best_sol = list(current_sol)

        T *= alpha  # cool temperature

    return best_sol, best_fit  # return best solution and its fitness


def run_tabu(evaluator, n_feats, params):
    """Tabu Search with aspiration criterion."""
    max_iter = params.get('max_iter', 200)  # iterations
    tabu_size = params.get('tabu_size', 10)  # tabu list size
    n_neighbors = params.get('n_neighbors', 10)  # neighbors per iteration

    current_sol = [random.randint(0, 1) for _ in range(n_feats)]  # initial solution
    best_sol = list(current_sol)  # best solution
    best_fit = evaluator.evaluar(current_sol)[0]  # best fitness
    tabu_list = []  # tabu list (move indices)

    for _ in range(max_iter):
        candidates = []  # candidate list (neighbor, fit, move)
        for _ in range(n_neighbors):
            neighbor = list(current_sol)  # copy solution
            move_idx = random.randint(0, n_feats - 1)  # index to change
            neighbor[move_idx] = 1 - neighbor[move_idx]  # apply move

            fit = evaluator.evaluar(neighbor)[0]  # evaluate neighbor

            if move_idx not in tabu_list or fit > best_fit:  # aspiration criterion
                candidates.append((neighbor, fit, move_idx))  # add candidate

        if not candidates:
            continue  # no valid candidates

        best_candidate = max(candidates, key=lambda x: x[1])  # choose best by fitness
        current_sol, fit_val, move_idx = best_candidate  # update current solution

        if fit_val > best_fit:  # update global if improved
            best_fit = fit_val
            best_sol = list(current_sol)

        tabu_list.append(move_idx)  # add move to tabu
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)  # maintain size

    return best_sol, best_fit  # return best solution and its fitness


def run_pso(evaluator, n_feats, params):
    """Binary PSO with sigmoid binarization."""
    n_particles = params.get('n_particles', 30)  # number of particles
    max_iter = params.get('max_iter', 50)  # iterations
    w = params.get('w', 0.7)  # inertia
    c1 = params.get('c1', 1.5)  # social coefficient
    c2 = params.get('c2', 1.5)  # cognitive coefficient

    X = np.random.randint(2, size=(n_particles, n_feats))  # binary positions
    V = np.random.uniform(-1, 1, size=(n_particles, n_feats))  # continuous velocities

    P_best = X.copy()  # personal bests
    P_best_fit = np.array([evaluator.evaluar(ind)[0] for ind in X])  # personal fitness

    g_best_idx = np.argmax(P_best_fit)  # global best index
    G_best = P_best[g_best_idx].copy()  # global best position
    G_best_fit = P_best_fit[g_best_idx]  # global best fitness

    for _ in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()  # random values
            V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])  # update velocity

            V[i] = np.clip(V[i], -10, 10)  # avoid overflow
            sigmoid = 1 / (1 + np.exp(-V[i]))  # sigmoid per component

            X[i] = (np.random.rand(n_feats) < sigmoid).astype(int)  # update binary position

            fit = evaluator.evaluar(X[i])[0]  # evaluate particle

            if fit > P_best_fit[i]:  # update personal best
                P_best_fit[i] = fit
                P_best[i] = X[i].copy()

                if fit > G_best_fit:  # update global if improved
                    G_best_fit = fit
                    G_best = X[i].copy()

    return G_best.tolist(), G_best_fit  # return global best


def run_gwo(evaluator, n_feats, params):
    """Binary Grey Wolf Optimizer with sigmoid."""
    pop_size = params.get('pop_size', 30)  # population size
    max_iter = params.get('max_iter', 20)  # iterations

    positions = np.random.randint(2, size=(pop_size, n_feats))  # initial positions
    fitness = np.array([evaluator.evaluar(ind)[0] for ind in positions])  # initial fitness

    sorted_indices = np.argsort(fitness)[::-1]  # sort descending
    alpha_pos, alpha_score = positions[sorted_indices[0]].copy(), fitness[sorted_indices[0]]  # alpha
    beta_pos, beta_score = positions[sorted_indices[1]].copy(), fitness[sorted_indices[1]]  # beta
    delta_pos, delta_score = positions[sorted_indices[2]].copy(), fitness[sorted_indices[2]]  # delta

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # decreasing parameter

        for i in range(pop_size):
            for j in range(n_feats):
                r1, r2 = np.random.random(), np.random.random()  # random values
                A1 = 2 * a * r1 - a  # coefficient A1
                C1 = 2 * r2  # coefficient C1
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])  # distance to alpha
                X1 = alpha_pos[j] - A1 * D_alpha  # alpha contribution

                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta  # beta contribution

                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta  # delta contribution

                X_continuous = (X1 + X2 + X3) / 3  # continuous average

                sigmoid = 1 / (1 + np.exp(-10 * (X_continuous - 0.5)))  # sigmoid for binarization
                positions[i, j] = 1 if np.random.random() < sigmoid else 0  # binarize position

            fit = evaluator.evaluar(positions[i])[0]  # evaluate position
            if fit > alpha_score:
                alpha_score, alpha_pos = fit, positions[i].copy()  # update alpha
            elif fit > beta_score:
                beta_score, beta_pos = fit, positions[i].copy()  # update beta
            elif fit > delta_score:
                delta_score, delta_pos = fit, positions[i].copy()  # update delta

    return alpha_pos.tolist(), alpha_score  # return best wolf and its fitness