import random  
import numpy as np  
from deap import base, creator, tools, algorithms  

# crear clase de fitness (maximización) si no existe
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # fitness a maximizar
# crear clase de individuo si no existe
if not hasattr(creator, "Individuo"):
    creator.create("Individuo", list, fitness=creator.FitnessMax)  # individuo basado en lista


def run_ga(evaluator, n_feats, params):
    """Algoritmo genético con DEAP."""
    pop_size = params.get('pop_size', 50)  # tamaño población
    n_gen = params.get('n_gen', 50)  # generaciones
    p_cx = params.get('p_cruce', 0.5)  # probabilidad cruce
    p_mut = params.get('p_mutacion', 0.2)  # probabilidad mutación
    tam_torneo = params.get('tam_torneo', 3)  # tamaño torneo

    toolbox = base.Toolbox()  # crear toolbox
    toolbox.register("attr_bool", random.randint, 0, 1)  # bit aleatorio 0/1
    toolbox.register("individual", tools.initRepeat, creator.Individuo, toolbox.attr_bool, n_feats)  # individuo
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # población

    toolbox.register("evaluate", evaluator.evaluar)  # función de evaluación

    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # cruce uniforme
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)  # mutación flip bit
    toolbox.register("select", tools.selTournament, tournsize=tam_torneo)  # selección por torneo

    pop = toolbox.population(n=pop_size)  # inicializar población
    hof = tools.HallOfFame(1)  # guardar mejor individuo
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # recolectar fitness
    stats.register("avg", np.mean)  # registrar media
    stats.register("max", np.max)  # registrar máximo

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen,
                             stats=stats, halloffame=hof, verbose=False)  # ejecutar algoritmo

    return hof[0], hof[0].fitness.values[0], log  # devolver mejor individuo, su fitness y log


def run_sa(evaluator, n_feats, params):
    """Simulated Annealing (enfriamiento geométrico)."""
    max_iter = params.get('max_iter', 1000)  # iteraciones
    T = params.get('temp_init', 1.0)  # temperatura inicial
    alpha = params.get('alpha', 0.99)  # factor de enfriamiento

    current_sol = [random.randint(0, 1) for _ in range(n_feats)]  # solución inicial aleatoria
    current_fit = evaluator.evaluar(current_sol)[0]  # fitness actual
    best_sol = list(current_sol)  # mejor solución encontrada
    best_fit = current_fit  # fitness de la mejor

    for _ in range(max_iter):
        neighbor = list(current_sol)  # copiar solución
        idx = random.randint(0, n_feats - 1)  # índice a mutar
        neighbor[idx] = 1 - neighbor[idx]  # flip bit en vecino

        neighbor_fit = evaluator.evaluar(neighbor)[0]  # evaluar vecino

        delta = neighbor_fit - current_fit  # diferencia de fitness
        if delta > 0 or random.random() < np.exp(delta / T):  # criterio Metropolis
            current_sol = neighbor  # aceptar vecino
            current_fit = neighbor_fit  # actualizar fitness actual

            if current_fit > best_fit:  # actualizar mejor si mejora
                best_fit = current_fit
                best_sol = list(current_sol)

        T *= alpha  # enfriar temperatura

    return best_sol, best_fit  # devolver mejor solución y su fitness


def run_tabu(evaluator, n_feats, params):
    """Tabu Search simple con aspiración."""
    max_iter = params.get('max_iter', 200)  # iteraciones
    tabu_size = params.get('tabu_size', 10)  # tamaño lista tabu
    n_neighbors = params.get('n_neighbors', 10)  # vecinos por iteración

    current_sol = [random.randint(0, 1) for _ in range(n_feats)]  # solución inicial
    best_sol = list(current_sol)  # mejor solución
    best_fit = evaluator.evaluar(current_sol)[0]  # fitness mejor
    tabu_list = []  # lista tabu (índices de movimientos)

    for _ in range(max_iter):
        candidates = []  # lista de candidatos (vecino, fit, movimiento)
        for _ in range(n_neighbors):
            neighbor = list(current_sol)  # copiar solución
            move_idx = random.randint(0, n_feats - 1)  # índice a cambiar
            neighbor[move_idx] = 1 - neighbor[move_idx]  # aplicar movimiento

            fit = evaluator.evaluar(neighbor)[0]  # evaluar vecino

            if move_idx not in tabu_list or fit > best_fit:  # criterio aspiración
                candidates.append((neighbor, fit, move_idx))  # añadir candidato

        if not candidates:
            continue  # sin candidatos válidos

        best_candidate = max(candidates, key=lambda x: x[1])  # elegir mejor por fitness
        current_sol, fit_val, move_idx = best_candidate  # actualizar solución actual

        if fit_val > best_fit:  # actualizar global si mejora
            best_fit = fit_val
            best_sol = list(current_sol)

        tabu_list.append(move_idx)  # añadir movimiento a tabu
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)  # mantener tamaño

    return best_sol, best_fit  # devolver mejor solución y su fitness


def run_pso(evaluator, n_feats, params):
    """Binary PSO con binarización sigmoide."""
    n_particles = params.get('n_particles', 30)  # número partículas
    max_iter = params.get('max_iter', 50)  # iteraciones
    w = params.get('w', 0.7)  # inercia
    c1 = params.get('c1', 1.5)  # coef social
    c2 = params.get('c2', 1.5)  # coef cognitivo

    X = np.random.randint(2, size=(n_particles, n_feats))  # posiciones binarias
    V = np.random.uniform(-1, 1, size=(n_particles, n_feats))  # velocidades continuas

    P_best = X.copy()  # mejores personales
    P_best_fit = np.array([evaluator.evaluar(ind)[0] for ind in X])  # fitness personales

    g_best_idx = np.argmax(P_best_fit)  # índice mejor global
    G_best = P_best[g_best_idx].copy()  # posición global mejor
    G_best_fit = P_best_fit[g_best_idx]  # fitness global mejor

    for _ in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()  # aleatorios
            V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])  # actualizar velocidad

            V[i] = np.clip(V[i], -10, 10)  # evitar overflow
            sigmoid = 1 / (1 + np.exp(-V[i]))  # sigmoide por componente

            X[i] = (np.random.rand(n_feats) < sigmoid).astype(int)  # actualizar posición binaria

            fit = evaluator.evaluar(X[i])[0]  # evaluar partícula

            if fit > P_best_fit[i]:  # actualizar personal best
                P_best_fit[i] = fit
                P_best[i] = X[i].copy()

                if fit > G_best_fit:  # actualizar global si mejora
                    G_best_fit = fit
                    G_best = X[i].copy()

    return G_best.tolist(), G_best_fit  # devolver mejor global


def run_gwo(evaluator, n_feats, params):
    """Grey Wolf Optimizer binario con sigmoide."""
    pop_size = params.get('pop_size', 30)  # tamaño población
    max_iter = params.get('max_iter', 20)  # iteraciones

    positions = np.random.randint(2, size=(pop_size, n_feats))  # posiciones iniciales
    fitness = np.array([evaluator.evaluar(ind)[0] for ind in positions])  # fitness inicial

    sorted_indices = np.argsort(fitness)[::-1]  # ordenar descendente
    alpha_pos, alpha_score = positions[sorted_indices[0]].copy(), fitness[sorted_indices[0]]  # alpha
    beta_pos, beta_score = positions[sorted_indices[1]].copy(), fitness[sorted_indices[1]]  # beta
    delta_pos, delta_score = positions[sorted_indices[2]].copy(), fitness[sorted_indices[2]]  # delta

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # parámetro que decrece

        for i in range(pop_size):
            for j in range(n_feats):
                r1, r2 = np.random.random(), np.random.random()  # aleatorios
                A1 = 2 * a * r1 - a  # coef A1
                C1 = 2 * r2  # coef C1
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])  # distancia alpha
                X1 = alpha_pos[j] - A1 * D_alpha  # contribución alpha

                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta  # contribución beta

                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta  # contribución delta

                X_continuous = (X1 + X2 + X3) / 3  # promedio continuo

                sigmoid = 1 / (1 + np.exp(-10 * (X_continuous - 0.5)))  # sigmoide para binarizar
                positions[i, j] = 1 if np.random.random() < sigmoid else 0  # binarizar posición

            fit = evaluator.evaluar(positions[i])[0]  # evaluar posición
            if fit > alpha_score:
                alpha_score, alpha_pos = fit, positions[i].copy()  # actualizar alpha
            elif fit > beta_score:
                beta_score, beta_pos = fit, positions[i].copy()  # actualizar beta
            elif fit > delta_score:
                delta_score, delta_pos = fit, positions[i].copy()  # actualizar delta

    return alpha_pos.tolist(), alpha_score  # devolver mejor lobo y su fitness