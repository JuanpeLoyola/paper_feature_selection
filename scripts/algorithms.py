import random
import numpy as np
from deap import base, creator, tools, algorithms

# Configuración DEAP para GA
# Evitamos re-crear la clase si ya existe para evitar warnings en ejecuciones repetidas
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individuo"):
    creator.create("Individuo", list, fitness=creator.FitnessMax)


def run_ga(evaluator, n_feats, params):
    """Algoritmo Genético usando DEAP."""
    # 1. Extraer parámetros del diccionario
    pop_size = params.get('pop_size', 50)
    n_gen = params.get('n_gen', 50)
    p_cx = params.get('p_cruce', 0.5)
    p_mut = params.get('p_mutacion', 0.2)
    tam_torneo = params.get('tam_torneo', 3)

    # Configuración del toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individuo, toolbox.attr_bool, n_feats)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # CORRECCIÓN: Usar .evaluar (español)
    toolbox.register("evaluate", evaluator.evaluar)
    
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)
    toolbox.register("select", tools.selTournament, tournsize=tam_torneo)
    
    # Inicialización
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Evolución
    # CORRECCIÓN: Pasar n_gen explícitamente extraído del diccionario
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                             cxpb=p_cx, mutpb=p_mut, ngen=n_gen, 
                             stats=stats, halloffame=hof, verbose=False)
    
    return hof[0], hof[0].fitness.values[0], log


def run_sa(evaluator, n_feats, params):
    """Simulated Annealing con enfriamiento geométrico."""
    # 1. Extraer parámetros
    max_iter = params.get('max_iter', 1000)
    T = params.get('temp_init', 1.0)
    alpha = params.get('alpha', 0.99)

    # Solución inicial aleatoria
    current_sol = [random.randint(0, 1) for _ in range(n_feats)]
    # CORRECCIÓN: Usar .evaluar
    current_fit = evaluator.evaluar(current_sol)[0]
    best_sol = list(current_sol)
    best_fit = current_fit
    
    for _ in range(max_iter):
        # Generar vecino (flip 1 bit aleatorio)
        neighbor = list(current_sol)
        idx = random.randint(0, n_feats - 1)
        neighbor[idx] = 1 - neighbor[idx]
        
        # CORRECCIÓN: Usar .evaluar
        neighbor_fit = evaluator.evaluar(neighbor)[0]
        
        # Criterio de aceptación (Metropolis)
        delta = neighbor_fit - current_fit
        # Maximizamos, así que si neighbor_fit > current_fit, delta es positivo
        if delta > 0 or random.random() < np.exp(delta / T):
            current_sol = neighbor
            current_fit = neighbor_fit
            
            if current_fit > best_fit:
                best_fit = current_fit
                best_sol = list(current_sol)
        
        T *= alpha
        
    return best_sol, best_fit


def run_tabu(evaluator, n_feats, params):
    """Tabu Search con lista tabu simple y criterio de aspiración."""
    # 1. Extraer parámetros
    max_iter = params.get('max_iter', 200)
    tabu_size = params.get('tabu_size', 10)
    n_neighbors = params.get('n_neighbors', 10)

    # Solución inicial
    current_sol = [random.randint(0, 1) for _ in range(n_feats)]
    best_sol = list(current_sol)
    # CORRECCIÓN: Usar .evaluar
    best_fit = evaluator.evaluar(current_sol)[0]
    tabu_list = []
    
    for _ in range(max_iter):
        # Explorar vecindario
        candidates = []
        for _ in range(n_neighbors):
            neighbor = list(current_sol)
            move_idx = random.randint(0, n_feats - 1)
            neighbor[move_idx] = 1 - neighbor[move_idx]
            
            # CORRECCIÓN: Usar .evaluar
            fit = evaluator.evaluar(neighbor)[0]
            
            # Criterio de aspiración: permitir movimiento tabu si mejora el mejor global
            if move_idx not in tabu_list or fit > best_fit:
                candidates.append((neighbor, fit, move_idx))
        
        if not candidates:
            continue
        
        # Seleccionar mejor candidato
        best_candidate = max(candidates, key=lambda x: x[1])
        current_sol, fit_val, move_idx = best_candidate
        
        if fit_val > best_fit:
            best_fit = fit_val
            best_sol = list(current_sol)
        
        # Actualizar lista tabu
        tabu_list.append(move_idx)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
            
    return best_sol, best_fit


def run_pso(evaluator, n_feats, params):
    """Binary Particle Swarm Optimization con función sigmoide."""
    # 1. Extraer parámetros
    n_particles = params.get('n_particles', 30)
    max_iter = params.get('max_iter', 50)
    w = params.get('w', 0.7)
    c1 = params.get('c1', 1.5)
    c2 = params.get('c2', 1.5)

    # Inicialización de posiciones y velocidades
    X = np.random.randint(2, size=(n_particles, n_feats))
    V = np.random.uniform(-1, 1, size=(n_particles, n_feats))
    
    # Personal best
    P_best = X.copy()
    # CORRECCIÓN: Usar .evaluar
    P_best_fit = np.array([evaluator.evaluar(ind)[0] for ind in X])
    
    # Global best
    g_best_idx = np.argmax(P_best_fit)
    G_best = P_best[g_best_idx].copy()
    G_best_fit = P_best_fit[g_best_idx]
    
    for _ in range(max_iter):
        for i in range(n_particles):
            # Actualizar velocidad
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = w*V[i] + c1*r1*(P_best[i] - X[i]) + c2*r2*(G_best - X[i])
            
            # Binarización con sigmoide
            # Clip para evitar overflow en exp
            V[i] = np.clip(V[i], -10, 10) 
            sigmoid = 1 / (1 + np.exp(-V[i]))
            
            X[i] = (np.random.rand(n_feats) < sigmoid).astype(int)
            
            # Evaluar y actualizar personal best
            # CORRECCIÓN: Usar .evaluar
            fit = evaluator.evaluar(X[i])[0]
            
            if fit > P_best_fit[i]:
                P_best_fit[i] = fit
                P_best[i] = X[i].copy()
                
                # Actualizar global best
                if fit > G_best_fit:
                    G_best_fit = fit
                    G_best = X[i].copy()
                    
    return G_best.tolist(), G_best_fit


def run_gwo(evaluator, n_feats, params):
    """Grey Wolf Optimizer con binarización mediante sigmoide."""
    pop_size = params.get('pop_size', 30)
    max_iter = params.get('max_iter', 20)
    
    # Inicialización de población
    positions = np.random.randint(2, size=(pop_size, n_feats))
    # CORRECCIÓN: Usar .evaluar
    fitness = np.array([evaluator.evaluar(ind)[0] for ind in positions])
    
    # Jerarquía inicial (alpha, beta, delta = 3 mejores lobos)
    sorted_indices = np.argsort(fitness)[::-1]
    alpha_pos, alpha_score = positions[sorted_indices[0]].copy(), fitness[sorted_indices[0]]
    beta_pos, beta_score = positions[sorted_indices[1]].copy(), fitness[sorted_indices[1]]
    delta_pos, delta_score = positions[sorted_indices[2]].copy(), fitness[sorted_indices[2]]
    
    for t in range(max_iter):
        # Parámetro a decrece linealmente de 2 a 0
        a = 2 - t * (2 / max_iter)
        
        for i in range(pop_size):
            for j in range(n_feats):
                # Movimiento hacia alpha
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # Movimiento hacia beta
                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # Movimiento hacia delta
                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # Posición promedio
                X_continuous = (X1 + X2 + X3) / 3
                
                # Binarización con sigmoide
                sigmoid = 1 / (1 + np.exp(-10 * (X_continuous - 0.5)))
                positions[i, j] = 1 if np.random.random() < sigmoid else 0
            
            # Evaluar y actualizar jerarquía
            # CORRECCIÓN: Usar .evaluar
            fit = evaluator.evaluar(positions[i])[0]
            if fit > alpha_score:
                alpha_score, alpha_pos = fit, positions[i].copy()
            elif fit > beta_score:
                beta_score, beta_pos = fit, positions[i].copy()
            elif fit > delta_score:
                delta_score, delta_pos = fit, positions[i].copy()
                
    return alpha_pos.tolist(), alpha_score