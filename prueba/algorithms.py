import random
import numpy as np
from deap import base, creator, tools, algorithms

# --- CONFIGURACIÓN DEAP (GA) ---
# Se define fuera para evitar errores de re-definición al importar
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMax)

# ==========================================
# 1. GENETIC ALGORITHM (GA)
# ==========================================
def run_ga(evaluator, n_feats, n_gen=50, pop_size=50):
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individuo, toolbox.attr_bool, n_feats)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Conectamos con la clase Evaluator
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/n_feats)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                         cxpb=0.7, mutpb=0.2, ngen=n_gen, 
                                         stats=stats, halloffame=hof, verbose=False)
    
    return hof[0], hof[0].fitness.values[0]

# ==========================================
# 2. SIMULATED ANNEALING (SA)
# ==========================================
def run_sa(evaluator, n_feats, max_iter=1000):
    # Configuración inicial
    current_sol = [random.randint(0, 1) for _ in range(n_feats)]
    current_fit = evaluator.evaluate(current_sol)[0]
    
    best_sol = list(current_sol)
    best_fit = current_fit
    
    T = 1.0
    alpha = 0.99 # Factor de enfriamiento
    
    for i in range(max_iter):
        # Generar vecino (Flip 1 bit)
        neighbor = list(current_sol)
        idx = random.randint(0, n_feats - 1)
        neighbor[idx] = 1 - neighbor[idx]
        
        neighbor_fit = evaluator.evaluate(neighbor)[0]
        
        # Criterio de aceptación
        delta = neighbor_fit - current_fit
        if delta > 0 or random.random() < np.exp(delta / T):
            current_sol = neighbor
            current_fit = neighbor_fit
            
            if current_fit > best_fit:
                best_fit = current_fit
                best_sol = list(current_sol)
        
        T *= alpha # Enfriar
        
    return best_sol, best_fit

# ==========================================
# 3. TABU SEARCH (TS)
# ==========================================
def run_tabu(evaluator, n_feats, max_iter=200, tabu_size=10):
    current_sol = [random.randint(0, 1) for _ in range(n_feats)]
    best_sol = list(current_sol)
    best_fit = evaluator.evaluate(current_sol)[0]
    
    tabu_list = [] # Guardaremos índices de features modificadas recientemente
    
    for _ in range(max_iter):
        # Explorar vecindario (ej: 20 vecinos aleatorios)
        candidates = []
        for _ in range(20):
            neighbor = list(current_sol)
            move_idx = random.randint(0, n_feats - 1)
            neighbor[move_idx] = 1 - neighbor[move_idx]
            
            # Verificar si es Tabu (simplificado: prohibimos mover el mismo indice)
            is_tabu = move_idx in tabu_list
            
            fit = evaluator.evaluate(neighbor)[0]
            
            # Criterio de Aspiración: Si es tabu pero mejora el global, lo permitimos
            if not is_tabu or fit > best_fit:
                candidates.append((neighbor, fit, move_idx))
        
        if not candidates: continue
        
        # Elegir el mejor candidato
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = candidates[0]
        
        current_sol = best_candidate[0]
        move_idx = best_candidate[2]
        
        if best_candidate[1] > best_fit:
            best_fit = best_candidate[1]
            best_sol = list(current_sol)
            
        # Actualizar lista Tabu
        tabu_list.append(move_idx)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
            
    return best_sol, best_fit

# ==========================================
# 4. PARTICLE SWARM OPTIMIZATION (Binary PSO)
# ==========================================
def run_pso(evaluator, n_feats, n_particles=30, max_iter=50):
    # Inicialización
    # X: Posiciones (0 o 1)
    # V: Velocidades (Reales)
    X = np.random.randint(2, size=(n_particles, n_feats))
    V = np.random.uniform(-1, 1, size=(n_particles, n_feats))
    
    P_best = X.copy()
    P_best_fit = np.array([evaluator.evaluate(ind)[0] for ind in X])
    
    g_best_idx = np.argmax(P_best_fit)
    G_best = P_best[g_best_idx].copy()
    G_best_fit = P_best_fit[g_best_idx]
    
    # Hiperparámetros PSO
    w = 0.7  # Inercia
    c1 = 1.5 # Cognitivo
    c2 = 1.5 # Social
    
    for _ in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Actualizar Velocidad
            V[i] = w*V[i] + c1*r1*(P_best[i] - X[i]) + c2*r2*(G_best - X[i])
            
            # Sigmoide para binarizar: S(v) = 1 / (1 + e^-v)
            sigmoid = 1 / (1 + np.exp(-V[i]))
            
            # Actualizar Posición (Regla probabilística BPSO)
            mask_rand = np.random.rand(n_feats)
            X[i] = (mask_rand < sigmoid).astype(int)
            
            # Evaluar
            fit = evaluator.evaluate(X[i])[0]
            
            # Actualizar Personal Best
            if fit > P_best_fit[i]:
                P_best_fit[i] = fit
                P_best[i] = X[i].copy()
                
                # Actualizar Global Best
                if fit > G_best_fit:
                    G_best_fit = fit
                    G_best = X[i].copy()
                    
    return G_best.tolist(), G_best_fit