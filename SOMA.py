import numpy as np
import random
import matplotlib.pyplot as plt

# 1. BENCHMARK FUNCTIONS (10D) AND THEIR BOUNDS
def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

BENCHMARKS = {
    'Sphere': {'func': sphere_function, 'bounds': (-5.12, 5.12)},
    'Rosenbrock': {'func': rosenbrock_function, 'bounds': (-2.048, 2.048)},
    'Rastrigin': {'func': rastrigin_function, 'bounds': (-5.12, 5.12)}
}

# 2. SOMA ALGORITHM IMPLEMENTATION (All-To-Best, All-To-Random, All-To-All)
def soma_algorithm(fit_func, dim, bounds, strategy='All-To-Best',
                   max_evals=10000, pop_size=25,
                   prt=0.3, path_length=3.0, step=0.11):
    lower_b, upper_b = bounds
    
    # Smaller population size for All-To-All to ensure enough migration cycles within MAXFES
    if strategy == 'All-To-All':
        pop_size = 15

    # Initialize population
    pop = np.random.uniform(lower_b, upper_b, (pop_size, dim))
    fitnesses = np.array([fit_func(ind) for ind in pop])
    
    evals = pop_size
    best_fit_so_far = np.min(fitnesses)
    history = [best_fit_so_far] * evals

    # Step sequence t from Step to PathLength
    steps = np.arange(step, path_length + 1e-9, step)

    def migrate_individual(migrant_pos, leader_pos, current_best_pos, current_best_fit):
        nonlocal evals, best_fit_so_far
        best_step_pos = current_best_pos.copy()
        best_step_fit = current_best_fit

        for t in steps:
            if evals >= max_evals:
                break
            
            # Generate PRTVector for each step
            prt_vector = (np.random.uniform(0, 1, dim) < prt).astype(float)
            if np.sum(prt_vector) == 0:
                prt_vector[random.randint(0, dim - 1)] = 1.0

            # Migration equation
            candidate = migrant_pos + (leader_pos - migrant_pos) * t * prt_vector
            candidate = np.clip(candidate, lower_b, upper_b)

            # Evaluate candidate
            f_val = fit_func(candidate)
            evals += 1

            if f_val < best_step_fit:
                best_step_fit = f_val
                best_step_pos = candidate.copy()

            if f_val < best_fit_so_far:
                best_fit_so_far = f_val

            history.append(best_fit_so_far)

        return best_step_pos, best_step_fit

    # Main migration loop
    while evals < max_evals:
        if strategy == 'All-To-Best':
            leader_idx = np.argmin(fitnesses)
            leader_pos = pop[leader_idx].copy()
            
            for i in range(pop_size):
                if evals >= max_evals:
                    break
                if i == leader_idx:
                    continue
                new_pos, new_fit = migrate_individual(pop[i], leader_pos, pop[i], fitnesses[i])
                pop[i] = new_pos
                fitnesses[i] = new_fit

        elif strategy == 'All-To-Random':
            leader_idx = random.randint(0, pop_size - 1)
            leader_pos = pop[leader_idx].copy()
            
            for i in range(pop_size):
                if evals >= max_evals:
                    break
                if i == leader_idx:
                    continue
                new_pos, new_fit = migrate_individual(pop[i], leader_pos, pop[i], fitnesses[i])
                pop[i] = new_pos
                fitnesses[i] = new_fit

        elif strategy == 'All-To-All':
            start_pop = pop.copy()
            next_pop = pop.copy()
            next_fitnesses = fitnesses.copy()

            for i in range(pop_size):
                if evals >= max_evals:
                    break
                for l_idx in range(pop_size):
                    if evals >= max_evals:
                        break
                    if i == l_idx:
                        continue
                    best_pos, best_fit = migrate_individual(
                        start_pop[i], start_pop[l_idx], next_pop[i], next_fitnesses[i]
                    )
                    next_pop[i] = best_pos
                    next_fitnesses[i] = best_fit

            pop = next_pop
            fitnesses = next_fitnesses

    return history[:max_evals], best_fit_so_far

# 3. RUN EXPERIMENTS AND PLOT GRAPHS & TABLE IN A SINGLE WINDOW
def run_soma_experiments():
    dim = 10
    max_evals = 10000
    runs = 10
    pop_size = 25
    
    soma_variants = [
        ('All-To-Best', 'All-To-Best'),
        ('All-To-Random', 'All-To-Random'),
        ('All-To-All', 'All-To-All')
    ]
    
    result_rows = []
    avg_histories_dict = {f_name: {} for f_name in BENCHMARKS}

    for f_idx, (f_name, f_info) in enumerate(BENCHMARKS.items()):
        print(f"---> Running SOMA for {f_name} ({dim}D)...", flush=True)
        
        for v_label, v_type in soma_variants:
            all_histories = []
            final_bests = []
            
            for r in range(runs):
                np.random.seed(r * 100 + f_idx * 10)
                random.seed(r * 100 + f_idx * 10)
                
                hist, best = soma_algorithm(
                    f_info['func'], dim, f_info['bounds'], strategy=v_type,
                    max_evals=max_evals, pop_size=pop_size
                )
                
                all_histories.append(hist)
                final_bests.append(best)
                
            avg_histories_dict[f_name][v_label] = np.mean(all_histories, axis=0)
            best_res = float(np.min(final_bests))
            mean_res = float(np.mean(final_bests))
            std_res = float(np.std(final_bests))
            
            result_rows.append((f_name, v_label, best_res, mean_res, std_res))

    # Print results summary to console
    print("\n" + "=" * 85)
    print(f"{'Function':<12} | {'SOMA Strategy':<20} | {'Best Run':>14} | {'Mean':>14} | {'Std. Dev.':>14}")
    print("=" * 85)
    for f_name, v_label, best_res, mean_res, std_res in result_rows:
        print(f"{f_name:<12} | {v_label:<20} | {best_res:14.4e} | {mean_res:14.4e} | {std_res:14.4e}")
    print("=" * 85)

    # --- UNIFIED WINDOW FOR CONVERGENCE PLOTS AND SUMMARY TABLE ---
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Top row: 3 convergence graphs
    for f_idx, (f_name, _) in enumerate(BENCHMARKS.items()):
        ax = fig.add_subplot(gs[0, f_idx])
        
        for v_label, _ in soma_variants:
            ax.plot(avg_histories_dict[f_name][v_label], label=v_label, linewidth=1.5)
            
        ax.set_yscale('log')
        ax.set_title(f"SOMA Convergence: {f_name} ({dim}D)")
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Fitness (Log Scale)")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)

    # Bottom row: Summary table
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis('off')
    table_ax.set_title(f'SOMA Results Summary ({runs} runs per strategy)', pad=10, fontsize=12, fontweight='bold')
    
    table_data = [
        [f_name, v_label, f"{best:.4e}", f"{mean:.4e}", f"{std:.4e}"]
        for f_name, v_label, best, mean, std in result_rows
    ]
    
    table = table_ax.table(
        cellText=table_data,
        colLabels=['Function', 'SOMA Strategy', 'Best Run', 'Mean', 'Std. Deviation'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.35, 0.16, 0.16, 0.16]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.show()

if __name__ == "__main__":
    run_soma_experiments()