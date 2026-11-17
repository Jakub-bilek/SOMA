# SOMA
Parameters: Dim = 10, MAXFES = 10000, Runs = 10, PopSize = 25 (15 for All-To-All), PathLength = 3.0, Step = 0.11, PRT = 0.3.
Reflection for each graph:

Sphere: All-To-Best achieves the fastest and deepest convergence because all individuals directly migrate toward the single best leader, maximizing exploitation on this smooth, unimodal landscape. All-To-All converges more slowly due to its high evaluation cost per migration cycle PopSize*(PopSize-1), while All-To-Random lacks directional pressure.

Rosenbrock: All-To-Best again outperforms the other strategies by efficiently pulling the population along the narrow curved valley toward the best-known solution. All-To-All provides steady progress but runs out of evaluation budget before fine-tuning the valley floor.   

Rastrigin: Unlike the unimodal functions, All-To-Best is prone to premature convergence in local optima because the entire swarm collapses toward a single leader. All-To-Random and All-To-All preserve higher population diversity through stochastic and multi-leader migrations, allowing individuals to escape local traps more effectively.

AI was used for grammar, bug fixing, and initial structuring of the study material.
