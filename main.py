import pygad
import numpy as np


#equation: a + 2b + 3c + 4c = 30
function_inputs = [1, 2, 3, 4] 
desired_output = 30 

def fitness_func(solution, solution_idx):
    output = np.sum(solution*function_inputs)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

num_generations = 30
num_parents_mating = 2

crossover_probability = 0.5
mutation_probability = 0.375

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       crossover_probability=crossover_probability,
                       mutation_probability=mutation_probability,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                    )
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Result of best solution: {sum}".format(sum=np.sum(solution*function_inputs)))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))