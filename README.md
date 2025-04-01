# Group Formation in Virtual Learning Environments using Genetic Algorithms

This project implements a Genetic Algorithm (GA) for optimizing the formation of student groups in Virtual Learning Environments (VLE). The implementation is based on the approach described in the paper "Uma Abordagem Baseada em Algoritmo Genético para Formação de Grupos de Estudos em Ambientes Virtuais de Aprendizagem" (A Genetic Algorithm-Based Approach for Study Group Formation in Virtual Learning Environments).

## Overview

The algorithm creates optimal student groups based on student characteristics (such as age, access time, study duration, and discipline preference), by minimizing the Euclidean distance between students and their group centroids. This clustering-based approach helps create more homogeneous groups, which can potentially improve collaborative learning experiences.

## Features

- Genetic algorithm implementation with configurable parameters
- Data normalization for fair comparison between different student characteristics
- Comparison with random group formation method
- Visualization of fitness improvement across generations
- Configurable number of groups and population size

## How It Works

1. **Initialization**: Creates an initial population of potential group formations
2. **Fitness Calculation**: Evaluates solutions based on the sum of Euclidean distances
3. **Selection**: Uses roulette wheel selection to choose parent solutions
4. **Crossover**: Implements uniform crossover to create new solutions
5. **Mutation**: Applies gene-based mutation to maintain diversity
6. **Elitism**: Preserves the best solution across generations

## Usage Example

```python
# Example parameters
num_alunos = 100  # Number of students
num_grupos = 5    # Number of groups to form

# Generate synthetic data
idade = np.random.uniform(10, 60, num_alunos).reshape(-1, 1)  # Age
hora_acesso = np.random.uniform(0, 23, num_alunos).reshape(-1, 1)  # Access hour
tempo_acesso = np.random.uniform(1, 90, num_alunos).reshape(-1, 1)  # Access duration
disciplina = np.random.randint(1, 5, num_alunos).reshape(-1, 1)  # Course/discipline

# Combine characteristics
dados_alunos = np.hstack((idade, hora_acesso, tempo_acesso, disciplina))

# Normalize data
dados_normalizados = normalizar_dados(dados_alunos)

# Create and run the GA
ag = AlgoritmoGeneticoAVA(
    dados_normalizados,
    num_grupos,
    tam_populacao=100,
    pc=0.1,  # Crossover probability
    pm=0.03,  # Mutation probability
    num_geracoes=100  # Number of generations
)

# Get results
melhor_solucao, melhor_fitness = ag.executar()
```

## Performance Comparison

The algorithm automatically compares its performance against random group formation, demonstrating significant improvements in group homogeneity. The example implementation includes a method for running multiple instances and calculating average performance.

## Visualization

The implementation includes a method to plot the evolution of fitness values across generations, allowing you to visualize the algorithm's convergence.

## Customization

You can customize various parameters of the genetic algorithm:
- Population size
- Number of generations
- Crossover probability
- Mutation probability
- Number of groups to form

You can also modify the student characteristics or add new ones according to your specific VLE requirements.
"# ai_analise_grupo" 
