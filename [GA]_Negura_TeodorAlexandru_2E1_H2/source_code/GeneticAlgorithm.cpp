#include "GeneticAlgorithm.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>

// Constructor implementation
GeneticAlgorithm::GeneticAlgorithm(int popSize, int dim, int maxGen, double mutRate, double crossRate, 
                                   double low, double high, 
                                   std::function<double(const std::vector<double>&)> func)
    : populationSize(popSize), dimension(dim), maxGenerations(maxGen), 
      mutationRate(mutRate), crossoverRate(crossRate), 
      lowerBound(low), upperBound(high), objectiveFunction(func) {
    
    // Seed the random number generator once with a random device
    std::random_device rd;
    rng.seed(rd());
}

// Helper to keep genes inside [lowerBound, upperBound]
double GeneticAlgorithm::clamp(double value) {
    if (value < lowerBound) return lowerBound;
    if (value > upperBound) return upperBound;
    return value;
}

void GeneticAlgorithm::initializePopulation() {
    population.clear();
    population.resize(populationSize);

    std::uniform_real_distribution<double> dist(lowerBound, upperBound);

    for (int i = 0; i < populationSize; ++i) {
        population[i].genes.resize(dimension);
        for (int j = 0; j < dimension; ++j) {
            population[i].genes[j] = dist(rng);
        }
        // Initial fitness calculation
        population[i].fitness = objectiveFunction(population[i].genes);
    }
}

void GeneticAlgorithm::evaluatePopulation() {
    for (auto& ind : population) {
        ind.fitness = objectiveFunction(ind.genes);
    }
}

Individual GeneticAlgorithm::tournamentSelection(int k) {
    // Randomly select 'k' individuals and return the best one.
    std::uniform_int_distribution<int> dist(0, populationSize - 1);
    
    int bestIdx = dist(rng);
    double bestFitness = population[bestIdx].fitness;

    for (int i = 1; i < k; ++i) {
        int idx = dist(rng);
        if (population[idx].fitness < bestFitness) { // Minimization
            bestFitness = population[idx].fitness;
            bestIdx = idx;
        }
    }
    return population[bestIdx];
}

// Simulated Binary Crossover (SBX)
// Adapted for real-coded GA to mimic binary crossover properties.
void GeneticAlgorithm::crossoverSBX(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2) {
    c1 = p1;
    c2 = p2;

    // Check crossover probability
    std::uniform_real_distribution<double> probDist(0.0, 1.0);
    if (probDist(rng) > crossoverRate) {
        return; // No crossover, children are copies of parents
    }

    double eta_c = 20.0; // Distribution index (higher = children closer to parents)
    std::uniform_real_distribution<double> randDist(0.0, 1.0);

    for (int i = 0; i < dimension; ++i) {
        double u = randDist(rng);
        if (u <= 0.5) {
            double beta = std::pow(2.0 * u, 1.0 / (eta_c + 1.0));
            c1.genes[i] = 0.5 * ((1.0 + beta) * p1.genes[i] + (1.0 - beta) * p2.genes[i]);
            c2.genes[i] = 0.5 * ((1.0 - beta) * p1.genes[i] + (1.0 + beta) * p2.genes[i]);
        } else {
            double beta = std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta_c + 1.0));
            c1.genes[i] = 0.5 * ((1.0 + beta) * p1.genes[i] + (1.0 - beta) * p2.genes[i]);
            c2.genes[i] = 0.5 * ((1.0 - beta) * p1.genes[i] + (1.0 + beta) * p2.genes[i]);
        }
        // Clamp to bounds immediately
        c1.genes[i] = clamp(c1.genes[i]);
        c2.genes[i] = clamp(c2.genes[i]);
    }
}

// Polynomial Mutation
// Allows for small adjustments to refine solutions (exploitation) 
// and occasional large jumps (exploration).
void GeneticAlgorithm::mutatePolynomial(Individual& ind) {
    double eta_m = 20.0; // Distribution index - Reverted to 20 for better exploration (Rastrigin needs this)
    std::uniform_real_distribution<double> probDist(0.0, 1.0);
    std::uniform_real_distribution<double> randDist(0.0, 1.0);

    for (int i = 0; i < dimension; ++i) {
        if (probDist(rng) < mutationRate) {
            double u = randDist(rng);
            double delta;
            if (u <= 0.5) {
                delta = std::pow(2.0 * u, 1.0 / (eta_m + 1.0)) - 1.0;
            } else {
                delta = 1.0 - std::pow(2.0 * (1.0 - u), 1.0 / (eta_m + 1.0));
            }
            
            // Apply mutation based on the total range
            ind.genes[i] += delta * (upperBound - lowerBound);
            ind.genes[i] = clamp(ind.genes[i]);
        }
    }
}

Individual GeneticAlgorithm::run() {
    initializePopulation();

    // Track the global best found so far
    Individual globalBest = population[0];
    for (const auto& ind : population) {
        if (ind.fitness < globalBest.fitness) {
            globalBest = ind;
        }
    }

    for (int gen = 0; gen < maxGenerations; ++gen) {
        std::vector<Individual> newPopulation;
        newPopulation.reserve(populationSize);

        // --- ELITISM ---
        // Always carry over the best individual to the next generation.
        // This ensures we never lose the best solution found (monotonic improvement).
        newPopulation.push_back(globalBest);

        // Fill the rest of the population
        while (newPopulation.size() < populationSize) {
            // 1. Selection
            Individual p1 = tournamentSelection();
            Individual p2 = tournamentSelection();

            Individual c1 = p1; // Placeholders
            Individual c2 = p2;

            // 2. Crossover
            crossoverSBX(p1, p2, c1, c2);

            // 3. Mutation
            mutatePolynomial(c1);
            mutatePolynomial(c2);

            // Add to new population
            newPopulation.push_back(c1);
            if (newPopulation.size() < populationSize) {
                newPopulation.push_back(c2);
            }
        }

        // Replace old population
        population = newPopulation;

        // Evaluate new population
        evaluatePopulation();

        // Update Global Best
        for (const auto& ind : population) {
            if (ind.fitness < globalBest.fitness) {
                globalBest = ind;
            }
        }

        // Optional: Print progress every 100 generations
        if (gen % 100 == 0) {
            // std::cout << "Gen " << gen << " Best: " << globalBest.fitness << std::endl;
        }
    }

    return globalBest;
}

Individual GeneticAlgorithm::runWithTracking(const std::string& outputFile) {
    initializePopulation();

    // Open file for writing convergence data
    std::ofstream outFile(outputFile);
    outFile << "Generation,BestFitness,WorstFitness,MeanFitness,StdDevFitness,FunctionEvaluations\n";

    int funcEvals = populationSize; // Initial population evaluation

    // Track the global best found so far
    Individual globalBest = population[0];
    for (const auto& ind : population) {
        if (ind.fitness < globalBest.fitness) {
            globalBest = ind;
        }
    }

    // Calculate initial population statistics
    double bestFit = globalBest.fitness;
    double worstFit = globalBest.fitness;
    double sumFit = 0.0;
    for (const auto& ind : population) {
        if (ind.fitness < bestFit) bestFit = ind.fitness;
        if (ind.fitness > worstFit) worstFit = ind.fitness;
        sumFit += ind.fitness;
    }
    double meanFit = sumFit / populationSize;
    double sqSum = 0.0;
    for (const auto& ind : population) {
        sqSum += (ind.fitness - meanFit) * (ind.fitness - meanFit);
    }
    double stdDev = std::sqrt(sqSum / populationSize);

    outFile << 0 << "," << bestFit << "," << worstFit << "," << meanFit << "," << stdDev << "," << funcEvals << "\n";

    for (int gen = 0; gen < maxGenerations; ++gen) {
        std::vector<Individual> newPopulation;
        newPopulation.reserve(populationSize);

        // --- ELITISM ---
        newPopulation.push_back(globalBest);

        // Fill the rest of the population
        while (newPopulation.size() < populationSize) {
            // 1. Selection
            Individual p1 = tournamentSelection();
            Individual p2 = tournamentSelection();

            Individual c1 = p1;
            Individual c2 = p2;

            // 2. Crossover
            crossoverSBX(p1, p2, c1, c2);

            // 3. Mutation
            mutatePolynomial(c1);
            mutatePolynomial(c2);

            // Add to new population
            newPopulation.push_back(c1);
            if (newPopulation.size() < populationSize) {
                newPopulation.push_back(c2);
            }
        }

        // Replace old population
        population = newPopulation;

        // Evaluate new population
        evaluatePopulation();
        funcEvals += populationSize;

        // Update Global Best
        for (const auto& ind : population) {
            if (ind.fitness < globalBest.fitness) {
                globalBest = ind;
            }
        }

        // Calculate statistics for this generation
        bestFit = globalBest.fitness;
        worstFit = population[0].fitness;
        sumFit = 0.0;
        for (const auto& ind : population) {
            if (ind.fitness > worstFit) worstFit = ind.fitness;
            sumFit += ind.fitness;
        }
        meanFit = sumFit / populationSize;
        sqSum = 0.0;
        for (const auto& ind : population) {
            sqSum += (ind.fitness - meanFit) * (ind.fitness - meanFit);
        }
        stdDev = std::sqrt(sqSum / populationSize);

        // Write to file
        outFile << (gen + 1) << "," << bestFit << "," << worstFit << "," << meanFit << "," << stdDev << "," << funcEvals << "\n";
    }

    outFile.close();
    return globalBest;
}
