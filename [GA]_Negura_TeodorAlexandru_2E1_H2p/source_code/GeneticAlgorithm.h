#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <vector>
#include <functional>
#include <random>
#include <iostream>

// Represents a single candidate solution in the population.
// Using a struct for simple data holding (POD-like).
struct Individual {
    std::vector<double> genes; // The real-valued coordinates (5, 10, or 30 dimensions)
    double fitness;            // The objective function value (lower is better)
};

// The main engine class encapsulating the Genetic Algorithm logic.
// Follows OOP principles: Encapsulation of state (population, parameters) and behavior (operators).
class GeneticAlgorithm {
private:
    // --- Algorithm Parameters ---
    int populationSize;
    int dimension;
    int maxGenerations;
    double mutationRate;
    double crossoverRate;
    double lowerBound;
    double upperBound;
    double eta_c; // Distribution index for crossover
    double eta_m; // Distribution index for mutation

    // --- State ---
    std::vector<Individual> population;
    
    // Random Number Generator
    // Kept as a member to avoid re-seeding and ensure good statistical properties.
    std::mt19937 rng;

    // Function Pointer for the objective function to minimize
    std::function<double(const std::vector<double>&)> objectiveFunction;

    // --- Private Helper Methods (The "Gears" of the GA) ---

    // Initializes the population with random values within bounds.
    void initializePopulation();

    // Evaluates fitness for the entire population.
    void evaluatePopulation();

    // Tournament Selection: Picks a small group and selects the best.
    // Better than Roulette Wheel for maintaining diversity and handling negative fitness.
    Individual tournamentSelection(int k = 3);

    // Simulated Binary Crossover (SBX): Blends two parents to create two children.
    // Designed specifically for real-valued representations.
    void crossoverSBX(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2);

    // Polynomial Mutation: Adds small, controlled noise to genes.
    // Crucial for escaping local minima (unlike Hill Climbing).
    void mutatePolynomial(Individual& ind);

    // Helper to clamp values to bounds
    double clamp(double value);

public:
    // Constructor: Sets up the GA with specific problem parameters.
    GeneticAlgorithm(int popSize, int dim, int maxGen, double mutRate, double crossRate, 
                     double low, double high, 
                     std::function<double(const std::vector<double>&)> func,
                     double etaC = 20.0, double etaM = 20.0);

    // The main execution method.
    // Returns the best Individual found after the run.
    Individual run();

    // Run with convergence tracking for visualization
    // Outputs generation-by-generation statistics to a CSV file
    Individual runWithTracking(const std::string& outputFile);
};

#endif // GENETIC_ALGORITHM_H
