#include <iostream>
#include <string>
#include "ObjectiveFunctions.h"
#include "GeneticAlgorithm.h"

// This program generates convergence tracking data for visualization
// Runs representative examples from each function for Graph 1 & 2

int main() {
    std::cout << "Generating convergence tracking data..." << std::endl;

    // convergence_data directory should already exist

    // Rastrigin 30D - Most interesting convergence (multimodal)
    {
        std::cout << "Running Rastrigin 30D..." << std::endl;
        int dimension = 30;
        int popSize = 300;
        int maxGen = 3000;
        double mutationRate = 1.0 / dimension;
        double crossoverRate = 0.9;
        double lowerBound, upperBound;
        MathFunctions::getBounds("Rastrigin", lowerBound, upperBound);

        GeneticAlgorithm ga(popSize, dimension, maxGen, mutationRate, crossoverRate,
                           lowerBound, upperBound, MathFunctions::Rastrigin);
        ga.runWithTracking("convergence_data/rastrigin_30d.csv");
    }

    // Schwefel 30D - Deceptive function
    {
        std::cout << "Running Schwefel 30D..." << std::endl;
        int dimension = 30;
        int popSize = 1000;
        int maxGen = 5000;
        double mutationRate = 1.0 / dimension;
        double crossoverRate = 0.9;
        double lowerBound, upperBound;
        MathFunctions::getBounds("Schwefel", lowerBound, upperBound);

        GeneticAlgorithm ga(popSize, dimension, maxGen, mutationRate, crossoverRate,
                           lowerBound, upperBound, MathFunctions::Schwefel);
        ga.runWithTracking("convergence_data/schwefel_30d.csv");
    }

    // DeJong1 30D - Unimodal (shows smooth convergence)
    {
        std::cout << "Running DeJong1 30D..." << std::endl;
        int dimension = 30;
        int popSize = 300;
        int maxGen = 3000;
        double mutationRate = 1.0 / dimension;
        double crossoverRate = 0.9;
        double lowerBound, upperBound;
        MathFunctions::getBounds("DeJong1", lowerBound, upperBound);

        GeneticAlgorithm ga(popSize, dimension, maxGen, mutationRate, crossoverRate,
                           lowerBound, upperBound, MathFunctions::DeJong1);
        ga.runWithTracking("convergence_data/dejong1_30d.csv");
    }

    // Michalewicz 30D - Steep valleys
    {
        std::cout << "Running Michalewicz 30D..." << std::endl;
        int dimension = 30;
        int popSize = 300;
        int maxGen = 3000;
        double mutationRate = 1.0 / dimension;
        double crossoverRate = 0.9;
        double lowerBound, upperBound;
        MathFunctions::getBounds("Michalewicz", lowerBound, upperBound);

        GeneticAlgorithm ga(popSize, dimension, maxGen, mutationRate, crossoverRate,
                           lowerBound, upperBound, MathFunctions::Michalewicz);
        ga.runWithTracking("convergence_data/michalewicz_30d.csv");
    }

    std::cout << "Tracking data generated in convergence_data/ folder!" << std::endl;
    return 0;
}
