#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <numeric>
#include "ObjectiveFunctions.h"
#include "GeneticAlgorithm.h"

// Structure to hold statistical results
struct ExperimentResult {
    double bestFitness;
    double worstFitness;
    double meanFitness;
    double stdDevFitness;
    double meanError; // Delta from optimum
};

ExperimentResult calculateStats(const std::vector<double>& results, double optimum) {
    ExperimentResult stats;
    if (results.empty()) return stats;

    stats.bestFitness = results[0];
    stats.worstFitness = results[0];
    double sum = 0.0;
    double sumError = 0.0;

    for (double val : results) {
        if (val < stats.bestFitness) stats.bestFitness = val;
        if (val > stats.worstFitness) stats.worstFitness = val;
        sum += val;
        sumError += std::abs(val - optimum);
    }

    stats.meanFitness = sum / results.size();
    stats.meanError = sumError / results.size();

    double sqSum = 0.0;
    for (double val : results) {
        sqSum += (val - stats.meanFitness) * (val - stats.meanFitness);
    }
    stats.stdDevFitness = std::sqrt(sqSum / results.size());

    return stats;
}

// Helper function to run a batch of experiments
void runBatchExperiment(std::string funcName, int dimension, std::function<double(const std::vector<double>&)> func) {
    // --- GA Configuration ---
    int numRuns = 30; // Statistical relevance requires multiple runs (usually 30+)
    
    // Population & Generations Tuning
    int popSize = (dimension == 30) ? 300 : 150; 
    int maxGen = (dimension == 30) ? 3000 : 1500; 

    // Special tuning for Schwefel 30D (Needs massive exploration)
    if (funcName == "Schwefel" && dimension == 30) {
        popSize = 1000;
        maxGen = 5000;
    }
    
    double mutationRate = 1.0 / dimension; 
    double crossoverRate = 0.9;

    double lowerBound, upperBound;
    MathFunctions::getBounds(funcName, lowerBound, upperBound);
    double optimum = MathFunctions::getGlobalOptimum(funcName, dimension);

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Running Batch: " << funcName << " (" << dimension << "D)" << std::endl;
    std::cout << "Runs: " << numRuns << " | Pop: " << popSize << " | Gen: " << maxGen << std::endl;

    std::vector<double> runResults;
    runResults.reserve(numRuns);

    for (int i = 0; i < numRuns; ++i) {
        GeneticAlgorithm ga(popSize, dimension, maxGen, mutationRate, crossoverRate, lowerBound, upperBound, func);
        Individual best = ga.run();
        runResults.push_back(best.fitness);
        
        // Optional: Progress bar
        if ((i + 1) % 5 == 0) std::cout << "." << std::flush;
    }
    std::cout << std::endl;

    // Calculate Statistics
    ExperimentResult stats = calculateStats(runResults, optimum);

    // Output results in a table-friendly format
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "RESULTS for " << funcName << " " << dimension << "D:" << std::endl;
    std::cout << "  Best Found: " << stats.bestFitness << std::endl;
    std::cout << "  Worst Found: " << stats.worstFitness << std::endl;
    std::cout << "  Mean Fitness: " << stats.meanFitness << std::endl;
    std::cout << "  Std Deviation: " << stats.stdDevFitness << std::endl;
    std::cout << "  Mean Delta (Error): " << stats.meanError << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
}

int main() {
    std::cout << "=== Genetic Algorithm Statistical Analysis (H2) ===" << std::endl;
    std::cout << "Running 30 independent trials per configuration." << std::endl;
    std::cout << std::endl;

    std::vector<int> dimensions = {5, 10, 30};

    // 1. De Jong 1
    for (int dim : dimensions) runBatchExperiment("DeJong1", dim, MathFunctions::DeJong1);

    // 2. Schwefel
    for (int dim : dimensions) runBatchExperiment("Schwefel", dim, MathFunctions::Schwefel);

    // 3. Rastrigin
    for (int dim : dimensions) runBatchExperiment("Rastrigin", dim, MathFunctions::Rastrigin);

    // 4. Michalewicz
    for (int dim : dimensions) runBatchExperiment("Michalewicz", dim, MathFunctions::Michalewicz);

    return 0;
}
