#include "MetaGA.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

MetaGA::MetaGA(const std::string& funcName, int dim,
               std::function<double(const std::vector<double>&)> func,
               int metaPop, int metaGen, int trialsPerConfig, int baseGen)
    : functionName(funcName), problemDimension(dim), objectiveFunc(func),
      metaPopSize(metaPop), metaGenerations(metaGen), 
      numTrialsPerConfig(trialsPerConfig), baseGAGenerations(baseGen) {
    
    std::random_device rd;
    rng.seed(rd());
}

double MetaGA::clamp(double value, double minVal, double maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

int MetaGA::clampInt(int value, int minVal, int maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

void MetaGA::initializeMetaPopulation() {
    metaPopulation.clear();
    metaPopulation.resize(metaPopSize);
    
    std::uniform_int_distribution<int> popDist(50, 1000);
    std::uniform_real_distribution<double> mutDist(0.001, 0.5);
    std::uniform_real_distribution<double> crossDist(0.5, 1.0);
    std::uniform_real_distribution<double> etaDist(5.0, 50.0);
    
    for (int i = 0; i < metaPopSize; ++i) {
        metaPopulation[i].populationSize = popDist(rng);
        metaPopulation[i].mutationRate = mutDist(rng);
        metaPopulation[i].crossoverRate = crossDist(rng);
        metaPopulation[i].eta_c = etaDist(rng);
        metaPopulation[i].eta_m = etaDist(rng);
        
        // Evaluate initial fitness
        metaPopulation[i].fitness = evaluateConfig(metaPopulation[i]);
    }
}

double MetaGA::evaluateConfig(const GAConfig& config) {
    // Run base GA multiple times with this config and average the results
    double totalFitness = 0.0;
    
    double lowerBound, upperBound;
    MathFunctions::getBounds(functionName, lowerBound, upperBound);
    
    for (int trial = 0; trial < numTrialsPerConfig; ++trial) {
        GeneticAlgorithm ga(
            config.populationSize,
            problemDimension,
            baseGAGenerations,
            config.mutationRate,
            config.crossoverRate,
            lowerBound,
            upperBound,
            objectiveFunc,
            config.eta_c,
            config.eta_m
        );
        
        Individual best = ga.run();
        totalFitness += best.fitness;
    }
    
    return totalFitness / numTrialsPerConfig;
}

GAConfig MetaGA::tournamentSelection(int k) {
    std::uniform_int_distribution<int> dist(0, metaPopSize - 1);
    
    int bestIdx = dist(rng);
    double bestFitness = metaPopulation[bestIdx].fitness;
    
    for (int i = 1; i < k; ++i) {
        int idx = dist(rng);
        if (metaPopulation[idx].fitness < bestFitness) { // Minimization
            bestFitness = metaPopulation[idx].fitness;
            bestIdx = idx;
        }
    }
    return metaPopulation[bestIdx];
}

void MetaGA::crossoverConfigs(const GAConfig& p1, const GAConfig& p2, 
                               GAConfig& c1, GAConfig& c2) {
    c1 = p1;
    c2 = p2;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Blend crossover for continuous parameters
    double alpha = 0.5;
    
    if (dist(rng) < 0.9) { // Crossover rate for meta-GA
        // Population size (integer)
        c1.populationSize = static_cast<int>(alpha * p1.populationSize + (1-alpha) * p2.populationSize);
        c2.populationSize = static_cast<int>((1-alpha) * p1.populationSize + alpha * p2.populationSize);
        
        // Mutation rate
        c1.mutationRate = alpha * p1.mutationRate + (1-alpha) * p2.mutationRate;
        c2.mutationRate = (1-alpha) * p1.mutationRate + alpha * p2.mutationRate;
        
        // Crossover rate
        c1.crossoverRate = alpha * p1.crossoverRate + (1-alpha) * p2.crossoverRate;
        c2.crossoverRate = (1-alpha) * p1.crossoverRate + alpha * p2.crossoverRate;
        
        // Eta_c
        c1.eta_c = alpha * p1.eta_c + (1-alpha) * p2.eta_c;
        c2.eta_c = (1-alpha) * p1.eta_c + alpha * p2.eta_c;
        
        // Eta_m
        c1.eta_m = alpha * p1.eta_m + (1-alpha) * p2.eta_m;
        c2.eta_m = (1-alpha) * p1.eta_m + alpha * p2.eta_m;
    }
    
    // Clamp to valid ranges
    c1.populationSize = clampInt(c1.populationSize, 50, 1000);
    c2.populationSize = clampInt(c2.populationSize, 50, 1000);
    c1.mutationRate = clamp(c1.mutationRate, 0.001, 0.5);
    c2.mutationRate = clamp(c2.mutationRate, 0.001, 0.5);
    c1.crossoverRate = clamp(c1.crossoverRate, 0.5, 1.0);
    c2.crossoverRate = clamp(c2.crossoverRate, 0.5, 1.0);
    c1.eta_c = clamp(c1.eta_c, 5.0, 50.0);
    c2.eta_c = clamp(c2.eta_c, 5.0, 50.0);
    c1.eta_m = clamp(c1.eta_m, 5.0, 50.0);
    c2.eta_m = clamp(c2.eta_m, 5.0, 50.0);
}

void MetaGA::mutateConfig(GAConfig& config) {
    std::uniform_real_distribution<double> probDist(0.0, 1.0);
    std::normal_distribution<double> noiseDist(0.0, 0.1); // Gaussian mutation
    
    double mutProb = 0.3; // Mutation probability per parameter
    
    if (probDist(rng) < mutProb) {
        int delta = static_cast<int>(noiseDist(rng) * 200);
        config.populationSize = clampInt(config.populationSize + delta, 50, 1000);
    }
    
    if (probDist(rng) < mutProb) {
        config.mutationRate = clamp(config.mutationRate * (1.0 + noiseDist(rng)), 0.001, 0.5);
    }
    
    if (probDist(rng) < mutProb) {
        config.crossoverRate = clamp(config.crossoverRate + noiseDist(rng) * 0.1, 0.5, 1.0);
    }
    
    if (probDist(rng) < mutProb) {
        config.eta_c = clamp(config.eta_c + noiseDist(rng) * 10.0, 5.0, 50.0);
    }
    
    if (probDist(rng) < mutProb) {
        config.eta_m = clamp(config.eta_m + noiseDist(rng) * 10.0, 5.0, 50.0);
    }
}

GAConfig MetaGA::optimize() {
    std::cout << "\n=== Meta-GA Optimization: " << functionName 
              << " (" << problemDimension << "D) ===" << std::endl;
    
    initializeMetaPopulation();
    
    // Find initial best
    GAConfig globalBest = metaPopulation[0];
    for (const auto& config : metaPopulation) {
        if (config.fitness < globalBest.fitness) {
            globalBest = config;
        }
    }
    
    std::cout << "Initial Best Fitness: " << globalBest.fitness << std::endl;
    
    for (int gen = 0; gen < metaGenerations; ++gen) {
        std::vector<GAConfig> newPopulation;
        newPopulation.reserve(metaPopSize);
        
        // Elitism
        newPopulation.push_back(globalBest);
        
        while (newPopulation.size() < static_cast<size_t>(metaPopSize)) {
            GAConfig p1 = tournamentSelection();
            GAConfig p2 = tournamentSelection();
            
            GAConfig c1, c2;
            crossoverConfigs(p1, p2, c1, c2);
            
            mutateConfig(c1);
            mutateConfig(c2);
            
            // Evaluate offspring
            c1.fitness = evaluateConfig(c1);
            c2.fitness = evaluateConfig(c2);
            
            newPopulation.push_back(c1);
            if (newPopulation.size() < static_cast<size_t>(metaPopSize)) {
                newPopulation.push_back(c2);
            }
        }
        
        metaPopulation = newPopulation;
        
        // Update global best
        for (const auto& config : metaPopulation) {
            if (config.fitness < globalBest.fitness) {
                globalBest = config;
            }
        }
        
        std::cout << "Generation " << (gen + 1) << "/" << metaGenerations 
                  << " - Best Fitness: " << globalBest.fitness << std::endl;
    }
    
    std::cout << "\nOptimization Complete!" << std::endl;
    std::cout << "Best Configuration Found:" << std::endl;
    std::cout << "  Population Size: " << globalBest.populationSize << std::endl;
    std::cout << "  Mutation Rate: " << globalBest.mutationRate << std::endl;
    std::cout << "  Crossover Rate: " << globalBest.crossoverRate << std::endl;
    std::cout << "  Eta_c: " << globalBest.eta_c << std::endl;
    std::cout << "  Eta_m: " << globalBest.eta_m << std::endl;
    std::cout << "  Average Fitness: " << globalBest.fitness << std::endl;
    
    return globalBest;
}

void MetaGA::saveConfig(const GAConfig& config, const std::string& filename) {
    std::ofstream outFile(filename);
    outFile << std::fixed << std::setprecision(6);
    outFile << "Function: " << functionName << std::endl;
    outFile << "Dimension: " << problemDimension << std::endl;
    outFile << "Population Size: " << config.populationSize << std::endl;
    outFile << "Mutation Rate: " << config.mutationRate << std::endl;
    outFile << "Crossover Rate: " << config.crossoverRate << std::endl;
    outFile << "Eta_c: " << config.eta_c << std::endl;
    outFile << "Eta_m: " << config.eta_m << std::endl;
    outFile << "Average Fitness: " << config.fitness << std::endl;
    outFile.close();
}
