#ifndef META_GA_H
#define META_GA_H

#include <vector>
#include <string>
#include <functional>
#include "GeneticAlgorithm.h"
#include "ObjectiveFunctions.h"

// Structure to hold a hyperparameter configuration
struct GAConfig {
    int populationSize;      // Range: [50, 1000]
    double mutationRate;     // Range: [0.001, 0.5]
    double crossoverRate;    // Range: [0.5, 1.0]
    double eta_c;            // Range: [5.0, 50.0]
    double eta_m;            // Range: [5.0, 50.0]
    
    double fitness;          // Average performance across trials
};

// Meta-GA class that optimizes hyperparameters for the base GA
class MetaGA {
private:
    // Target problem configuration
    std::string functionName;
    int problemDimension;
    std::function<double(const std::vector<double>&)> objectiveFunc;
    
    // Meta-GA parameters
    int metaPopSize;
    int metaGenerations;
    int numTrialsPerConfig;  // How many times to run base GA per config
    int baseGAGenerations;   // Max generations for base GA during evaluation
    
    // Population of configurations
    std::vector<GAConfig> metaPopulation;
    
    // Random number generator
    std::mt19937 rng;
    
    // Helper methods
    void initializeMetaPopulation();
    double evaluateConfig(const GAConfig& config);
    GAConfig tournamentSelection(int k = 3);
    void crossoverConfigs(const GAConfig& p1, const GAConfig& p2, GAConfig& c1, GAConfig& c2);
    void mutateConfig(GAConfig& config);
    double clamp(double value, double minVal, double maxVal);
    int clampInt(int value, int minVal, int maxVal);
    
public:
    // Constructor
    MetaGA(const std::string& funcName, int dim,
           std::function<double(const std::vector<double>&)> func,
           int metaPop = 20, int metaGen = 15, 
           int trialsPerConfig = 5, int baseGen = 1000);
    
    // Run meta-optimization and return best config
    GAConfig optimize();
    
    // Save configuration to file
    void saveConfig(const GAConfig& config, const std::string& filename);
};

#endif // META_GA_H
