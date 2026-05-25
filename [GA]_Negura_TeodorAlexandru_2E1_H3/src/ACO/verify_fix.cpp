#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>
#include <algorithm>
#include "../TSPInstance.h"
#include "AntColonyOptimized.h"

// Simplified verification script based on aco_experiment_final.cpp

struct InstanceConfig {
    std::string name;
    std::string filename;
    int optimalTourLength;
    int numCities;
};

struct RunResult {
    double distance{0.0};
    double gap{0.0};
    long long durationMs{0};
    int runNumber{0};
    int iterationsUsed{0};
};

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  VERIFICATION RUN - Checking pr1002 fix                        " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::string dataPath = "C:/Users/TEODO/Desktop/Facultate/Cod/Alg Genetici/[GA]_Negura_TeodorAlexandru_2E1_H3/data/tsplib/pr1002.tsp";
    InstanceConfig config = {"pr1002", dataPath, 259045, 1002};
    
    // Load instance
    TSPInstance tsp;
    if (!tsp.loadFromFile(config.filename)) {
        std::cerr << "Failed to load: " << config.filename << std::endl;
        return 1;
    }
    tsp.setOptimalTourLength(config.optimalTourLength);

    std::cout << "Instance loaded: " << tsp.dimension << " cities." << std::endl;

    // Create ACO 
    AntColonyOptimized aco(tsp);
    
    // Use fewer iterations for a quick check, but enough to ensure it runs
    // The bug was appearing when no solution was found or construction failed
    aco.setMaxIterations(50); 
    aco.setForceFullIterations(false);

    std::cout << "Starting 5 test runs..." << std::endl;

    for (int i = 1; i <= 5; i++) {
        std::cout << "Run " << i << "... ";
        
        // Unique seed
        auto seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ (i * 12345));
        aco.reseed(seed);

        auto start = std::chrono::high_resolution_clock::now();
        Ant best = aco.run(false); // verbose=false
        auto end = std::chrono::high_resolution_clock::now();
        
        double gap = tsp.getGapFromOptimal(best.distance);
        
        if (best.distance < 1.0) {
            std::cout << "FAILED! Distance is " << best.distance << " (Gap: " << gap << "%)" << std::endl;
        } else {
            std::cout << "SUCCESS. Dist: " << (int)best.distance << " Gap: " << std::fixed << std::setprecision(2) << gap << "%" << std::endl;
        }
    }

    return 0;
}
