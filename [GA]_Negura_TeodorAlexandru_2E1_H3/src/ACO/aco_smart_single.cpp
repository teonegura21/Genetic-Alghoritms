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
#include "TSPInstance.h"
#include "AntColonyOptimized.h"  // Use the optimized MMAS version

// Instance configuration
struct InstanceConfig {
    std::string name;
    std::string filename;
    int optimalTourLength;
};

struct RunResult {
    double distance{0.0};
    double gap{0.0};
    long long durationMs{0};
    int runNumber{1};
    std::string outputFile;
    std::vector<AntColonyOptimized::IterationStats> stats;
    int dimension{0};
};

// Run optimized ACO (MMAS) on a single instance and save results to file
RunResult runExperimentSingle(const InstanceConfig& config, int runNumber = 1, bool verbose = true) {
    if (verbose) {
        std::cout << "Run " << runNumber << " (" << config.name << ")" << std::endl;
    }

    // Load instance
    TSPInstance tsp;
    if (!tsp.loadFromFile(config.filename)) {
        std::cerr << "Failed to load: " << config.filename << std::endl;
        return RunResult{};
    }
    tsp.setOptimalTourLength(config.optimalTourLength);

    // Create optimized ACO with exact required parameters: 200 ants, 2000 iterations
    AntColonyOptimized aco(tsp);
    
    // Fix the random seed issue: reseed with unique value based on run number and high-res time
    auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        (runNumber << 16) ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );
    aco.reseed(seed);

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run optimized ACO (exactly 2000 iterations, 200 ants as required)
    Ant best = aco.run(verbose);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::string outputFile = "results/ACO_opt_" + config.name + "_single_run" + std::to_string(runNumber) + ".txt";
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Failed to create output file: " << outputFile << std::endl;
        return RunResult{};
    }

    out << "============================================" << std::endl;
    out << "OPTIMIZED ANT COLONY OPTIMIZATION RESULTS (SINGLE RUN)" << std::endl;
    out << "============================================" << std::endl;
    out << "Instance: " << config.name << std::endl;
    out << "Cities: " << tsp.dimension << std::endl;
    out << "Optimal Tour Length: " << config.optimalTourLength << std::endl;
    out << "Run Number: " << runNumber << std::endl;
    out << "============================================" << std::endl;
    out << std::endl;

    out << "--- OPTIMIZED ACO Parameters (MMAS) ---" << std::endl;
    out << "Number of Ants: " << aco.getNumAnts() << " (Fixed)" << std::endl;
    out << "Max Iterations: 2000 (Fixed)" << std::endl;
    out << "Alpha (Pheromone Importance): adaptive (0.8-1.0)" << std::endl;
    out << "Beta (Heuristic Importance):   adaptive (2.5-3.0)" << std::endl;
    out << "Rho (Evaporation Rate):        adaptive (0.01-0.02)" << std::endl;
    out << "Local Search:                  Lin-Kernighan 2-opt improvement" << std::endl;
    out << "Search Strategy:               MAX-MIN Ant System with bounds" << std::endl;
    out << "Elitist Strategy:              Best-so-far + elite ants (5-8)" << std::endl;
    out << std::endl;

    out << "--- Final Results ---" << std::endl;
    out << "Best Tour Distance: " << std::fixed << std::setprecision(2) << best.distance << std::endl;
    out << "Optimal Distance: " << config.optimalTourLength << std::endl;
    out << "Gap from Optimal: " << std::fixed << std::setprecision(4)
        << tsp.getGapFromOptimal(best.distance) << "%" << std::endl;
    out << "Execution Time:          " << duration.count() << " ms (" << (duration.count() / 1000.0) << " s)" << std::endl;
    out << "Final Iteration:           2000 (All completed)" << std::endl;

    if (!aco.getStatsHistory().empty()) {
        double initialBest = aco.getStatsHistory().front().bestDistance;
        double finalBest = aco.getStatsHistory().back().bestDistance;
        double improvement = ((initialBest - finalBest) / initialBest) * 100.0;
        out << "Initial Best Distance:   " << std::fixed << std::setprecision(2) << initialBest << std::endl;
        out << "Improvement Achieved:    " << std::fixed << std::setprecision(2) << improvement << "%" << std::endl;
    }
    out << std::endl;

    RunResult res;
    res.distance = best.distance;
    res.gap = tsp.getGapFromOptimal(best.distance);
    res.durationMs = duration.count();
    res.runNumber = runNumber;
    res.outputFile = outputFile;
    res.stats = aco.getStatsHistory();
    res.dimension = tsp.dimension;
    if (verbose) {
        std::cout << "Run " << runNumber << ": gap " << res.gap << "%" << std::endl;
    }

    out.close();
    return res;
}

int main() {
    // Just pr1002 instance
    InstanceConfig config = {"pr1002", "data/tsplib/pr1002.tsp", 259045};

    std::cout << "Starting single run of OPTIMIZED ACO (MMAS) on pr1002..." << std::endl;
    RunResult result = runExperimentSingle(config, 1, true);

    std::cout << "\nSingle run completed!" << std::endl;
    std::cout << "Result: " << result.distance << " (Gap: " << result.gap << "%)" << std::endl;
    std::cout << "Time: " << (result.durationMs / 1000.0) << " seconds" << std::endl;

    return 0;
}