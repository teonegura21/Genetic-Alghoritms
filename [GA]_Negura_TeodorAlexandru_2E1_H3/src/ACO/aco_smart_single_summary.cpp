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
#include "AntColonySmart.h"  // Use the smart version

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
    std::vector<AntColonySmart::IterationStats> stats;
    int dimension{0};
};

// Run smart ACO on a single instance and save results to file
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

    // Create smart ACO with exact required parameters: 200 ants, 2000 iterations
    AntColonySmart aco(tsp);
    
    // Fix the random seed issue: reseed with unique value based on run number and high-res time
    auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        (runNumber << 16) ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );
    aco.reseed(seed);

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run smart ACO (exactly 2000 iterations, 200 ants as required)
    Ant best = aco.run(verbose);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::string outputFile = "results/ACO_smart_" + config.name + "_single_run" + std::to_string(runNumber) + ".txt";
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Failed to create output file: " << outputFile << std::endl;
        return RunResult{};
    }

    out << "============================================" << std::endl;
    out << "SMART ANT COLONY OPTIMIZATION RESULTS (SINGLE RUN)" << std::endl;
    out << "============================================" << std::endl;
    out << "Instance: " << config.name << std::endl;
    out << "Cities: " << tsp.dimension << std::endl;
    out << "Optimal Tour Length: " << config.optimalTourLength << std::endl;
    out << "Run Number: " << runNumber << std::endl;
    out << "============================================" << std::endl;
    out << std::endl;

    out << "--- SMART ACO Parameters (Fixed) ---" << std::endl;
    out << "Number of Ants: " << aco.getNumAnts() << " (Fixed)" << std::endl;
    out << "Max Iterations: 2000 (Fixed)" << std::endl;
    out << "Alpha (Pheromone Importance): 1.0" << std::endl;
    out << "Beta (Heuristic Importance):   2.5" << std::endl;
    out << "Rho (Evaporation Rate):        0.02" << std::endl;
    out << "Local Search:                  2-opt improvement + Smart pheromone updates" << std::endl;
    out << "Search Strategy:               Hill-climber approach with SA escape" << std::endl;
    out << "Elitist Strategy:              Best-so-far + elite ants (5)" << std::endl;
    out << std::endl;

    // Get the stats history for detailed analysis
    const auto& statsHistory = aco.getStatsHistory();
    
    out << "--- ITERATION SUMMARY INTERVALS ---" << std::endl;
    
    if (statsHistory.size() >= 1) {
        // Show stats at key intervals
        std::vector<int> intervals = {0, 499, 999, 1499, static_cast<int>(statsHistory.size())-1};
        
        for (int interval : intervals) {
            if (interval < static_cast<int>(statsHistory.size())) {
                auto& stat = statsHistory[interval];
                double gap = tsp.getGapFromOptimal(stat.bestDistance);
                out << "Iteration " << stat.iteration << ": Best=" << std::fixed << std::setprecision(2) 
                    << stat.bestDistance << ", Gap=" << std::setprecision(4) << gap << "%" << std::endl;
            }
        }
    }
    
    out << std::endl;

    out << "--- FINAL RESULTS ---" << std::endl;
    out << "Best Tour Distance: " << std::fixed << std::setprecision(2) << best.distance << std::endl;
    out << "Optimal Distance: " << config.optimalTourLength << std::endl;
    out << "Gap from Optimal: " << std::fixed << std::setprecision(4)
        << tsp.getGapFromOptimal(best.distance) << "%" << std::endl;
    out << "Execution Time:          " << duration.count() << " ms (" << (duration.count() / 1000.0) << " s)" << std::endl;
    out << "Final Iteration:           2000 (All completed)" << std::endl;

    if (!statsHistory.empty()) {
        double initialBest = statsHistory.front().bestDistance;
        double finalBest = statsHistory.back().bestDistance;
        double improvement = ((initialBest - finalBest) / initialBest) * 100.0;
        out << "Initial Best Distance:   " << std::fixed << std::setprecision(2) << initialBest << std::endl;
        out << "Improvement Achieved:    " << std::fixed << std::setprecision(2) << improvement << "%" << std::endl;
    }
    
    // Calculate and display improvement in each segment
    if (statsHistory.size() >= 500) {
        out << std::endl;
        out << "--- IMPROVEMENT ANALYSIS BY SEGMENT ---" << std::endl;
        
        // Segment 1: 0-500
        if (statsHistory.size() > 500) {
            double startSeg1 = statsHistory[0].bestDistance;
            double endSeg1 = statsHistory[499].bestDistance;
            double improvementSeg1 = ((startSeg1 - endSeg1) / startSeg1) * 100.0;
            out << "Segment 0-500: Improvement = " << std::fixed << std::setprecision(2) << improvementSeg1 << "%" << std::endl;
        }
        
        // Segment 2: 500-1000
        if (statsHistory.size() > 1000) {
            double startSeg2 = statsHistory[499].bestDistance;
            double endSeg2 = statsHistory[999].bestDistance;
            double improvementSeg2 = ((startSeg2 - endSeg2) / startSeg2) * 100.0;
            out << "Segment 500-1000: Improvement = " << std::fixed << std::setprecision(2) << improvementSeg2 << "%" << std::endl;
        }
        
        // Segment 3: 1000-1500
        if (statsHistory.size() > 1500) {
            double startSeg3 = statsHistory[999].bestDistance;
            double endSeg3 = statsHistory[1499].bestDistance;
            double improvementSeg3 = ((startSeg3 - endSeg3) / startSeg3) * 100.0;
            out << "Segment 1000-1500: Improvement = " << std::fixed << std::setprecision(2) << improvementSeg3 << "%" << std::endl;
        }
        
        // Segment 4: 1500-2000
        if (statsHistory.size() > 1500) {
            double startSeg4 = statsHistory[1499].bestDistance;
            double endSeg4 = statsHistory.back().bestDistance;
            double improvementSeg4 = ((startSeg4 - endSeg4) / startSeg4) * 100.0;
            out << "Segment 1500-2000: Improvement = " << std::fixed << std::setprecision(2) << improvementSeg4 << "%" << std::endl;
        }
    }
    
    out << std::endl;
    out << "============================================" << std::endl;
    out << "RUN COMPLETED SUCCESSFULLY" << std::endl;
    out << "============================================" << std::endl;

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

    std::cout << "Starting single run of SMART ACO on pr1002..." << std::endl;
    RunResult result = runExperimentSingle(config, 1, true);

    std::cout << "\nSingle run completed!" << std::endl;
    std::cout << "Result: " << result.distance << " (Gap: " << result.gap << "%)" << std::endl;
    std::cout << "Time: " << (result.durationMs / 1000.0) << " seconds" << std::endl;

    return 0;
}