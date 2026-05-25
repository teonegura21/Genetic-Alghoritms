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

// Instance configuration
struct InstanceConfig {
    std::string name;
    std::string filename;
    int optimalTourLength;
    int numCities;  // For thread optimization
};

struct RunResult {
    double distance{0.0};
    double gap{0.0};
    long long durationMs{0};
    int runNumber{0};
    int iterationsUsed{0};
};

// Mutex for thread-safe console output
std::mutex coutMutex;

// Determine optimal thread count based on instance size and hardware
// i7-13650HX: 14 cores (6P + 8E), 20 threads, 24GB RAM
int getOptimalThreadCount(int numCities) {
    // Memory per run estimation: ~4 matrices of n² doubles + overhead
    // n=3056: ~350MB per run, n=1785: ~120MB, n=1376: ~70MB, n=1002: ~40MB

    if (numCities > 2500) {
        return 4;   // ~1.4GB total, safe for 24GB
    } else if (numCities > 1500) {
        return 6;   // ~720MB total
    } else if (numCities > 1000) {
        return 8;   // ~560MB total
    } else {
        return 12;  // Use more threads for smaller instances
    }
}

// Run single ACO experiment (minimal console output)
RunResult runSingleExperiment(const InstanceConfig& config, int runNumber) {
    // Load instance
    TSPInstance tsp;
    if (!tsp.loadFromFile(config.filename)) {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cerr << "Failed to load: " << config.filename << std::endl;
        return RunResult{};
    }
    tsp.setOptimalTourLength(config.optimalTourLength);

    // Create ACO with fixed constraints: 200 ants, 2000 iterations
    AntColonyOptimized aco(tsp);
    aco.setForceFullIterations(true);  // Run all 2000 iterations

    // Unique seed per run
    auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        (runNumber * 31337) ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );
    aco.reseed(seed);

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run ACO (verbose=false for minimal output)
    Ant best = aco.run(false);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Build result
    RunResult res;
    res.distance = best.distance;
    res.gap = tsp.getGapFromOptimal(best.distance);
    res.durationMs = duration.count();
    res.runNumber = runNumber;
    res.iterationsUsed = aco.getStatsHistory().size();

    // Print minimal console output
    {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cout << "  Run " << std::setw(2) << runNumber << ": "
                  << std::fixed << std::setprecision(2) << res.gap << "%" << std::endl;
    }

    return res;
}

// Run all experiments for an instance and write comprehensive results
void runInstanceExperiments(const InstanceConfig& config, int totalRuns = 30) {
    int maxThreads = getOptimalThreadCount(config.numCities);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Instance: " << config.name << " (" << config.numCities << " cities)" << std::endl;
    std::cout << "Runs: " << totalRuns << " | Parallel threads: " << maxThreads << std::endl;
    std::cout << "========================================" << std::endl;

    // Run experiments with limited parallelism
    std::vector<RunResult> results;
    results.reserve(totalRuns);

    // Process in batches to limit memory usage
    int completed = 0;
    while (completed < totalRuns) {
        int batchSize = std::min(maxThreads, totalRuns - completed);
        std::vector<std::future<RunResult>> futures;
        futures.reserve(batchSize);

        // Launch batch
        for (int i = 0; i < batchSize; i++) {
            int runNum = completed + i + 1;
            futures.push_back(std::async(std::launch::async, [&config, runNum]() {
                return runSingleExperiment(config, runNum);
            }));
        }

        // Collect batch results
        for (auto& f : futures) {
            results.push_back(f.get());
        }

        completed += batchSize;
    }

    // Sort by run number for display
    std::sort(results.begin(), results.end(), [](const RunResult& a, const RunResult& b) {
        return a.runNumber < b.runNumber;
    });

    // Calculate statistics
    double sumDist = 0.0, sumDistSq = 0.0;
    double sumGap = 0.0, sumGapSq = 0.0;
    long long sumTime = 0;
    double bestGap = 1e9, worstGap = -1e9;
    double bestDist = 1e9, worstDist = 0;
    int bestRun = 0;

    for (const auto& r : results) {
        sumDist += r.distance;
        sumDistSq += r.distance * r.distance;
        sumGap += r.gap;
        sumGapSq += r.gap * r.gap;
        sumTime += r.durationMs;

        if (r.gap < bestGap) {
            bestGap = r.gap;
            bestDist = r.distance;
            bestRun = r.runNumber;
        }
        if (r.gap > worstGap) {
            worstGap = r.gap;
            worstDist = r.distance;
        }
    }

    double meanDist = sumDist / totalRuns;
    double meanGap = sumGap / totalRuns;
    double meanTime = static_cast<double>(sumTime) / totalRuns;
    double varDist = std::max(0.0, (sumDistSq / totalRuns) - meanDist * meanDist);
    double varGap = std::max(0.0, (sumGapSq / totalRuns) - meanGap * meanGap);
    double stdDist = std::sqrt(varDist);
    double stdGap = std::sqrt(varGap);

    // Print summary to console
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Summary: Best=" << std::fixed << std::setprecision(2) << bestGap
              << "% | Mean=" << meanGap << "% | Worst=" << worstGap << "%" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Write comprehensive results to file
    std::string outputFile = "results/ACO_" + config.name + "_30runs.txt";
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Failed to create output file: " << outputFile << std::endl;
        return;
    }

    out << "================================================================" << std::endl;
    out << "       ANT COLONY OPTIMIZATION - COMPREHENSIVE RESULTS          " << std::endl;
    out << "================================================================" << std::endl;
    out << std::endl;

    out << "INSTANCE INFORMATION" << std::endl;
    out << "--------------------" << std::endl;
    out << "Name:                  " << config.name << std::endl;
    out << "Optimal Tour Length:   " << config.optimalTourLength << std::endl;
    out << "Total Runs:            " << totalRuns << std::endl;
    out << std::endl;

    out << "ALGORITHM PARAMETERS (Fixed)" << std::endl;
    out << "----------------------------" << std::endl;
    out << "Number of Ants:        200" << std::endl;
    out << "Max Iterations:        2000" << std::endl;
    out << "Algorithm:             MAX-MIN Ant System (MMAS)" << std::endl;
    out << "Pheromone Bounds:      [tau_min, tau_max] dynamically computed" << std::endl;
    out << "Alpha (pheromone):     0.8 - 1.0 (adaptive)" << std::endl;
    out << "Beta (heuristic):      2.5 - 3.0 (adaptive)" << std::endl;
    out << "Rho (evaporation):     0.01 - 0.02 (adaptive)" << std::endl;
    out << "Local Search:          Lin-Kernighan style 2-opt" << std::endl;
    out << "Elite Ants:            5 - 8 (adaptive)" << std::endl;
    out << std::endl;

    out << "================================================================" << std::endl;
    out << "                    INDIVIDUAL RUN RESULTS                      " << std::endl;
    out << "================================================================" << std::endl;
    out << std::endl;

    out << std::setw(6) << "Run"
        << std::setw(15) << "Distance"
        << std::setw(12) << "Gap (%)"
        << std::setw(12) << "Time (s)"
        << std::setw(10) << "Iters" << std::endl;
    out << std::string(55, '-') << std::endl;

    for (const auto& r : results) {
        out << std::setw(6) << r.runNumber
            << std::setw(15) << std::fixed << std::setprecision(2) << r.distance
            << std::setw(12) << std::setprecision(4) << r.gap
            << std::setw(12) << std::setprecision(2) << (r.durationMs / 1000.0)
            << std::setw(10) << r.iterationsUsed << std::endl;
    }

    out << std::string(55, '-') << std::endl;
    out << std::endl;

    out << "================================================================" << std::endl;
    out << "                    AGGREGATE STATISTICS                        " << std::endl;
    out << "================================================================" << std::endl;
    out << std::endl;

    out << "DISTANCE STATISTICS" << std::endl;
    out << "-------------------" << std::endl;
    out << "Best Distance:         " << std::fixed << std::setprecision(2) << bestDist
        << " (Run " << bestRun << ")" << std::endl;
    out << "Worst Distance:        " << worstDist << std::endl;
    out << "Mean Distance:         " << meanDist << std::endl;
    out << "Std Dev Distance:      " << stdDist << std::endl;
    out << std::endl;

    out << "GAP FROM OPTIMAL (%)" << std::endl;
    out << "--------------------" << std::endl;
    out << "Best Gap:              " << std::setprecision(4) << bestGap << "%" << std::endl;
    out << "Worst Gap:             " << worstGap << "%" << std::endl;
    out << "Mean Gap:              " << meanGap << "%" << std::endl;
    out << "Std Dev Gap:           " << stdGap << "%" << std::endl;
    out << std::endl;

    out << "EXECUTION TIME" << std::endl;
    out << "--------------" << std::endl;
    out << "Mean Time:             " << std::setprecision(2) << (meanTime / 1000.0) << " seconds" << std::endl;
    out << "Total Time:            " << (sumTime / 1000.0) << " seconds" << std::endl;
    out << std::endl;

    out << "================================================================" << std::endl;
    out << "                         QUALITY ASSESSMENT                     " << std::endl;
    out << "================================================================" << std::endl;
    out << std::endl;

    std::string quality;
    if (meanGap < 2.0) quality = "EXCELLENT";
    else if (meanGap < 5.0) quality = "GOOD";
    else if (meanGap < 10.0) quality = "FAIR";
    else quality = "POOR";

    out << "Overall Quality:       " << quality << std::endl;
    out << "Consistency:           " << (stdGap < 1.0 ? "HIGH" : stdGap < 3.0 ? "MEDIUM" : "LOW") << std::endl;
    out << std::endl;

    out << "================================================================" << std::endl;
    out << "                         END OF REPORT                          " << std::endl;
    out << "================================================================" << std::endl;

    out.close();
    std::cout << "Results saved to: " << outputFile << std::endl;
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  ACO EXPERIMENT - MMAS with 200 ants, 2000 iterations          " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Last 4 instances as requested (name, filename, optimal, numCities)
    std::vector<InstanceConfig> instances = {
        {"pr1002",   "data/tsplib/pr1002.tsp",   259045, 1002},   // 8 threads
        {"dka1376",  "data/tsplib/dka1376.tsp",  4666,   1376},   // 8 threads
        {"djc1785",  "data/tsplib/djc1785.tsp",  6115,   1785},   // 6 threads
        {"pia3056",  "data/tsplib/pia3056.tsp",  8258,   3056},   // 4 threads
    };

    const int RUNS = 30;

    for (const auto& inst : instances) {
        runInstanceExperiments(inst, RUNS);
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  ALL EXPERIMENTS COMPLETED                                     " << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
