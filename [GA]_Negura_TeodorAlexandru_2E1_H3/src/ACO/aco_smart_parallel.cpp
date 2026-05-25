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
#include "../TSPInstance.h"
#include "AntColonySmart.h"

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
RunResult runExperiment(const InstanceConfig& config, int runNumber = 1, bool verbose = true) {
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
    Ant best = aco.run(false); // Run with minimal output during execution

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    double gap = tsp.getGapFromOptimal(best.distance);
    if (verbose) {
        std::cout << "Run " << runNumber << " gap: " << std::fixed << std::setprecision(4) << gap << "%" << std::endl;
    }

    RunResult res;
    res.distance = best.distance;
    res.gap = gap;
    res.durationMs = duration.count();
    res.runNumber = runNumber;
    res.dimension = tsp.dimension;
    res.stats = aco.getStatsHistory();
    return res;
}

// Run multiple times and aggregate stats (mean, stddev, min, max) - this is the key function for stats
void runExperimentMulti(const InstanceConfig& config, int runs, int runNumberForSummary = 1) {
    std::vector<std::future<RunResult>> futures;
    futures.reserve(runs);

    // Determine instance size from config name for thread optimization
    int instance_size = 0;
    if (config.name.find("berlin") != std::string::npos) instance_size = 52;
    else if (config.name.find("kroA100") != std::string::npos) instance_size = 100;
    else if (config.name.find("ch150") != std::string::npos) instance_size = 150;
    else if (config.name.find("kroA200") != std::string::npos) instance_size = 200;
    else if (config.name.find("a280") != std::string::npos) instance_size = 280;
    else if (config.name.find("pcb442") != std::string::npos) instance_size = 442;
    else if (config.name.find("pr1002") != std::string::npos) instance_size = 1002;
    else if (config.name.find("dka1376") != std::string::npos) instance_size = 1376;
    else if (config.name.find("djc1785") != std::string::npos) instance_size = 1785;
    else if (config.name.find("pia3056") != std::string::npos) instance_size = 3056;

    // Use up to 8 threads for parallel execution as requested
    int num_threads = std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
    std::cout << "Running " << runs << " experiments for " << config.name << " using " << num_threads << " parallel threads..." << std::endl;

    for (int r = 1; r <= runs; r++) {
        futures.push_back(std::async(std::launch::async, [config, r]() {
            return runExperiment(config, r, false); // Silent per-run, only gap to console
        }));
    }

    std::vector<double> distances;
    distances.reserve(runs);
    std::vector<long long> times;
    times.reserve(runs);

    RunResult bestRun;
    bestRun.distance = std::numeric_limits<double>::max();

    // Collect results - only show gap percentage as requested
    for (int r = 0; r < runs; r++) {
        RunResult res = futures[r].get(); // Wait for each future to complete
        distances.push_back(res.distance);
        times.push_back(res.durationMs);
        if (res.distance < bestRun.distance) {
            bestRun = res;
        }
        // Only output gap percentage as requested
        std::cout << "Run " << (r + 1) << " gap: " << std::fixed << std::setprecision(4) << res.gap << "%" << std::endl;
    }

    // Calculate comprehensive statistics - this is what we need for best stats
    double sum = 0.0, sumSq = 0.0;
    long long timeSum = 0, timeSumSq = 0;
    double bestGap = std::numeric_limits<double>::max();
    double worstGap = -std::numeric_limits<double>::max();
    double minDist = std::numeric_limits<double>::max();
    double maxDist = 0.0;

    for (int i = 0; i < runs; i++) {
        double d = distances[i];
        sum += d;
        sumSq += d * d;
        long long t = times[i];
        timeSum += t;
        timeSumSq += t * t;
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        double gap = ((d - config.optimalTourLength) / config.optimalTourLength) * 100.0;
        bestGap = std::min(bestGap, gap);
        worstGap = std::max(worstGap, gap);
    }

    double mean = sum / runs;
    double var = std::max(0.0, (sumSq / runs) - mean * mean);
    double stddev = std::sqrt(var);
    double meanTime = static_cast<double>(timeSum) / runs;
    double timeStdDev = std::sqrt(std::max(0.0, (static_cast<double>(timeSumSq) / runs) - meanTime * meanTime));

    // Save comprehensive results in the same format as other result files
    std::string summaryFile = "results/ACO_smart_" + config.name + "_run" + std::to_string(runNumberForSummary) + ".txt";
    std::ofstream out(summaryFile);
    if (!out.is_open()) {
        std::cerr << "Failed to write summary: " << summaryFile << std::endl;
        return;
    }

    // Single-line summary in the same format as other files (space-separated)
    // Fields: instance runs mean stddev min max best_gap worst_gap mean_time_ms best_run_num best_dist best_gap_real best_time_ms
    out << config.name << ' '
        << runs << ' '
        << mean << ' '
        << stddev << ' '
        << minDist << ' '
        << maxDist << ' '
        << bestGap << ' '
        << worstGap << ' '
        << meanTime << ' '
        << bestRun.runNumber << ' '
        << bestRun.distance << ' '
        << bestRun.gap << ' '
        << bestRun.durationMs << std::endl;
    
    out.close();

    std::cout << "Completed " << config.name << " - Best gap: " << std::fixed << std::setprecision(4) 
              << bestGap << "%, Mean gap: " << (mean - config.optimalTourLength) / config.optimalTourLength * 100.0 
              << "%, Worst gap: " << worstGap << "%" << std::endl;
    std::cout << "Summary saved to: " << summaryFile << std::endl;
}

int main() {
    // Last 4 instances as requested
    std::vector<InstanceConfig> instances = {
        {"pr1002",    "data/tsplib/pr1002.tsp",    259045},
        {"dka1376",   "data/tsplib/dka1376.tsp",   4666},
        {"djc1785",   "data/tsplib/djc1785.tsp",   6115},
        {"pia3056",   "data/tsplib/pia3056.tsp",   8258}
    };

    // Run each instance with 30 runs as requested
    int runs = 30;

    std::cout << "Starting SMART ACO experiments for last 4 instances with 30 runs each..." << std::endl;
    std::cout << "Running with 8 parallel threads per instance." << std::endl;
    std::cout << "Console will show only gap percentages as requested." << std::endl << std::endl;

    for (const auto& inst : instances) {
        std::cout << "Starting " << inst.name << " (" << runs << " runs)..." << std::endl;
        runExperimentMulti(inst, runs, 1);
        std::cout << std::endl;
    }

    std::cout << "\nAll SMART ACO experiments completed!" << std::endl;
    std::cout << "Results saved in results/ folder in the same format as other files." << std::endl;

    return 0;
}