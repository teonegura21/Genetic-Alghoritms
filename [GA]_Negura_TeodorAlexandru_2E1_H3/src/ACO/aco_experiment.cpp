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
#include "AntColony.h"

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
    std::vector<AntColony::IterationStats> stats;
    int dimension{0};
};

// Run ACO on a single instance and save results to file
RunResult runExperiment(const InstanceConfig& config, int runNumber = 1, bool verbose = true, bool writeTour = true, bool writeFile = true) {
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

    // Determine instance size from name for parameter adjustment
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

    // Create ACO with tuned defaults
    AntColony aco(tsp);
    int numAnts = 200;
    aco.setNumAnts(numAnts);

    // Adjust parameters based on instance size to use intensive settings for small instances
    // and optimized settings for large instances
    if (instance_size >= 1000) {
        // For large instances (≥1000 nodes), use optimized parameters for reasonable runtime
        aco.setMaxIterations(1500); // Reduced from 2000 for time efficiency
        aco.setLocalSearchProbability(0.15);  // Reduced from 0.25 to save time
        aco.setLocalSearchStartIter(100);     // Start LS at iteration 100
        aco.setImmigrantInterval(12);         // Moderate diversity injection
        aco.setNoImprovementLimit(200);       // Earlier termination for large instances
    } else {
        // For smaller instances (<1000 nodes), use original intensive parameters
        aco.setMaxIterations(2000); // Full iterations for best quality
        aco.setLocalSearchProbability(0.25); // Full local search probability
        aco.setNoImprovementLimit(300);      // Longer before early termination
        aco.setLocalSearchStartIter(200);    // Later start as per original
    }

    aco.setAlpha(1.0);
    aco.setBeta(2.0);
    aco.setRho(0.1);
    aco.setQ0(0.8);
    aco.setEliteAnts(3);
    // Optional extra tuning for runs from CLI
    aco.setImmigrantRate(0.10);
    aco.setImmigrantInterval(15);
    aco.setRestartThreshold(40);

    // Enable early termination for better performance
    aco.setEnableEarlyTermination(true);
    aco.setTargetGap(1.5);          // Stop if within 1.5% of known optimal

    // Fix the random seed issue: reseed with unique value based on run number and high-res time
    auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        (runNumber << 16) ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );
    aco.reseed(seed);

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run ACO
    Ant best = aco.run(verbose);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::string outputFile;
    std::ofstream out;
    if (writeFile) {
        outputFile = "results/ACO_" + config.name + "_run" + std::to_string(runNumber) + ".txt";
        out.open(outputFile);
        if (!out.is_open()) {
            std::cerr << "Failed to create output file: " << outputFile << std::endl;
            writeFile = false;
        }
    }

    if (writeFile) {
        out << "============================================" << std::endl;
        out << "ANT COLONY OPTIMIZATION RESULTS" << std::endl;
        out << "============================================" << std::endl;
        out << "Instance: " << config.name << std::endl;
        out << "Cities: " << tsp.dimension << std::endl;
        out << "Optimal Tour Length: " << config.optimalTourLength << std::endl;
        out << "Run Number: " << runNumber << std::endl;
        out << "============================================" << std::endl;
        out << std::endl;
    }

    // Write ACO parameters (using actual values!)
    if (writeFile) {
        out << "--- ACO Parameters ---" << std::endl;
        out << "Number of Ants: " << numAnts << std::endl;
        out << "Max Iterations: 2000" << std::endl;
        out << "Alpha (Pheromone Importance): 1.0" << std::endl;
        out << "Beta (Heuristic Importance):   2.0" << std::endl;
        out << "Rho (Evaporation Rate):        0.1" << std::endl;
        out << "Local Search:                  2-opt improvement, periodic LK" << std::endl;
        out << "Diversity Controls:            Immigrants(10% every 15 iter), Adaptive evaporation" << std::endl;
        out << "Elitist Strategy:              Best-so-far + elite ants (3)" << std::endl;
        out << "Early Termination:             Enabled (gap threshold, stagnation)" << std::endl;
        out << std::endl;
    }

    // Write final results
    if (writeFile) {
        out << "--- Final Results ---" << std::endl;
        out << "Best Tour Distance: " << std::fixed << std::setprecision(2) << best.distance << std::endl;
        out << "Optimal Distance: " << config.optimalTourLength << std::endl;
        out << "Gap from Optimal: " << std::fixed << std::setprecision(4)
            << tsp.getGapFromOptimal(best.distance) << "%" << std::endl;
        out << "Execution Time:          " << duration.count() << " ms (" << (duration.count() / 1000.0) << " s)" << std::endl;
        out << "Final Iteration:           " << aco.getStatsHistory().size() << std::endl;

        // Calculate and display improvement
        if (!aco.getStatsHistory().empty()) {
            double initialBest = aco.getStatsHistory().front().bestDistance;
            double finalBest = aco.getStatsHistory().back().bestDistance;
            double improvement = ((initialBest - finalBest) / initialBest) * 100.0;
            out << "Initial Best Distance:   " << std::fixed << std::setprecision(2) << initialBest << std::endl;
            out << "Improvement Achieved:    " << std::fixed << std::setprecision(2) << improvement << "%" << std::endl;
        }
        out << std::endl;
    }

    // Write iteration statistics
    if (writeFile) {
        out << "--- Iteration Statistics ---" << std::endl;
        out << std::setw(8) << "Iter"
            << std::setw(15) << "Best"
            << std::setw(15) << "Mean"
            << std::setw(15) << "Worst"
            << std::setw(15) << "StdDev"
            << std::setw(12) << "Delta" << std::endl;
        out << std::string(79, '-') << std::endl;

        const auto& stats = aco.getStatsHistory();
        // Show every iteration for the first 10, then every 100th, plus last
        for (size_t i = 0; i < stats.size(); i++) {
            const auto& s = stats[i];
            bool show = (i < 10 || s.iteration % 100 == 0 || i == stats.size() - 1);

            if (show) {
                double gap = tsp.getGapFromOptimal(s.bestDistance);
                out << std::setw(6) << s.iteration
                    << std::setw(13) << std::fixed << std::setprecision(2) << s.bestDistance
                    << std::setw(13) << std::setprecision(2) << s.meanDistance
                    << std::setw(13) << std::setprecision(2) << s.worstDistance
                    << std::setw(12) << std::setprecision(2) << s.stdDevDistance
                    << std::setw(10) << std::setprecision(2) << s.delta
                    << std::setw(10) << std::setprecision(2) << gap << std::endl;
            }
        }
        out << std::endl;

        // Add performance metrics
        out << "--- PERFORMANCE METRICS ---" << std::endl;

        if (!aco.getStatsHistory().empty()) {
            // Find best iteration
            int bestIter = 0;
            double bestDist = aco.getStatsHistory()[0].bestDistance;
            for (size_t i = 0; i < aco.getStatsHistory().size(); i++) {
                if (aco.getStatsHistory()[i].bestDistance < bestDist) {
                    bestDist = aco.getStatsHistory()[i].bestDistance;
                    bestIter = aco.getStatsHistory()[i].iteration;
                }
            }

            out << "Iteration of Best Solution: " << bestIter << std::endl;

            // Calculate convergence characteristics
            if (aco.getStatsHistory().size() > 1) {
                double improvementFirstHalf = 0, improvementSecondHalf = 0;
                size_t half = aco.getStatsHistory().size() / 2;

                for (size_t i = 1; i < half; i++) {
                    improvementFirstHalf += aco.getStatsHistory()[i-1].bestDistance - aco.getStatsHistory()[i].bestDistance;
                }
                for (size_t i = half; i < aco.getStatsHistory().size(); i++) {
                    improvementSecondHalf += aco.getStatsHistory()[i-1].bestDistance - aco.getStatsHistory()[i].bestDistance;
                }

                out << "Improvement in First Half:   " << std::fixed << std::setprecision(2) << improvementFirstHalf << std::endl;
                out << "Improvement in Second Half:  " << std::fixed << std::setprecision(2) << improvementSecondHalf << std::endl;
                out << "Convergence Pattern:         ";
                if (improvementSecondHalf > improvementFirstHalf * 0.5) {
                    out << "Good late optimization" << std::endl;
                } else {
                    out << "Early convergence" << std::endl;
                }
            }
        }

        out << std::endl;

        // Add algorithm behavior summary
        out << "--- ALGORITHM BEHAVIOR ANALYSIS ---" << std::endl;
        out << "Population Statistics:" << std::endl;
        out << "  Initial Best Distance:     " << (aco.getStatsHistory().empty() ? 0.0 : aco.getStatsHistory().front().bestDistance) << std::endl;
        if (!aco.getStatsHistory().empty()) {
            const auto& final_stats = aco.getStatsHistory().back();
            out << "  Final Best Distance:       " << final_stats.bestDistance << std::endl;
        }
        out << "  Average Standard Deviation: ";

        if (!aco.getStatsHistory().empty()) {
            double avgStdDev = 0;
            for (const auto& stat : aco.getStatsHistory()) {
                avgStdDev += stat.stdDevDistance;
            }
            avgStdDev /= aco.getStatsHistory().size();
            out << std::fixed << std::setprecision(2) << avgStdDev << std::endl;
        } else {
            out << "0.00" << std::endl;
        }

        if (!aco.getStatsHistory().empty()) {
            const auto& final_stats = aco.getStatsHistory().back();
            out << "  Convergence Status:        " << (final_stats.stdDevDistance < 1.0 ? "Converged" : "Diverse Population") << std::endl;
            out << "  Search Efficiency:         ";

            // Calculate search efficiency metric
            if (!aco.getStatsHistory().empty() && aco.getStatsHistory().front().bestDistance > final_stats.bestDistance) {
                double improvement_percent = ((aco.getStatsHistory().front().bestDistance - final_stats.bestDistance) / aco.getStatsHistory().front().bestDistance) * 100;
                out << std::fixed << std::setprecision(2) << improvement_percent << "% improvement" << std::endl;
            } else {
                out << "No improvement" << std::endl;
            }
        }
        out << std::endl;

        // Write final comprehensive summary
        out << "=========================================================" << std::endl;
        out << "                    RUN SUMMARY                            " << std::endl;
        out << "=========================================================" << std::endl;
        out << "Instance:  " << config.name << " (" << tsp.dimension << " cities)" << std::endl;
        out << "Run:       " << runNumber << std::endl;
        out << "Result:    " << std::fixed << std::setprecision(2) << best.distance
            << " (Gap: " << std::setprecision(4) << tsp.getGapFromOptimal(best.distance) << "%)" << std::endl;
        out << "Time:      " << (duration.count() / 1000.0) << " seconds" << std::endl;
        out << "Quality:   " << (tsp.getGapFromOptimal(best.distance) < 2.0 ? "EXCELLENT" :
                          tsp.getGapFromOptimal(best.distance) < 5.0 ? "GOOD" :
                          tsp.getGapFromOptimal(best.distance) < 10.0 ? "FAIR" : "POOR") << std::endl;
        out << "=========================================================" << std::endl;
        out << "END OF RESULTS" << std::endl;
        out << "=========================================================" << std::endl;

        out.close();

    }

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

    return res;
}

// Run multiple times and aggregate stats (mean, stddev, min, max)
void runExperimentMulti(const InstanceConfig& config, int runs, int runNumberForSummary = 1) {
    std::vector<std::future<RunResult>> futures;
    futures.reserve(runs);

    // Launch runs in parallel - optimized for hardware with memory-intensive operations
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

    // For TSP with large instances, fewer threads prevent memory contention and cache thrashing
    int num_threads;
    if (instance_size > 1000) {
        // Use 8 threads for very large instances to balance performance vs memory contention
        num_threads = 8;
    } else if (instance_size > 500) {
        // Use 8 threads for large instances to prevent memory bottleneck
        num_threads = 8;
    } else {
        // Use up to 12 threads for smaller instances
        num_threads = std::min(12, static_cast<int>(std::thread::hardware_concurrency()));
    }
    std::cout << "Running " << runs << " experiments for " << config.name << " using up to " << num_threads << " threads" << std::endl;

    for (int r = 1; r <= runs; r++) {
        futures.push_back(std::async(std::launch::async, [config, r]() {
            return runExperiment(config, r, false, true, true); // Include detailed output for each run
        }));
    }

    std::vector<double> distances;
    distances.reserve(runs);
    std::vector<long long> times;
    times.reserve(runs);

    RunResult bestRun;
    bestRun.distance = std::numeric_limits<double>::max();

    // Collect results
    for (int r = 0; r < runs; r++) {
        RunResult res = futures[r].get(); // Wait for each future to complete
        distances.push_back(res.distance);
        times.push_back(res.durationMs);
        if (res.distance < bestRun.distance) {
            bestRun = res; // keep only stats, not tour
        }
        // Output simplified progress for each run
        std::cout << "Run " << (r + 1) << " gap: " << res.gap << "%" << std::endl;
    }

    double sum = 0.0, sumSq = 0.0;
    long long timeSum = 0;
    double bestGap = std::numeric_limits<double>::max();
    double worstGap = -std::numeric_limits<double>::max();
    double minDist = std::numeric_limits<double>::max();
    double maxDist = 0.0;

    for (int i = 0; i < runs; i++) {
        double d = distances[i];
        sum += d;
        sumSq += d * d;
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        double gap = ((d - config.optimalTourLength) / config.optimalTourLength) * 100.0;
        bestGap = std::min(bestGap, gap);
        worstGap = std::max(worstGap, gap);
        timeSum += times[i];
    }

    double mean = sum / runs;
    double var = std::max(0.0, (sumSq / runs) - mean * mean);
    double stddev = std::sqrt(var);
    double meanTime = static_cast<double>(timeSum) / runs;

    std::string summaryFile = "results/ACO_" + config.name + "_run" + std::to_string(runNumberForSummary) + ".txt";
    std::ofstream out(summaryFile);
    if (!out.is_open()) {
        std::cerr << "Failed to write summary: " << summaryFile << std::endl;
        return;
    }

    // Minimal single-line summary (space-separated)
    // Fields: instance runs mean stddev min max best_gap worst_gap mean_time_ms best_run_num best_dist best_gap best_time_ms
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

    std::cout << "Summary saved to: " << summaryFile << std::endl;
}

// Function to run instances sequentially, with parallel runs within each instance
void runAllInstancesSequentially(const std::vector<InstanceConfig>& instances, int runs) {
    for (const auto& inst : instances) {
        std::cout << "Starting " << inst.name << " (" << runs << " runs)..." << std::endl;
        runExperimentMulti(inst, runs, 1);  // This already runs 30 threads per instance
    }

    std::cout << "\nAll ACO experiments completed!" << std::endl;
}

int main(int argc, char* argv[]) {
    // Only the last 4 large instances (>1000 nodes) - as requested
    std::vector<InstanceConfig> instances = {
        // {"berlin52",  "data/tsplib/berlin52.tsp",  7542},   // 52 nodes - commented out
        // {"kroA100",   "data/tsplib/kroA100.tsp",   21282},  // 100 nodes - commented out
        // {"ch150",     "data/tsplib/ch150.tsp",     6528},   // 150 nodes - commented out
        // {"kroA200",   "data/tsplib/kroA200.tsp",   29368},  // 200 nodes - commented out
        // {"a280",      "data/tsplib/a280.tsp",      2579},   // 280 nodes - commented out
        // {"pcb442",    "data/tsplib/pcb442.tsp",    50778},  // 442 nodes - commented out
        {"pr1002",    "data/tsplib/pr1002.tsp",    259045},  // 1002 nodes
        {"dka1376",   "data/tsplib/dka1376.tsp",   4666},    // 1376 nodes
        {"djc1785",   "data/tsplib/djc1785.tsp",   6115},    // 1785 nodes
        {"pia3056",   "data/tsplib/pia3056.tsp",   8258}     // 3056 nodes
    };

    // Run all instances with 30 runs each (automatic execution)
    int runs = 30;

    // Run instances sequentially, with 30 parallel runs per instance
    runAllInstancesSequentially(instances, runs);

    return 0;
}