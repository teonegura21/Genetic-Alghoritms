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
#include "GeneticAlgorithm.h"

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
    std::vector<GeneticAlgorithm::GenerationStats> stats;
    int dimension{0};
};

// Run GA on a single instance and save results to file
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

    // Create GA with tuned defaults inside GeneticAlgorithm
    GeneticAlgorithm ga(tsp);
    int popSize = 200;
    int offspring = 200;  // keep 200 after selection (<= requirement)
    ga.setPopulationSize(popSize);
    ga.setOffspringCount(offspring);
    ga.setMaxGenerations(2000); // respect requirement cap
    // Optional extra tuning for runs from CLI
    ga.setTournamentSize(2);
    ga.setMutationRate(0.25);
    ga.setLocalSearchStartGen(200);
    ga.setLocalSearchProbability(0.25);
    ga.setImmigrantRate(0.10);
    ga.setImmigrantInterval(15);
    ga.setRestartThreshold(40);

    // Enable early termination for better performance
    ga.setEnableEarlyTermination(true);
    ga.setNoImprovementLimit(300);  // Stop after 300 generations without improvement
    ga.setTargetGap(1.5);          // Stop if within 1.5% of known optimal

    // Fix the random seed issue: reseed with unique value based on run number and high-res time
    auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        (runNumber << 16) ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );
    ga.reseed(seed);

    // int elitismPercent = (100 * popSize) / (popSize + offspring);
    int elitismPercent = 5; // informational only

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run GA
    Individual best = ga.run(verbose);

    // End timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::string outputFile;
    std::ofstream out;
    if (writeFile) {
        outputFile = "results/GA_" + config.name + "_run" + std::to_string(runNumber) + ".txt";
        out.open(outputFile);
        if (!out.is_open()) {
            std::cerr << "Failed to create output file: " << outputFile << std::endl;
            writeFile = false;
        }
    }

    if (writeFile) {
        out << "============================================" << std::endl;
        out << "GENETIC ALGORITHM RESULTS" << std::endl;
        out << "============================================" << std::endl;
        out << "Instance: " << config.name << std::endl;
        out << "Cities: " << tsp.dimension << std::endl;
        out << "Optimal Tour Length: " << config.optimalTourLength << std::endl;
        out << "Run Number: " << runNumber << std::endl;
        out << "============================================" << std::endl;
        out << std::endl;
    }

    // Write GA parameters (using actual values!)
    if (writeFile) {
        out << "--- GA Parameters ---" << std::endl;
        out << "Population Size: " << popSize << std::endl;
        out << "Offspring Count: " << offspring << " (Generational Replacement + " << elitismPercent << "% elitism)" << std::endl;
        out << "Max Generations: 2000" << std::endl;
        out << "Crossover Rate:          0.9" << std::endl;
        out << "Base Mutation Rate:      0.25 (adaptive with floor 0.10, max 0.9)" << std::endl;
        out << "Tournament Size: 2" << std::endl;
        out << "Local Search:            2-opt improvement, periodic LK" << std::endl;
        out << "Diversity Controls:      Immigrants(10% every 15 gens), Adaptive mutation" << std::endl;
        out << "Early Termination:       Enabled (gap threshold, stagnation)" << std::endl;
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
        out << "Final Generation:        " << ga.getStatsHistory().size() << std::endl;

        // Calculate and display improvement
        if (!ga.getStatsHistory().empty()) {
            double initialBest = ga.getStatsHistory().front().bestDistance;
            double finalBest = ga.getStatsHistory().back().bestDistance;
            double improvement = ((initialBest - finalBest) / initialBest) * 100.0;
            out << "Initial Best Distance:   " << std::fixed << std::setprecision(2) << initialBest << std::endl;
            out << "Improvement Achieved:    " << std::fixed << std::setprecision(2) << improvement << "%" << std::endl;
        }
        out << std::endl;
    }

    // Skip writing best tour to save memory and reduce output
    // Best tour is available in the Individual object if needed for analysis

    // Write generation statistics
    if (writeFile) {
        out << "--- Generation Statistics ---" << std::endl;
        out << std::setw(8) << "Gen" 
            << std::setw(15) << "Best" 
            << std::setw(15) << "Mean" 
            << std::setw(15) << "Worst"
            << std::setw(15) << "StdDev"
            << std::setw(12) << "Delta" << std::endl;
        out << std::string(79, '-') << std::endl;

        const auto& stats = ga.getStatsHistory();
        for (size_t i = 0; i < stats.size(); i++) {
            const auto& s = stats[i];
            // Show every 100th generation, first, and last
            bool show = (s.generation == 1 || s.generation % 100 == 0 || i == stats.size() - 1);

            if (show) {
                double gap = tsp.getGapFromOptimal(s.bestDistance);
                out << std::setw(6) << s.generation
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

        if (!ga.getStatsHistory().empty()) {
            // Find best generation
            int bestGen = 0;
            double bestDist = ga.getStatsHistory()[0].bestDistance;
            for (size_t i = 0; i < ga.getStatsHistory().size(); i++) {
                if (ga.getStatsHistory()[i].bestDistance < bestDist) {
                    bestDist = ga.getStatsHistory()[i].bestDistance;
                    bestGen = ga.getStatsHistory()[i].generation;
                }
            }

            out << "Generation of Best Solution: " << bestGen << std::endl;

            // Calculate convergence characteristics
            if (ga.getStatsHistory().size() > 1) {
                double improvementFirstHalf = 0, improvementSecondHalf = 0;
                size_t half = ga.getStatsHistory().size() / 2;

                for (size_t i = 1; i < half; i++) {
                    improvementFirstHalf += ga.getStatsHistory()[i-1].bestDistance - ga.getStatsHistory()[i].bestDistance;
                }
                for (size_t i = half; i < ga.getStatsHistory().size(); i++) {
                    improvementSecondHalf += ga.getStatsHistory()[i-1].bestDistance - ga.getStatsHistory()[i].bestDistance;
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
        out << "  Initial Best Distance:     " << (ga.getStatsHistory().empty() ? 0.0 : ga.getStatsHistory().front().bestDistance) << std::endl;
        if (!ga.getStatsHistory().empty()) {
            const auto& final_stats = ga.getStatsHistory().back();
            out << "  Final Best Distance:       " << final_stats.bestDistance << std::endl;
        }
        out << "  Average Standard Deviation: ";

        if (!ga.getStatsHistory().empty()) {
            double avgStdDev = 0;
            for (const auto& stat : ga.getStatsHistory()) {
                avgStdDev += stat.stdDevDistance;
            }
            avgStdDev /= ga.getStatsHistory().size();
            out << std::fixed << std::setprecision(2) << avgStdDev << std::endl;
        } else {
            out << "0.00" << std::endl;
        }

        if (!ga.getStatsHistory().empty()) {
            const auto& final_stats = ga.getStatsHistory().back();
            out << "  Convergence Status:        " << (final_stats.stdDevDistance < 1.0 ? "Converged" : "Diverse Population") << std::endl;
            out << "  Search Efficiency:         ";

            // Calculate search efficiency metric
            if (!ga.getStatsHistory().empty() && ga.getStatsHistory().front().bestDistance > final_stats.bestDistance) {
                double improvement_percent = ((ga.getStatsHistory().front().bestDistance - final_stats.bestDistance) / ga.getStatsHistory().front().bestDistance) * 100;
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
    res.stats = ga.getStatsHistory();
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

    // Launch runs in parallel
    const int num_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));

    for (int r = 1; r <= runs; r++) {
        futures.push_back(std::async(std::launch::async, [config, r]() {
            return runExperiment(config, r, false, false, false); // silent per-run
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

    std::string summaryFile = "results/GA_" + config.name + "_run" + std::to_string(runNumberForSummary) + ".txt";
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

int main(int argc, char* argv[]) {
    // All 10 instances (ordered by size)
    std::vector<InstanceConfig> instances = {
        {"berlin52",  "data/tsplib/berlin52.tsp",  7542},
        {"kroA100",   "data/tsplib/kroA100.tsp",   21282},
        {"ch150",     "data/tsplib/ch150.tsp",     6528},
        {"kroA200",   "data/tsplib/kroA200.tsp",   29368},
        {"a280",      "data/tsplib/a280.tsp",      2579},
        {"pcb442",    "data/tsplib/pcb442.tsp",    50778},
        {"pr1002",    "data/tsplib/pr1002.tsp",    259045},
        {"dka1376",   "data/tsplib/dka1376.tsp",   4666},
        {"djc1785",   "data/tsplib/djc1785.tsp",   6115},
        {"pia3056",   "data/tsplib/pia3056.tsp",   8258}
    };

    // Run all instances with 30 runs each (automatic execution)
    int runs = 30;

    for (const auto& inst : instances) {
        std::cout << "Starting " << inst.name << " (" << runs << " runs)..." << std::endl;
        runExperimentMulti(inst, runs, 1);
    }

    std::cout << "\nAll experiments completed!" << std::endl;

    return 0;
}
