#ifndef ANT_COLONY_H
#define ANT_COLONY_H

#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <limits>
#include <cmath>
#include "TSPInstance.h"
#include "LinKernighan.h"

/**
 * @brief Represents an individual ant (solution) in the ACO
 */
struct Ant {
    std::vector<int> tour;      // Permutation of cities (Hamiltonian cycle)
    double distance;            // Total tour length
    double fitness;             // Fitness value (1.0 / distance)

    Ant() : distance(0.0), fitness(0.0) {}

    Ant(int numCities) : distance(0.0), fitness(0.0) {
        tour.resize(numCities);
    }

    // Comparison operator for sorting (higher fitness = better)
    bool operator<(const Ant& other) const {
        return fitness > other.fitness;  // Descending order (best first)
    }
};

/**
 * @brief Ant Colony Optimization Algorithm for solving TSP
 * Implements the Ant System with optional elitist strategy and local search
 */
class AntColony {
private:
    // Problem instance
    const TSPInstance& tsp;

    // ACO Parameters
    int numAnts;                    // Number of ants (population)
    int maxIterations;              // Maximum number of iterations
    double alpha;                   // Pheromone importance (1.0)
    double beta;                    // Heuristic importance (2.0-5.0)
    double rho;                     // Pheromone evaporation rate (0.05-0.2)
    double q0;                     // Local optimization probability (0.8-0.95)
    int eliteAnts;                 // Number of elitist ants (1-5)

    // Diversity controls
    double localSearchProbability;  // Probability to apply local search on a solution
    double immigrantRate;           // Fraction of replaced solutions on schedule
    int immigrantInterval;          // How often (iterations) to inject immigrants
    int restartThreshold;           // Iterations without improvement before restart
    double restartPheromoneIntensification; // Pheromone intensification after restart
    int localSearchStartIter;       // Burn-in iterations before applying local search
    bool forceFullIterations;       // If true, never stop early before maxIterations

    // Early termination controls
    int noImprovementLimit;         // Stop if no improvement for this many iterations
    double targetGap;               // Stop if gap from optimal is below this threshold
    bool enableEarlyTermination;    // Whether to allow early termination

    // Pheromone matrix and heuristic information
    std::vector<std::vector<double>> pheromone;    // Pheromone levels
    std::vector<std::vector<double>> eta;          // Heuristic information (1/distance)
    std::vector<std::vector<double>> eta_alpha;    // eta^beta for efficiency
    std::vector<std::vector<double>> transition_prob; // Temporary storage for transition probabilities

    // Current population of ants
    std::vector<Ant> ants;

    // Best solution found
    Ant bestSolution;
    Ant iterationBest;             // Best solution in current iteration

    // Random number generator
    std::mt19937 rng;

    // Statistics tracking
    std::vector<double> bestDistanceHistory;
    std::vector<double> meanDistanceHistory;

public:
    /**
     * @brief Constructor with default parameters
     * Typical ACO settings balanced for TSP
     */
    AntColony(const TSPInstance& instance)
        : tsp(instance),
          numAnts(200),           // Matching GA population size
          maxIterations(2000),    // Assignment limit (same as GA)
          alpha(1.0),             // Pheromone importance
          beta(2.0),              // Heuristic importance
          rho(0.1),               // Evaporation rate
          q0(0.8),                // Local optimization probability
          eliteAnts(3),           // Elitist ants (similar to GA elitism)
          localSearchProbability(0.25),
          immigrantRate(0.10),    // Similar to GA
          immigrantInterval(15),  // Same as GA
          restartThreshold(40),   // Same as GA
          restartPheromoneIntensification(2.0),
          localSearchStartIter(200),
          forceFullIterations(true) {

        rng.seed(static_cast<unsigned>(time(nullptr)));
        bestSolution.distance = std::numeric_limits<double>::max();
        iterationBest.distance = std::numeric_limits<double>::max();

        // Initialize early termination parameters
        noImprovementLimit = 300;  // Same as GA
        targetGap = 2.0;          // Same as GA
        enableEarlyTermination = true;

        // Initialize matrices
        pheromone.resize(tsp.dimension, std::vector<double>(tsp.dimension, 1.0));
        eta.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        eta_alpha.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        transition_prob.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));

        // Precompute heuristic information
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                if (i != j && tsp.getDistance(i, j) > 0) {
                    eta[i][j] = 1.0 / tsp.getDistance(i, j);
                    eta_alpha[i][j] = std::pow(eta[i][j], beta);
                }
            }
        }
    }

    // ==================== SETTERS FOR PARAMETERS ====================

    void setNumAnts(int ants) { numAnts = ants; }
    void setMaxIterations(int iter) { maxIterations = iter; }
    void setAlpha(double a) { alpha = a; }
    void setBeta(double b) { beta = b; }
    void setRho(double r) { rho = r; }
    void setQ0(double q) { q0 = q; }
    void setEliteAnts(int e) { eliteAnts = e; }
    void setLocalSearchProbability(double p) { localSearchProbability = p; }
    void setImmigrantRate(double r) { immigrantRate = r; }
    void setImmigrantInterval(int interval) { immigrantInterval = interval; }
    void setRestartThreshold(int threshold) { restartThreshold = threshold; }
    void setLocalSearchStartIter(int start) { localSearchStartIter = start; }
    void setForceFullIterations(bool force) { forceFullIterations = force; }

    // Early termination setters
    void setNoImprovementLimit(int limit) { noImprovementLimit = limit; }
    void setTargetGap(double gap) { targetGap = gap; }
    void setEnableEarlyTermination(bool enable) { enableEarlyTermination = enable; }

    // Reseed method for unique random seed per run
    void reseed(unsigned seed) { rng.seed(seed); }

    // ==================== SOLUTION CONSTRUCTION ====================

    /**
     * @brief Construct a solution for an ant using probability-based selection
     */
    Ant constructSolution() {
        Ant ant(tsp.dimension);
        std::vector<bool> visited(tsp.dimension, false);
        std::uniform_int_distribution<int> startDist(0, tsp.dimension - 1);

        // Start from a random city
        int currentCity = startDist(rng);
        ant.tour[0] = currentCity;
        visited[currentCity] = true;

        // Build the rest of the tour
        for (int i = 1; i < tsp.dimension; i++) {
            // Calculate transition probabilities for unvisited cities
            std::vector<std::pair<double, int>> weightedCities; // (probability, city)
            double sum = 0.0;

            for (int j = 0; j < tsp.dimension; j++) {
                if (!visited[j]) {
                    double tau_alpha = std::pow(pheromone[currentCity][j], alpha);
                    double eta_beta = std::pow(eta[currentCity][j], beta);
                    double probability = tau_alpha * eta_beta;
                    weightedCities.push_back({probability, j});
                    sum += probability;
                }
            }

            // Select next city based on probability
            int nextCity = -1;
            if (sum > 1e-10) { // Avoid division by zero
                std::uniform_real_distribution<double> probDist(0.0, sum);
                double r = probDist(rng);
                double cumulative = 0.0;

                for (auto& pair : weightedCities) {
                    cumulative += pair.first;
                    if (r <= cumulative) {
                        nextCity = pair.second;
                        break;
                    }
                }

                // Fallback in case of floating point precision issues
                if (nextCity == -1 && !weightedCities.empty()) {
                    nextCity = weightedCities[0].second;
                }
            } else {
                // If all probabilities are effectively zero, choose randomly among unvisited
                std::vector<int> unvisited;
                for (int j = 0; j < tsp.dimension; j++) {
                    if (!visited[j]) {
                        unvisited.push_back(j);
                    }
                }
                if (!unvisited.empty()) {
                    std::uniform_int_distribution<int> unvDist(0, unvisited.size() - 1);
                    nextCity = unvisited[unvDist(rng)];
                }
            }

            if (nextCity != -1) {
                ant.tour[i] = nextCity;
                visited[nextCity] = true;
                currentCity = nextCity;
            } else {
                // This should not happen, but just in case
                std::cerr << "Error in solution construction" << std::endl;
                break;
            }
        }

        // Calculate fitness
        evaluateAnt(ant);
        return ant;
    }

    /**
     * @brief Evaluate fitness of an ant
     */
    void evaluateAnt(Ant& ant) {
        ant.distance = tsp.calculateTourLength(ant.tour);
        ant.fitness = 1.0 / ant.distance;
    }

    /**
     * @brief Update best solution if better found
     */
    void updateBestSolution(const Ant& ant) {
        if (ant.distance > 1e-6 && ant.distance < bestSolution.distance) {
            bestSolution = ant;
        }
    }

    /**
     * @brief Update iteration best solution if better found
     */
    void updateIterationBest(const Ant& ant) {
        if (ant.distance < iterationBest.distance) {
            iterationBest = ant;
        }
    }

    // ==================== PHEROMONE UPDATE ====================

    /**
     * @brief Global pheromone update (evaporation and deposition)
     */
    void globalPheromoneUpdate() {
        // Evaporate pheromone
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                pheromone[i][j] *= (1.0 - rho);
            }
        }

        // Deposit pheromone for best solution found so far
        if (iterationBest.distance < bestSolution.distance) {
            // Update global best if iteration best is better
            bestSolution = iterationBest;
        }

        for (int i = 0; i < tsp.dimension; i++) {
            int from = bestSolution.tour[i];
            int to = bestSolution.tour[(i + 1) % tsp.dimension];
            pheromone[from][to] += 1.0 / bestSolution.distance;
            pheromone[to][from] += 1.0 / bestSolution.distance; // Symmetric TSP
        }

        // Elitist strategy: add extra pheromone for elite ants
        // Sort current iteration's ants to identify elites
        std::vector<Ant> sortedAnts = ants;
        std::sort(sortedAnts.begin(), sortedAnts.end());

        for (int e = 0; e < eliteAnts && e < sortedAnts.size(); e++) {
            for (int i = 0; i < tsp.dimension; i++) {
                int from = sortedAnts[e].tour[i];
                int to = sortedAnts[e].tour[(i + 1) % tsp.dimension];
                pheromone[from][to] += 1.0 / sortedAnts[e].distance; // Additional pheromone
                pheromone[to][from] += 1.0 / sortedAnts[e].distance;
            }
        }
    }

    /**
     * @brief Local pheromone update (for diversification)
     */
    void localPheromoneUpdate(const std::vector<int>& tour) {
        double tau0 = 1.0 / (tsp.dimension * tsp.getDistance(0, 1)); // Initial pheromone level

        for (int i = 0; i < tsp.dimension; i++) {
            int from = tour[i];
            int to = tour[(i + 1) % tsp.dimension];
            // Decrease pheromone level to promote exploration
            pheromone[from][to] = (1.0 - rho) * pheromone[from][to] + rho * tau0;
            pheromone[to][from] = (1.0 - rho) * pheromone[to][from] + rho * tau0;
        }
    }

    // ==================== 2-OPT LOCAL SEARCH ====================

    /**
     * @brief Apply 2-opt local search to an ant's solution
     */
    void twoOptImprove(Ant& ant) {
        std::vector<int> improvedTour = ant.tour;
        LinKernighan lk(tsp);
        improvedTour = lk.optimize(improvedTour);
        
        Ant improvedAnt = ant;
        improvedAnt.tour = improvedTour;
        evaluateAnt(improvedAnt);
        
        if (improvedAnt.distance < ant.distance) {
            ant = improvedAnt;
        }
    }

    /**
     * @brief Apply Lin-Kernighan optimization to an ant's solution
     */
    void linKernighanOptimize(Ant& ant) {
        LinKernighan lk(tsp);
        ant.tour = lk.optimizeFull(ant.tour);
        // Always recalculate distance from TSP instance to ensure correctness
        ant.distance = tsp.calculateTourLength(ant.tour);
        ant.fitness = 1.0 / ant.distance;
    }

    // ==================== STATISTICS ====================

    struct IterationStats {
        double bestDistance;
        double worstDistance;
        double meanDistance;
        double stdDevDistance;
        double delta;  // improvement from previous iteration
        int iteration;
    };

    std::vector<IterationStats> statsHistory;

    IterationStats calculateStats(int iteration, double prevBest) {
        IterationStats stats;
        stats.iteration = iteration;

        double sum = 0.0;
        double sumSq = 0.0;
        stats.bestDistance = std::numeric_limits<double>::max();
        stats.worstDistance = 0.0;

        for (const auto& ant : ants) {
            sum += ant.distance;
            sumSq += ant.distance * ant.distance;
            if (ant.distance < stats.bestDistance) {
                stats.bestDistance = ant.distance;
            }
            if (ant.distance > stats.worstDistance) {
                stats.worstDistance = ant.distance;
            }
        }

        int n = ants.size();
        stats.meanDistance = sum / n;
        double variance = (sumSq / n) - (stats.meanDistance * stats.meanDistance);
        stats.stdDevDistance = std::sqrt(std::max(0.0, variance));
        stats.delta = prevBest - stats.bestDistance;  // positive = improvement

        return stats;
    }

    const std::vector<IterationStats>& getStatsHistory() const {
        return statsHistory;
    }

    void printStats(const IterationStats& stats) {
        std::cout << "Iter " << stats.iteration
                  << " | Best: " << stats.bestDistance
                  << " | Mean: " << stats.meanDistance
                  << " | Worst: " << stats.worstDistance
                  << " | StdDev: " << stats.stdDevDistance
                  << " | Delta: " << stats.delta << std::endl;
    }

    // ==================== MAIN ACO LOOP ====================

    /**
     * @brief Run the Ant Colony Optimization algorithm
     * @param verbose If true, print progress
     * @return Best solution found
     */
    Ant run(bool verbose = true) {
        // Initialize ants
        ants.reserve(numAnts);
        for (int i = 0; i < numAnts; i++) {
            ants.push_back(Ant(tsp.dimension));
        }

        // Initialize best and iteration best solutions
        bestSolution.distance = std::numeric_limits<double>::max();
        iterationBest.distance = std::numeric_limits<double>::max();

        // Statistics
        double prevBest = bestSolution.distance;
        int stagnationCounter = 0;
        int diversityInjections = 0;  // Count how many times we injected diversity
        double currentRho = rho;      // Adaptive evaporation rate
        const int stagnationThreshold = 10;    // Same as GA: 10 iterations without improvement
        const double maxRho = 0.5;    // Higher evaporation when stuck
        const double minRho = 0.05;   // Lower floor for evaporation

        if (verbose) {
            std::cout << "\n=== Starting ACO Evolution ===" << std::endl;
            std::cout << "Ants: " << numAnts
                      << " | Max iterations: " << maxIterations
                      << " | Alpha: " << alpha
                      << " | Beta: " << beta
                      << " | Rho: " << rho << std::endl;
        }

        // Main ACO loop
        for (int iter = 1; iter <= maxIterations; iter++) {
            // Reset iteration best
            iterationBest.distance = std::numeric_limits<double>::max();

            // Each ant constructs a solution
            for (int antIdx = 0; antIdx < numAnts; antIdx++) {
                // Construct solution
                ants[antIdx] = constructSolution();

                // Local search (2-opt) on ant's solution
                // Delay local search until burn-in to avoid early homogenization
                if (iter >= localSearchStartIter) {
                    std::uniform_real_distribution<double> prob01(0.0, 1.0);
                    double lsProb = localSearchProbability;
                    // Reduce local search probability for larger instances to save time
                    if (tsp.dimension > 500) {
                        lsProb *= 0.5;
                    }
                    // Further reduce if we're in the later stages to save time
                    if (iter > maxIterations * 0.8) {
                        lsProb *= 0.7; // Reduce in final 20% of iterations
                    }

                    if (prob01(rng) < lsProb) {
                        twoOptImprove(ants[antIdx]);
                    }
                }

                // Update iteration best and global best solutions
                updateIterationBest(ants[antIdx]);
                updateBestSolution(ants[antIdx]);
            }

            // Calculate population diversity for convergence detection
            double currentDiversity = calculatePopulationDiversity();

            // Apply diversity control if convergence detected
            applyDiversityControl(currentDiversity);

            // Pheromone update
            globalPheromoneUpdate();

            // Scheduled immigrant injection (diversity maintenance)
            if (immigrantInterval > 0 && iter % immigrantInterval == 0) {
                int immigrantCount = std::max(1, (int)std::round(numAnts * immigrantRate));
                int protectCount = std::max(eliteAnts, numAnts / 10); // keep top 10% safe
                for (int i = 0; i < immigrantCount; i++) {
                    int replaceIdx = numAnts - 1 - i;
                    if (replaceIdx <= protectCount) break;
                    ants[replaceIdx] = constructSolution(); // Replace with new random solution
                }
                diversityInjections++;
            }

            // Calculate and store statistics
            IterationStats stats = calculateStats(iter, prevBest);
            statsHistory.push_back(stats);

            // Additional diversity injection when population becomes too homogeneous
            if (currentDiversity < 0.05) { // If less than 5% of positions differ on average
                int injectCount = numAnts / 4; // Replace worst 25%
                int startIdx = numAnts - injectCount;
                for (int i = 0; i < injectCount; i++) {
                    int idx = startIdx + i;
                    ants[idx] = constructSolution();
                }
                currentRho = std::min(maxRho, currentRho * 2.0);
                stagnationCounter = 0;
                diversityInjections++;
            }

            // Check for stagnation (adaptive pheromone management + diversity injection)
            if (stats.delta < 1e-9) {
                stagnationCounter++;

                // After 10 iterations stuck, take action
                if (stagnationCounter >= stagnationThreshold) {
                    // Increase evaporation rate to promote exploration
                    currentRho = std::min(maxRho, currentRho * 2.0);

                    // PROTECTED ELITE + DIVERSITY INJECTION
                    // Protect top 20% (40 ants), replace bottom 30% (60 ants)
                    int protectedCount = numAnts / 5;     // 20% = 40 protected
                    int injectCount = (numAnts * 3) / 10; // 30% = 60 to replace
                    int startIdx = numAnts - injectCount;

                    for (int i = 0; i < injectCount; i++) {
                        int targetIdx = startIdx + i;

                        // Alternate between random immigrants and solutions with increased pheromone
                        if (i % 2 == 0) {
                            // Random immigrant
                            ants[targetIdx] = constructSolution();
                        } else {
                            // Take from protected elite and apply some random changes
                            Ant mutant = ants[i % protectedCount];
                            // Perform some random changes to the solution
                            std::uniform_int_distribution<int> dist(0, tsp.dimension - 1);
                            int pos1 = dist(rng);
                            int pos2 = dist(rng);
                            if (pos1 != pos2) {
                                std::swap(mutant.tour[pos1], mutant.tour[pos2]);
                                evaluateAnt(mutant);
                            }
                            ants[targetIdx] = mutant;
                        }
                    }

                    // Reset stagnation counter after injection
                    stagnationCounter = 0;
                    diversityInjections++;
                }

                // Hard restart after prolonged stall
                if (stagnationCounter >= restartThreshold) {
                    int keep = std::max(1, numAnts / 20); // keep top 5%
                    std::sort(ants.begin(), ants.end());
                    for (int i = keep; i < numAnts; i++) {
                        ants[i] = constructSolution();
                    }
                    // Intensify pheromone on best solution
                    for (int i = 0; i < tsp.dimension; i++) {
                        int from = bestSolution.tour[i];
                        int to = bestSolution.tour[(i + 1) % tsp.dimension];
                        pheromone[from][to] *= restartPheromoneIntensification;
                        pheromone[to][from] *= restartPheromoneIntensification;
                    }
                    currentRho = rho; // Reset evaporation to normal
                    stagnationCounter = 0;
                }
            } else {
                stagnationCounter = 0;
                // Decrease evaporation rate if improving
                currentRho = std::max(minRho, currentRho * 0.95);
            }

            prevBest = bestSolution.distance;

            // Print progress
            if (verbose && (iter % 100 == 0 || iter == 1)) {
                printStats(stats);
                if (tsp.optimalTourLength > 0) {
                    double gap = tsp.getGapFromOptimal(bestSolution.distance);
                    std::cout << "  Gap: " << gap << "% | Injections: " << diversityInjections
                              << " | Rho: " << currentRho << std::endl;
                }
            }

            // Early termination checks (if enabled)
            if (enableEarlyTermination && !forceFullIterations) {
                // Check if gap from optimal is within target threshold
                if (tsp.optimalTourLength > 0) {
                    double gap = tsp.getGapFromOptimal(bestSolution.distance);
                    if (gap <= targetGap) {
                        if (verbose) {
                            std::cout << "\n*** Early termination: target gap (" << targetGap << "%) reached at iteration " << iter << " (actual gap: " << gap << "%) ***" << std::endl;
                        }
                        break;
                    }
                }

                // Check if no improvement for specified iterations
                if (stagnationCounter >= noImprovementLimit) {
                    if (verbose) {
                        std::cout << "\n*** Early termination: no improvement for " << noImprovementLimit << " iterations at iteration " << iter << " ***" << std::endl;
                    }
                    break;
                }
            }
        }

        // Make sure bestSolution is properly updated at the end of iterations
        // Check if any ant in the final iteration is better than the stored best
        for (const auto& ant : ants) {
            if (ant.distance > 1e-6 && ant.distance < bestSolution.distance) {
                bestSolution = ant;
            }
        }

        // Final 2-opt on best solution
        Ant finalBest = bestSolution;
        std::vector<int> improvedTour = finalBest.tour;
        LinKernighan lk(tsp);
        improvedTour = lk.optimizeFull(improvedTour);

        finalBest.tour = improvedTour;
        finalBest.distance = tsp.calculateTourLength(finalBest.tour);
        finalBest.fitness = 1.0 / finalBest.distance;

        // Update bestSolution after improvement
        if (finalBest.distance < bestSolution.distance) {
            bestSolution = finalBest;
        }

        // Final LK polish on the best solution
        linKernighanOptimize(finalBest);

        // Update bestSolution if LK improved the solution
        if (finalBest.distance < bestSolution.distance) {
            bestSolution = finalBest;
        }

        if (verbose) {
            std::cout << "\n=== ACO Complete ===" << std::endl;
            std::cout << "Best tour distance: " << bestSolution.distance << std::endl;
            if (tsp.optimalTourLength > 0) {
                double gap = tsp.getGapFromOptimal(bestSolution.distance);
                std::cout << "Gap from optimal: " << gap << "%" << std::endl;
            }
        }

        return bestSolution;
    }

    // ==================== GETTERS ====================

    const Ant& getBestSolution() const { return bestSolution; }
    const std::vector<double>& getBestDistanceHistory() const { return bestDistanceHistory; }
    const std::vector<double>& getMeanDistanceHistory() const { return meanDistanceHistory; }
    int getNumAnts() const { return numAnts; }

    /**
     * @brief Print current best solution info
     */
    void printBestSolution() const {
        std::cout << "Best tour distance: " << bestSolution.distance << std::endl;
        if (tsp.optimalTourLength > 0) {
            double gap = tsp.getGapFromOptimal(bestSolution.distance);
            std::cout << "Gap from optimal: " << gap << "%" << std::endl;
        }
    }

    /**
     * @brief Calculate diversity of the ant population
     * Measures how different the solutions are from each other
     */
    double calculatePopulationDiversity() {
        if (ants.size() < 2) return 0.0;

        double totalDistance = 0.0;
        int count = 0;

        // Calculate pairwise distances between solutions
        for (size_t i = 0; i < ants.size(); i++) {
            for (size_t j = i + 1; j < ants.size(); j++) {
                // Calculate edit distance between tours as a measure of diversity
                int diffCount = 0;
                for (int k = 0; k < tsp.dimension; k++) {
                    if (ants[i].tour[k] != ants[j].tour[k]) {
                        diffCount++;
                    }
                }
                totalDistance += static_cast<double>(diffCount) / tsp.dimension; // Normalize
                count++;
            }
        }

        return count > 0 ? totalDistance / count : 0.0;
    }

    /**
     * @brief Apply intensified pheromone evaporation to increase diversity when convergence detected
     */
    void applyDiversityControl(double currentDiversity) {
        const double DIVERSITY_THRESHOLD = 0.1; // If diversity drops below 10%, apply control

        if (currentDiversity < DIVERSITY_THRESHOLD) {
            // Increase evaporation rate to promote exploration
            double originalRho = rho;
            double newRho = std::min(0.5, rho * 2.0); // Increase evaporation

            // Apply increased evaporation to pheromone matrix
            for (int i = 0; i < tsp.dimension; i++) {
                for (int j = 0; j < tsp.dimension; j++) {
                    pheromone[i][j] = (1.0 - newRho) * pheromone[i][j] + (newRho - rho) * 1.0; // Add some baseline pheromone
                }
            }
        }
    }
};

#endif // ANT_COLONY_H