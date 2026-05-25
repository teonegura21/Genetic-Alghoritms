#ifndef ANT_COLONY_OPTIMIZED_H
#define ANT_COLONY_OPTIMIZED_H

#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <limits>
#include <cmath>
#include <queue>
#include <numeric>
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
 * @brief Optimized Ant Colony Optimization Algorithm for solving TSP
 * Implements MAX-MIN Ant System (MMAS) with dynamic parameter adaptation and effective local search
 */
class AntColonyOptimized {
private:
    // Problem instance
    const TSPInstance& tsp;

    // ACO Parameters
    int numAnts;                    // Number of ants (population)
    int maxIterations;              // Maximum number of iterations
    double alpha;                   // Pheromone importance (1.0)
    double beta;                    // Heuristic importance (2.0-5.0)
    double rho;                     // Pheromone evaporation rate (0.05-0.2)
    double tau_min, tau_max;        // Pheromone bounds (MMAS)
    int eliteAnts;                 // Number of elitist ants (1-5)

    // Diversity controls and adaptation
    double localSearchProbability;  // Probability to apply local search on a solution
    int immigrantInterval;          // How often (iterations) to inject immigrants
    int restartThreshold;           // Iterations without improvement before restart
    bool forceFullIterations;       // If true, never stop early before maxIterations

    // Early termination controls
    int noImprovementLimit;         // Stop if no improvement for this many iterations
    double targetGap;               // Stop if gap from optimal is below this threshold
    bool enableEarlyTermination;    // Whether to allow early termination

    // Pheromone matrix and heuristic information
    std::vector<std::vector<double>> pheromone;    // Pheromone levels
    std::vector<std::vector<double>> eta;          // Heuristic information (1/distance)
    std::vector<std::vector<double>> eta_beta;     // eta^beta for efficiency
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
     * @brief Constructor with optimized parameters for different problem sizes
     */
    AntColonyOptimized(const TSPInstance& instance)
        : tsp(instance),
          numAnts(200),
          maxIterations(2000),
          alpha(1.0),
          beta(2.5),              // Increased for better heuristic guidance
          rho(0.02),              // Reduced for slower convergence on large problems
          eliteAnts(5),           // Increased for better elite guidance
          localSearchProbability(0.3),
          immigrantInterval(20),   // Less frequent but more effective
          restartThreshold(100),   // More patience before restart
          forceFullIterations(true) {

        rng.seed(static_cast<unsigned>(time(nullptr)));
        bestSolution.distance = std::numeric_limits<double>::max();
        iterationBest.distance = std::numeric_limits<double>::max();

        // Initialize early termination parameters
        noImprovementLimit = 500;  // More patience on large problems
        targetGap = 1.0;          // Tighter target
        enableEarlyTermination = true;

        // Initialize matrices
        pheromone.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        eta.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        eta_beta.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        transition_prob.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));

        // Problem-specific tuning (respecting 200 ants / 2000 iterations constraint)
        if (tsp.dimension >= 1000) {
            alpha = 0.8;            // Slightly reduced pheromone importance
            beta = 3.0;             // Increased heuristic importance
            rho = 0.01;             // Even slower evaporation for large problems
            eliteAnts = 8;          // More elite ants
            localSearchProbability = 0.4; // More local search to compensate
            immigrantInterval = 15; // More diversity injection
            restartThreshold = 200; // Patient but within iteration limit
        } else if (tsp.dimension >= 500) {
            alpha = 0.9;
            beta = 2.8;
            rho = 0.015;
            eliteAnts = 6;
            localSearchProbability = 0.35;
        }

        // Precompute heuristic information and initialize pheromone matrix
        double maxDist = 0.0, minDist = std::numeric_limits<double>::max();
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                if (i != j && tsp.getDistance(i, j) > 0) {
                    double dist = tsp.getDistance(i, j);
                    eta[i][j] = 1.0 / dist;
                    eta_beta[i][j] = std::pow(eta[i][j], beta);
                    maxDist = std::max(maxDist, dist);
                    minDist = std::min(minDist, dist);
                }
            }
        }

        // Calculate tau_max and tau_min based on problem characteristics (MMAS approach)
        double initial_tour_length = tsp.dimension * (maxDist + minDist) / 2.0; // Rough estimate
        tau_max = 1.0 / (rho * initial_tour_length);
        tau_min = tau_max / (2.0 * tsp.dimension);

        // Initialize pheromone with different values for different problem sizes
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                if (i != j) {
                    pheromone[i][j] = tau_max; // Start with max pheromone
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
    void setEliteAnts(int e) { eliteAnts = e; }
    void setLocalSearchProbability(double p) { localSearchProbability = p; }
    void setImmigrantInterval(int interval) { immigrantInterval = interval; }
    void setRestartThreshold(int threshold) { restartThreshold = threshold; }
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
     * Uses dynamic selection between exploitation and exploration based on diversity
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
                    double tau_alpha = std::pow(std::max(pheromone[currentCity][j], tau_min), alpha);
                    double eta_beta_val = std::pow(std::max(eta[currentCity][j], 1e-10), beta);
                    double probability = tau_alpha * eta_beta_val;
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
                // If all probabilities are effectively zero, choose among nearest unvisited cities
                std::vector<int> nearestCities;
                double maxEta = 0.0;
                for (int j = 0; j < tsp.dimension; j++) {
                    if (!visited[j] && eta[currentCity][j] > maxEta) {
                        maxEta = eta[currentCity][j];
                        nearestCities.clear();
                        nearestCities.push_back(j);
                    } else if (!visited[j] && eta[currentCity][j] == maxEta) {
                        nearestCities.push_back(j);
                    }
                }
                
                if (!nearestCities.empty()) {
                    std::uniform_int_distribution<int> nearestDist(0, nearestCities.size() - 1);
                    nextCity = nearestCities[nearestDist(rng)];
                }
            }

            // FINAL FALLBACK: If still no city selected (e.g., all eta are 0), pick the first unvisited one
            if (nextCity == -1) {
                 for (int j = 0; j < tsp.dimension; j++) {
                     if (!visited[j]) {
                         nextCity = j;
                         break;
                     }
                 }
            }

            if (nextCity != -1) {
                ant.tour[i] = nextCity;
                visited[nextCity] = true;
                currentCity = nextCity;
            } else {
                // This should not happen, but just in case
                std::cerr << "Error in solution construction" << std::endl;
                // Try to complete the tour with remaining unvisited cities in random order
                std::vector<int> unvisited;
                for (int j = 0; j < tsp.dimension; j++) {
                    if (!visited[j]) {
                        unvisited.push_back(j);
                    }
                }

                // Shuffle and append remaining cities
                std::shuffle(unvisited.begin(), unvisited.end(), rng);
                size_t unvisitedIdx = 0;
                for (int k = i; k < tsp.dimension; k++) {
                    if (unvisitedIdx < unvisited.size()) {
                        ant.tour[k] = unvisited[unvisitedIdx++];
                    } else {
                        // This shouldn't happen if logic is correct
                        break;
                    }
                }
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
        if (ant.distance > 1e-6 && ant.distance < iterationBest.distance) {
            iterationBest = ant;
        }
    }

    // ==================== PHEROMONE UPDATE (MAX-MIN) ====================

    /**
     * @brief Global pheromone update (MAX-MIN Ant System)
     */
    void globalPheromoneUpdate() {
        // Evaporate pheromone
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                pheromone[i][j] *= (1.0 - rho);
                // Ensure pheromone stays within bounds
                pheromone[i][j] = std::min(tau_max, std::max(pheromone[i][j], tau_min));
            }
        }

        // Deposit pheromone for best solution found so far (MAX-MIN)
        for (int i = 0; i < tsp.dimension; i++) {
            int from = bestSolution.tour[i];
            int to = bestSolution.tour[(i + 1) % tsp.dimension];
            pheromone[from][to] += 1.0 / bestSolution.distance;
            // Ensure bounds
            pheromone[from][to] = std::min(tau_max, pheromone[from][to]);
            pheromone[to][from] = pheromone[from][to]; // Symmetric TSP
        }

        // Elitist strategy: add extra pheromone for best ants
        std::vector<Ant> sortedAnts = ants;
        std::sort(sortedAnts.begin(), sortedAnts.end());

        for (size_t e = 0; e < static_cast<size_t>(eliteAnts) && e < sortedAnts.size(); e++) {
            for (int i = 0; i < tsp.dimension; i++) {
                int from = sortedAnts[e].tour[i];
                int to = sortedAnts[e].tour[(i + 1) % tsp.dimension];
                pheromone[from][to] += 1.0 / sortedAnts[e].distance; // Additional pheromone
                // Ensure bounds
                pheromone[from][to] = std::min(tau_max, pheromone[from][to]);
                pheromone[to][from] = pheromone[from][to];
            }
        }
    }

    /**
     * @brief Reset pheromone matrix with good heuristics (restart strategy)
     */
    void resetPheromone() {
        // Initialize with heuristic information-based values
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                if (i != j) {
                    // Use heuristic information as a base for initialization
                    pheromone[i][j] = std::max(tau_min, tau_max * eta[i][j] * tsp.dimension * 0.1);
                    pheromone[i][j] = std::min(tau_max, pheromone[i][j]);
                }
            }
        }
    }

    // ==================== 2-OPT LOCAL SEARCH ====================

    /**
     * @brief Apply 2-opt local search to an ant's solution
     * Optimized for performance on large instances
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
     * @brief Run the Optimized Ant Colony Optimization algorithm
     * @param verbose If true, print progress
     * @return Best solution found
     */
    Ant run(bool verbose = true) {
        // Initialize ants
        ants.clear();
        ants.reserve(numAnts);
        for (int i = 0; i < numAnts; i++) {
            ants.push_back(Ant(tsp.dimension));
        }

        // Initialize best and iteration best solutions
        bestSolution.distance = std::numeric_limits<double>::max();
        iterationBest.distance = std::numeric_limits<double>::max();

        // Safety: Initialize bestSolution with a valid random tour
        // This ensures that even if all ants fail (unlikely), we have a valid structure
        Ant initialAnt(tsp.dimension);
        initialAnt.tour = tsp.generateRandomTour();
        evaluateAnt(initialAnt);
        bestSolution = initialAnt;

        // Statistics
        double prevBest = bestSolution.distance;
        int stagnationCounter = 0;
        int diversityInjections = 0;  // Count how many times we injected diversity
        int restarts = 0;             // Count restarts

        if (verbose) {
            std::cout << "\n=== Starting OPTIMIZED ACO Evolution ===" << std::endl;
            std::cout << "Ants: " << numAnts
                      << " | Max iterations: " << maxIterations
                      << " | Alpha: " << alpha
                      << " | Beta: " << beta
                      << " | Rho: " << rho << std::endl;
            std::cout << "| Tau_min: " << tau_min << " | Tau_max: " << tau_max << std::endl;
        }

        // Main ACO loop
        for (int iter = 1; iter <= maxIterations; iter++) {
            // Reset iteration best
            iterationBest.distance = std::numeric_limits<double>::max();

            // Each ant constructs a solution with adaptive probability
            for (int antIdx = 0; antIdx < numAnts; antIdx++) {
                // Construct solution
                ants[antIdx] = constructSolution();

                // Local search (2-opt) on ant's solution with adaptive probability
                std::uniform_real_distribution<double> prob01(0.0, 1.0);
                double current_ls_prob = localSearchProbability;
                
                // Reduce local search probability in early iterations to maintain diversity
                if (iter < maxIterations * 0.2) {
                    current_ls_prob *= 0.7;  // Less local search in early iterations
                }
                
                // Apply local search with probability
                if (prob01(rng) < current_ls_prob) {
                    twoOptImprove(ants[antIdx]);
                }

                // Update iteration best and global best solutions
                updateIterationBest(ants[antIdx]);
                updateBestSolution(ants[antIdx]);
            }

            // Calculate population diversity for convergence detection
            double currentDiversity = calculatePopulationDiversity();

            // Pheromone update (MAX-MIN approach)
            globalPheromoneUpdate();

            // Scheduled immigrant injection (diversity maintenance)
            if (immigrantInterval > 0 && iter % immigrantInterval == 0) {
                int immigrantCount = std::max(1, (int)std::round(numAnts * 0.1)); // Replace 10%
                int protectCount = std::max(eliteAnts, numAnts / 5); // keep top 20% safe
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
            if (currentDiversity < 0.03) { // If diversity drops too low
                int injectCount = numAnts / 3; // Replace worst 33%
                int startIdx = numAnts - injectCount;
                for (int i = 0; i < injectCount; i++) {
                    int idx = startIdx + i;
                    ants[idx] = constructSolution();
                }
                diversityInjections++;
            }

            // Check for stagnation (adaptive response)
            if (stats.delta < 1e-9) {
                stagnationCounter++;
            } else {
                stagnationCounter = 0; // Reset when improvement occurs
            }

            // Adaptive restart strategy for large problems
            if (stagnationCounter >= restartThreshold) {
                // Full restart with reinitialization of pheromone
                resetPheromone();
                
                // Keep the best solution found so far
                // Replace all but the best solution with new random solutions
                std::sort(ants.begin(), ants.end());
                for (int i = 1; i < numAnts; i++) {
                    ants[i] = constructSolution();
                }
                
                currentDiversity = calculatePopulationDiversity();
                stagnationCounter = 0;
                restarts++;
                
                if (verbose) {
                    std::cout << "Restart " << restarts << " at iteration " << iter << std::endl;
                }
            }

            prevBest = bestSolution.distance;

            // Print progress
            if (verbose && (iter % 200 == 0 || iter == 1)) {
                printStats(stats);
                if (tsp.optimalTourLength > 0) {
                    double gap = tsp.getGapFromOptimal(bestSolution.distance);
                    std::cout << "  Gap: " << gap << "% | Restarts: " << restarts
                              << " | Injections: " << diversityInjections
                              << " | Diversity: " << currentDiversity << std::endl;
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

        // Final local search on best solution
        if (verbose) std::cout << "\nApplying final local search..." << std::endl;
        
        Ant finalBest = bestSolution;
        LinKernighan lk(tsp);
        finalBest.tour = lk.optimizeFull(finalBest.tour);
        finalBest.distance = tsp.calculateTourLength(finalBest.tour);
        finalBest.fitness = 1.0 / finalBest.distance;

        // Update bestSolution if LK improved the solution
        if (finalBest.distance > 1e-6 && finalBest.distance < bestSolution.distance) {
            bestSolution = finalBest;
        }

        if (verbose) {
            std::cout << "\n=== OPTIMIZED ACO Complete ===" << std::endl;
            std::cout << "Best tour distance: " << bestSolution.distance << std::endl;
            if (tsp.optimalTourLength > 0) {
                double gap = tsp.getGapFromOptimal(bestSolution.distance);
                std::cout << "Gap from optimal: " << gap << "%" << std::endl;
                std::cout << "Restarts: " << restarts << " | Diversity injections: " << diversityInjections << std::endl;
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
};

#endif // ANT_COLONY_OPTIMIZED_H