#ifndef ANT_COLONY_SMART_H
#define ANT_COLONY_SMART_H

#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <limits>
#include <cmath>
#include <queue>
#include <numeric>
#include "../TSPInstance.h"
#include "../LinKernighan.h"

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
 * @brief Smart Ant Colony Optimization Algorithm with Hill-Climbing and Simulated Annealing elements
 * Maintains 2000 iterations and 200 ants as required, but uses smarter pheromone updating
 */
class AntColonySmart {
private:
    // Problem instance
    const TSPInstance& tsp;

    // Fixed parameters as per requirements
    int numAnts;                    // Fixed at 200
    int maxIterations;              // Fixed at 2000
    double alpha;                   // Pheromone importance
    double beta;                    // Heuristic importance
    double rho;                     // Pheromone evaporation rate
    int eliteAnts;                 // Number of elitist ants
    double initialTemperature;      // For simulated annealing component
    double coolingRate;             // Cooling schedule

    // Adaptive mechanisms within fixed constraints
    double localSearchProbability;  // Probability to apply local search
    double tau_min, tau_max;        // Pheromone bounds (MMAS approach)
    std::vector<double> temperatureSchedule; // Temperature for each iteration

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
     * @brief Constructor with parameters as required
     */
    AntColonySmart(const TSPInstance& instance)
        : tsp(instance),
          numAnts(200),             // Required: 200 ants
          maxIterations(2000),      // Required: 2000 iterations
          alpha(1.0),
          beta(2.5),                // Slightly increased for better heuristic guidance
          rho(0.02),                // Reduced for slower convergence
          eliteAnts(5),             // Increased for better guidance
          initialTemperature(1000.0), // High initial temperature for exploration
          coolingRate(0.995),       // Slow cooling
          localSearchProbability(0.3) {

        rng.seed(static_cast<unsigned>(time(nullptr)));
        bestSolution.distance = std::numeric_limits<double>::max();
        iterationBest.distance = std::numeric_limits<double>::max();

        // Initialize matrices
        pheromone.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        eta.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        eta_beta.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));
        transition_prob.resize(tsp.dimension, std::vector<double>(tsp.dimension, 0.0));

        // Calculate tau_max and tau_min based on problem characteristics (MMAS approach)
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

        // Calculate pheromone bounds for MMAS
        double rough_tour_length = tsp.dimension * (maxDist + minDist) / 2.0;
        tau_max = 1.0 / (rho * rough_tour_length);
        tau_min = tau_max / (2.0 * tsp.dimension);

        // Initialize pheromone matrix
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                if (i != j) {
                    pheromone[i][j] = tau_max; // Start with max pheromone for exploration
                }
            }
        }

        // Create temperature schedule
        temperatureSchedule.resize(maxIterations);
        for (int i = 0; i < maxIterations; i++) {
            temperatureSchedule[i] = initialTemperature * std::pow(coolingRate, i);
        }
    }

    // ==================== SETTERS FOR PARAMETERS ====================

    void setAlpha(double a) { alpha = a; }
    void setBeta(double b) { beta = b; }
    void setRho(double r) { rho = r; }
    void setEliteAnts(int e) { eliteAnts = e; }
    void setLocalSearchProbability(double p) { localSearchProbability = p; }

    // Reseed method for unique random seed per run
    void reseed(unsigned seed) { rng.seed(seed); }

    // ==================== SOLUTION CONSTRUCTION ====================

    /**
     * @brief Construct a solution for an ant using probability-based selection
     * Uses the best solution as a reference to generate new solutions (hill-climber approach)
     */
    Ant constructSolution() {
        Ant ant(tsp.dimension);
        
        // Sometimes start with completely random construction, sometimes with guided construction
        std::uniform_real_distribution<double> prob01(0.0, 1.0);
        bool useGuidedConstruction = prob01(rng) > 0.3; // 70% of the time use guided, 30% random
        
        if (useGuidedConstruction && bestSolution.distance < std::numeric_limits<double>::max()) {
            // Use bestSolution as a basis for new solution (hill-climber approach)
            ant.tour = bestSolution.tour;
            
            // Apply small random perturbations to the best solution
            int perturbationSteps = std::max(2, tsp.dimension / 50); // 2-20 swaps depending on size
            for (int p = 0; p < perturbationSteps; p++) {
                std::uniform_int_distribution<int> posDist(0, tsp.dimension - 1);
                int pos1 = posDist(rng);
                int pos2 = posDist(rng);
                std::swap(ant.tour[pos1], ant.tour[pos2]);
            }
        } else {
            // Standard ACO construction but improved
            std::vector<bool> visited(tsp.dimension, false);
            std::uniform_int_distribution<int> startDist(0, tsp.dimension - 1);

            // Start from a random city
            int currentCity = startDist(rng);
            ant.tour[0] = currentCity;
            visited[currentCity] = true;

            // Build the rest of the tour using ACO probability rules
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
                    // If all probabilities are effectively zero, choose nearest unvisited city
                    int nearestCity = -1;
                    double maxEta = 0.0;
                    for (int j = 0; j < tsp.dimension; j++) {
                        if (!visited[j] && eta[currentCity][j] > maxEta) {
                            maxEta = eta[currentCity][j];
                            nearestCity = j;
                        }
                    }
                    nextCity = (nearestCity != -1) ? nearestCity : 0;
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
        if (ant.distance < bestSolution.distance) {
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

    // ==================== SMART PHEROMONE UPDATE ====================

    /**
     * @brief Smart pheromone update using best solution as foundation (hill-climber approach)
     */
    void smartPheromoneUpdate() {
        // Evaporate pheromone
        for (int i = 0; i < tsp.dimension; i++) {
            for (int j = 0; j < tsp.dimension; j++) {
                pheromone[i][j] *= (1.0 - rho);
                // Ensure pheromone stays within bounds
                pheromone[i][j] = std::min(tau_max, std::max(pheromone[i][j], tau_min));
            }
        }

        // Deposit pheromone for best solution found so far (MAX-MIN approach)
        for (int i = 0; i < tsp.dimension; i++) {
            int from = bestSolution.tour[i];
            int to = bestSolution.tour[(i + 1) % tsp.dimension];
            pheromone[from][to] += 1.0 / bestSolution.distance;
            // Ensure bounds
            pheromone[from][to] = std::min(tau_max, pheromone[from][to]);
            pheromone[to][from] = pheromone[from][to]; // Symmetric TSP
        }

        // Elitist strategy: add extra pheromone for top solutions
        std::vector<Ant> sortedAnts = ants;
        std::sort(sortedAnts.begin(), sortedAnts.end());

        for (int e = 0; e < eliteAnts && e < sortedAnts.size(); e++) {
            double elite_pheromone = 1.0 / sortedAnts[e].distance;
            for (int i = 0; i < tsp.dimension; i++) {
                int from = sortedAnts[e].tour[i];
                int to = sortedAnts[e].tour[(i + 1) % tsp.dimension];
                pheromone[from][to] += elite_pheromone; // Additional pheromone
                // Ensure bounds
                pheromone[from][to] = std::min(tau_max, pheromone[from][to]);
                pheromone[to][from] = pheromone[from][to];
            }
        }
    }

    /**
     * @brief Apply Simulated Annealing to potentially accept worse solutions (escape local minima)
     */
    bool acceptWithSA(double currentDistance, double candidateDistance, double temperature) {
        if (candidateDistance < currentDistance) {
            return true; // Always accept better solutions
        }
        
        // Accept worse solution with probability based on temperature
        double delta = candidateDistance - currentDistance;
        double probability = std::exp(-delta / temperature);
        std::uniform_real_distribution<double> prob01(0.0, 1.0);
        return prob01(rng) < probability;
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

    // ==================== MAIN ACO LOOP (SMART) ====================

    /**
     * @brief Run the Smart Ant Colony Optimization algorithm with hill-climbing approach
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

        // Statistics
        double prevBest = bestSolution.distance;
        int stagnationCounter = 0;

        if (verbose) {
            std::cout << "\n=== Starting SMART ACO Evolution (Hill-Climber + SA + MMAS) ===" << std::endl;
            std::cout << "Ants: " << numAnts
                      << " | Max iterations: " << maxIterations
                      << " | Alpha: " << alpha
                      << " | Beta: " << beta
                      << " | Rho: " << rho << std::endl;
            std::cout << "| Tau_min: " << tau_min << " | Tau_max: " << tau_max << std::endl;
        }

        // Main ACO loop - exactly 2000 iterations as required
        for (int iter = 1; iter <= maxIterations; iter++) {
            // Reset iteration best
            iterationBest.distance = std::numeric_limits<double>::max();

            // Each ant constructs a solution based on the best solution (hill-climber approach)
            for (int antIdx = 0; antIdx < numAnts; antIdx++) {
                // Construct solution based on best solution so far
                ants[antIdx] = constructSolution();

                // Apply local search with probability
                std::uniform_real_distribution<double> prob01(0.0, 1.0);
                if (prob01(rng) < localSearchProbability) {
                    twoOptImprove(ants[antIdx]);
                }

                // Update iteration best and global best solutions
                updateIterationBest(ants[antIdx]);
                updateBestSolution(ants[antIdx]);
            }

            // Smart pheromone update (hill-climber approach with MMAS)
            smartPheromoneUpdate();

            // Calculate and store statistics
            IterationStats stats = calculateStats(iter, prevBest);
            statsHistory.push_back(stats);

            // SA escape mechanism: if no improvement in recent iterations, accept some worse solutions to escape local minima
            if (stats.delta < 1e-9) {
                stagnationCounter++;
                
                if (stagnationCounter > 50) {  // If stuck for 50 iterations
                    double currentTemp = temperatureSchedule[std::min(iter-1, (int)temperatureSchedule.size()-1)];
                    
                    // Try to accept some worse solutions to escape local minima
                    for (int i = 0; i < 5; i++) {  // Try with 5 random ants
                        std::uniform_int_distribution<int> antDist(0, numAnts - 1);
                        int randAntIdx = antDist(rng);
                        
                        // Generate a perturbed version of the current best
                        Ant perturbed = bestSolution;
                        // Apply perturbation
                        std::uniform_int_distribution<int> posDist(0, tsp.dimension - 1);
                        int pos1 = posDist(rng);
                        int pos2 = posDist(rng);
                        std::swap(perturbed.tour[pos1], perturbed.tour[pos2]);
                        evaluateAnt(perturbed);
                        
                        if (acceptWithSA(bestSolution.distance, perturbed.distance, currentTemp)) {
                            ants[randAntIdx] = perturbed;
                            // If this new solution is better than global best, update it
                            if (perturbed.distance < bestSolution.distance) {
                                bestSolution = perturbed;
                            }
                        }
                    }
                }
            } else {
                stagnationCounter = 0; // Reset counter when improvement occurs
            }

            prevBest = bestSolution.distance;

            // Print progress in summary intervals
            if (verbose && (iter == 1 || iter == 500 || iter == 1000 || iter == 1500 || iter == maxIterations)) {
                printStats(stats);
                if (tsp.optimalTourLength > 0) {
                    double gap = tsp.getGapFromOptimal(bestSolution.distance);
                    std::cout << "  Gap: " << gap << "% | Stagnation: " << stagnationCounter << std::endl;
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
        if (finalBest.distance < bestSolution.distance) {
            bestSolution = finalBest;
        }

        if (verbose) {
            std::cout << "\n=== SMART ACO Complete ===" << std::endl;
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
};

#endif // ANT_COLONY_SMART_H