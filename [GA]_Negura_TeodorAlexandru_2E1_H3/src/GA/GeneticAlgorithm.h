#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <limits>
#include "TSPInstance.h"
#include "LinKernighan.h"

/**
 * @brief Represents an individual (solution) in the GA population
 */
struct Individual {
    std::vector<int> tour;  // Permutation of cities (Hamiltonian cycle)
    double distance;        // Total tour length
    double fitness;         // Fitness value (1.0 / distance)

    Individual() : distance(0.0), fitness(0.0) {}

    Individual(int numCities) : distance(0.0), fitness(0.0) {
        tour.resize(numCities);
    }

    // Comparison operator for sorting (higher fitness = better)
    bool operator<(const Individual& other) const {
        return fitness > other.fitness;  // Descending order (best first)
    }
};

/**
 * @brief Genetic Algorithm for solving TSP
 */
class GeneticAlgorithm {
private:
    // Problem instance
    const TSPInstance& tsp;

    // GA Parameters
    int populationSize;
    int offspringCount;
    int maxGenerations;
    double crossoverRate;
    double mutationRate;
    int tournamentSize;
    int eliteCount;

    // Diversity controls
    double localSearchProbability;   // Probability to apply local search on a child
    double immigrantRate;            // Fraction of population replaced on schedule
    int immigrantInterval;           // How often (generations) to inject immigrants
    int restartThreshold;            // Generations without improvement before restart
    double restartMutationRate;      // Mutation rate used immediately after restart
    int localSearchStartGen;         // Burn-in generations before applying LS/LK
    bool forceFullGenerations;       // If true, never stop early before maxGenerations

    // Early termination controls
    int noImprovementLimit;          // Stop if no improvement for this many generations
    double targetGap;                // Stop if gap from optimal is below this threshold
    bool enableEarlyTermination;     // Whether to allow early termination

    // Population
    std::vector<Individual> population;

    // Best solution found
    Individual bestSolution;

    // Random number generator
    std::mt19937 rng;

    // Statistics tracking
    std::vector<double> fitnessHistory;

public:
    /**
     * @brief Constructor with default parameters
     * Population: 200 (at limit after selection)
     * Offspring: 200 (100% of population = 50% elitism)
     * Total during evolution: 400 → select best 200 (50% elitism)
     * Protected elite: top 20% never replaced during diversity injection
     */
    GeneticAlgorithm(const TSPInstance& instance)
        : tsp(instance),
          populationSize(200),   // Maximum allowed after selection
          offspringCount(200),   // 100% = 50% elitism (200+200=400 → keep 200)
          maxGenerations(2000),  // Assignment limit
          crossoverRate(0.9),
                      mutationRate(0.25),    // Higher base mutation to fight convergence
                      tournamentSize(2),     // Lower selection pressure
                      eliteCount(3),         // ~1.5% for population 200
                      localSearchProbability(0.25),
                      immigrantRate(0.10),   // More frequent immigrants
                      immigrantInterval(15),
                      restartThreshold(40),
                      restartMutationRate(0.6),
                      localSearchStartGen(200),
                      forceFullGenerations(true) {        // Keep top few guaranteed

        rng.seed(static_cast<unsigned>(time(nullptr)));
        bestSolution.distance = std::numeric_limits<double>::max();

        // Initialize early termination parameters
        noImprovementLimit = 300;      // Stop after 300 generations without improvement
        targetGap = 2.0;              // Stop if within 2% of optimal (if known)
        enableEarlyTermination = true; // Enable early termination by default
    }

    // ==================== SETTERS FOR PARAMETERS ====================

    void setPopulationSize(int size) { populationSize = size; }
    void setOffspringCount(int count) { offspringCount = count; }
    void setMaxGenerations(int gens) { maxGenerations = gens; }
    void setCrossoverRate(double rate) { crossoverRate = rate; }
    void setMutationRate(double rate) { mutationRate = rate; }
    void setTournamentSize(int size) { tournamentSize = size; }
    void setEliteCount(int count) { eliteCount = count; }
    void setLocalSearchProbability(double p) { localSearchProbability = p; }
    void setImmigrantRate(double r) { immigrantRate = r; }
    void setImmigrantInterval(int interval) { immigrantInterval = interval; }
    void setRestartThreshold(int threshold) { restartThreshold = threshold; }
    void setLocalSearchStartGen(int start) { localSearchStartGen = start; }
    void setForceFullGenerations(bool force) { forceFullGenerations = force; }

    // Early termination setters
    void setNoImprovementLimit(int limit) { noImprovementLimit = limit; }
    void setTargetGap(double gap) { targetGap = gap; }
    void setEnableEarlyTermination(bool enable) { enableEarlyTermination = enable; }

    // Reseed method for unique random seed per run
    void reseed(unsigned seed) { rng.seed(seed); }

    // ==================== POPULATION INITIALIZATION ====================

    /**
     * @brief Generate a random individual (random permutation)
     */
    Individual generateRandomIndividual() {
        Individual ind(tsp.dimension);

        // Fill with 0, 1, 2, ..., n-1
        for (int i = 0; i < tsp.dimension; i++) {
            ind.tour[i] = i;
        }

        // Fisher-Yates shuffle
        for (int i = tsp.dimension - 1; i > 0; i--) {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(rng);
            std::swap(ind.tour[i], ind.tour[j]);
        }

        // Calculate fitness
        evaluateIndividual(ind);

        return ind;
    }

    /**
     * @brief Initialize population with random individuals
     */
    void initializePopulation() {
        population.clear();
        population.reserve(populationSize);

        for (int i = 0; i < populationSize; i++) {
            population.push_back(generateRandomIndividual());
        }

        // Track best solution
        updateBestSolution();

        std::cout << "Population initialized with " << populationSize << " individuals" << std::endl;
        std::cout << "Initial best distance: " << bestSolution.distance << std::endl;
    }

    // ==================== FITNESS EVALUATION ====================

    /**
     * @brief Evaluate fitness of an individual
     */
    void evaluateIndividual(Individual& ind) {
        ind.distance = tsp.calculateTourLength(ind.tour);
        ind.fitness = 1.0 / ind.distance;
    }

    /**
     * @brief Update best solution if better found
     */
    void updateBestSolution() {
        for (const auto& ind : population) {
            if (ind.distance > 1e-6 && ind.distance < bestSolution.distance) {
                bestSolution = ind;
            }
        }
    }

    // ==================== GETTERS ====================

    const Individual& getBestSolution() const { return bestSolution; }
    const std::vector<double>& getFitnessHistory() const { return fitnessHistory; }
    int getPopulationSize() const { return populationSize; }

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

    // ==================== SELECTION OPERATORS ====================

    /**
     * @brief Tournament Selection - select one parent
     * Randomly picks k individuals, returns the best one
     * @return Index of selected individual
     */
    int tournamentSelection() {
        std::uniform_int_distribution<int> dist(0, population.size() - 1);

        int bestIndex = dist(rng);
        double bestFitness = population[bestIndex].fitness;

        for (int i = 1; i < tournamentSize; i++) {
            int candidateIndex = dist(rng);
            if (population[candidateIndex].fitness > bestFitness) {
                bestIndex = candidateIndex;
                bestFitness = population[candidateIndex].fitness;
            }
        }

        return bestIndex;
    }

    /**
     * @brief Select two parents using tournament selection
     * @return Pair of parent indices
     */
    std::pair<int, int> selectParents() {
        int parent1 = tournamentSelection();
        int parent2 = tournamentSelection();

        // Ensure different parents (optional but recommended)
        int attempts = 0;
        while (parent2 == parent1 && attempts < 10) {
            parent2 = tournamentSelection();
            attempts++;
        }

        return {parent1, parent2};
    }

    /**
     * @brief Survivor Selection - keep the best individuals
     * After adding offspring, reduce population back to populationSize
     * Uses elitism: always keeps the best eliteCount individuals
     */
    void survivorSelection() {
        // Sort population by fitness (descending - best first)
        std::sort(population.begin(), population.end());

        // Keep only the best populationSize individuals
        if (population.size() > static_cast<size_t>(populationSize)) {
            population.resize(populationSize);
        }

        // Update best solution
        updateBestSolution();
    }

    /**
     * @brief Add offspring to population (before survivor selection)
     */
    void addOffspring(const std::vector<Individual>& offspring) {
        for (const auto& child : offspring) {
            population.push_back(child);
        }
    }

    // ==================== CROSSOVER OPERATORS ====================

    /**
     * @brief Get the next city in a tour after a given city
     */
    int getNextCity(const std::vector<int>& tour, int city) {
        for (size_t i = 0; i < tour.size(); i++) {
            if (tour[i] == city) {
                return tour[(i + 1) % tour.size()];
            }
        }
        return -1;
    }

    /**
     * @brief Greedy Edge Crossover - Custom crossover
     * At each step, choose the shorter edge from either parent
     * Creates the "best" child by combining good edges from both parents
     */
    Individual greedyEdgeCrossover(const Individual& parent1, const Individual& parent2) {
        int n = tsp.dimension;
        Individual child(n);
        std::vector<bool> visited(n, false);

        // Start from a random city
        std::uniform_int_distribution<int> startDist(0, n - 1);
        int currentCity = startDist(rng);
        child.tour[0] = currentCity;
        visited[currentCity] = true;

        // Build the rest of the tour
        for (int i = 1; i < n; i++) {
            // Get next city suggestions from both parents
            int next1 = getNextCity(parent1.tour, currentCity);
            int next2 = getNextCity(parent2.tour, currentCity);

            int nextCity = -1;

            // Check if suggestions are valid (not visited)
            bool valid1 = (next1 != -1 && !visited[next1]);
            bool valid2 = (next2 != -1 && !visited[next2]);

            if (valid1 && valid2) {
                // Both valid: choose the shorter edge
                double dist1 = tsp.getDistance(currentCity, next1);
                double dist2 = tsp.getDistance(currentCity, next2);
                nextCity = (dist1 <= dist2) ? next1 : next2;
            } else if (valid1) {
                nextCity = next1;
            } else if (valid2) {
                nextCity = next2;
            } else {
                // Neither valid: find nearest unvisited city
                double minDist = std::numeric_limits<double>::max();
                for (int j = 0; j < n; j++) {
                    if (!visited[j]) {
                        double d = tsp.getDistance(currentCity, j);
                        if (d < minDist) {
                            minDist = d;
                            nextCity = j;
                        }
                    }
                }
            }

            child.tour[i] = nextCity;
            visited[nextCity] = true;
            currentCity = nextCity;
        }

        // Evaluate the child
        evaluateIndividual(child);
        return child;
    }

    /**
     * @brief Order Crossover (OX) - Standard crossover for TSP
     * Copies a segment from parent1, fills rest from parent2 in order
     */
    Individual orderCrossover(const Individual& parent1, const Individual& parent2) {
        int n = tsp.dimension;
        Individual child(n);
        std::fill(child.tour.begin(), child.tour.end(), -1);
        std::vector<bool> inChild(n, false);

        // Select random segment [start, end]
        std::uniform_int_distribution<int> dist(0, n - 1);
        int start = dist(rng);
        int end = dist(rng);
        if (start > end) std::swap(start, end);

        // Copy segment from parent1
        for (int i = start; i <= end; i++) {
            child.tour[i] = parent1.tour[i];
            inChild[parent1.tour[i]] = true;
        }

        // Fill remaining positions from parent2 (in order, skipping already included)
        int childPos = (end + 1) % n;
        for (int i = 0; i < n; i++) {
            int parent2Pos = (end + 1 + i) % n;
            int city = parent2.tour[parent2Pos];

            if (!inChild[city]) {
                child.tour[childPos] = city;
                inChild[city] = true;
                childPos = (childPos + 1) % n;
            }
        }

        evaluateIndividual(child);
        return child;
    }

    /**
     * @brief Perform crossover between two parents
     * @param useGreedy If true, use Greedy Edge Crossover; otherwise use OX
     */
    Individual crossover(const Individual& parent1, const Individual& parent2, bool useGreedy = true) {
        if (useGreedy) {
            return greedyEdgeCrossover(parent1, parent2);
        } else {
            return orderCrossover(parent1, parent2);
        }
    }

    /**
     * @brief Mixed crossover: randomly choose among greedy edge, OX, and random mix
     */
    Individual mixedCrossover(const Individual& parent1, const Individual& parent2) {
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        double r = prob(rng);
        if (r < 0.5) {
            return greedyEdgeCrossover(parent1, parent2);
        } else if (r < 0.8) {
            return orderCrossover(parent1, parent2);
        }
        return randomMixCrossover(parent1, parent2);
    }

    // ==================== MUTATION OPERATORS ====================

    /**
     * @brief Inversion Mutation - reverse a random segment
     * This is like a 2-opt move, good for TSP
     */
    void inversionMutation(Individual& ind) {
        int n = ind.tour.size();
        std::uniform_int_distribution<int> dist(0, n - 1);

        int i = dist(rng);
        int j = dist(rng);
        if (i > j) std::swap(i, j);

        // Reverse segment [i, j]
        while (i < j) {
            std::swap(ind.tour[i], ind.tour[j]);
            i++;
            j--;
        }

        // Re-evaluate
        evaluateIndividual(ind);
    }

    /**
     * @brief Swap Mutation - exchange two random cities
     */
    void swapMutation(Individual& ind) {
        int n = ind.tour.size();
        std::uniform_int_distribution<int> dist(0, n - 1);

        int i = dist(rng);
        int j = dist(rng);
        while (j == i) j = dist(rng);

        std::swap(ind.tour[i], ind.tour[j]);
        evaluateIndividual(ind);
    }

    /**
     * @brief Random Mix Crossover - your custom idea!
     * For each position, randomly pick from parent A or B (50% each)
     * Then fix duplicates to ensure valid Hamiltonian cycle
     */
    Individual randomMixCrossover(const Individual& parent1, const Individual& parent2) {
        int n = tsp.dimension;
        Individual child(n);
        std::vector<bool> used(n, false);
        std::vector<int> missing;
        std::uniform_real_distribution<double> prob(0.0, 1.0);

        // Step 1: For each position, try to pick from A or B (50% chance)
        for (int i = 0; i < n; i++) {
            int cityA = parent1.tour[i];
            int cityB = parent2.tour[i];

            int chosen = -1;
            if (prob(rng) < 0.5) {
                // Try A first, then B
                if (!used[cityA]) {
                    chosen = cityA;
                } else if (!used[cityB]) {
                    chosen = cityB;
                }
            } else {
                // Try B first, then A
                if (!used[cityB]) {
                    chosen = cityB;
                } else if (!used[cityA]) {
                    chosen = cityA;
                }
            }

            if (chosen != -1) {
                child.tour[i] = chosen;
                used[chosen] = true;
            } else {
                child.tour[i] = -1;  // Mark for later
            }
        }

        // Step 2: Find missing cities
        for (int c = 0; c < n; c++) {
            if (!used[c]) {
                missing.push_back(c);
            }
        }

        // Shuffle missing cities for randomness
        for (int i = missing.size() - 1; i > 0; i--) {
            std::uniform_int_distribution<int> d(0, i);
            std::swap(missing[i], missing[d(rng)]);
        }

        // Step 3: Fill empty positions with missing cities
        int missingIdx = 0;
        for (int i = 0; i < n; i++) {
            if (child.tour[i] == -1) {
                child.tour[i] = missing[missingIdx++];
            }
        }

        evaluateIndividual(child);
        return child;
    }

    /**
     * @brief Apply mutation to an individual with given probability
     */
    void mutate(Individual& ind, double rate) {
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        if (prob(rng) < rate) {
            if (prob(rng) < 0.5) {
                inversionMutation(ind);
            } else {
                swapMutation(ind);
            }
        }
    }

    // ==================== 2-OPT LOCAL SEARCH ====================

    /**
     * @brief 2-opt improvement move
     * Tries reversing all segments, keeps improvement if found
     * @return true if improvement was made
     */
    bool twoOptImprove(Individual& ind) {
        int n = ind.tour.size();
        bool improved = false;

        for (int i = 0; i < n - 1 && !improved; i++) {
            for (int j = i + 2; j < n; j++) {
                // Calculate delta of reversing segment [i+1, j]
                int a = ind.tour[i];
                int b = ind.tour[i + 1];
                int c = ind.tour[j];
                int d = ind.tour[(j + 1) % n];

                double oldDist = tsp.getDistance(a, b) + tsp.getDistance(c, d);
                double newDist = tsp.getDistance(a, c) + tsp.getDistance(b, d);

                if (newDist < oldDist - 1e-9) {
                    // Reverse segment [i+1, j]
                    int left = i + 1;
                    int right = j;
                    while (left < right) {
                        std::swap(ind.tour[left], ind.tour[right]);
                        left++;
                        right--;
                    }
                    ind.distance = ind.distance - oldDist + newDist;
                    ind.fitness = 1.0 / ind.distance;
                    improved = true;
                    break;
                }
            }
        }
        return improved;
    }

    /**
     * @brief Apply 2-opt until no improvement (local optimum)
     */
    void twoOptFull(Individual& ind) {
        bool improved = true;
        int maxIterations = 100;  // Limit iterations to avoid too much time
        int iter = 0;
        while (improved && iter < maxIterations) {
            improved = twoOptImprove(ind);
            iter++;
        }
    }

    /**
     * @brief Apply Lin-Kernighan optimization to an individual
     * Uses the full LK algorithm with candidate lists, positive gain criterion,
     * don't-look bits, and 2-opt fallback for comprehensive optimization
     */
    void linKernighanOptimize(Individual& ind) {
        LinKernighan lk(tsp);
        ind.tour = lk.optimizeFull(ind.tour);
        // Always recalculate distance from TSP instance to ensure correctness
        ind.distance = tsp.calculateTourLength(ind.tour);
        ind.fitness = 1.0 / ind.distance;
    }

    /**
     * @brief Apply LK to top K individuals
     * Used periodically to escape deep local optima
     */
    void applyLKToTop(int k) {
        k = std::min(k, (int)population.size());
        for (int i = 0; i < k; i++) {
            linKernighanOptimize(population[i]);
        }
        // Re-sort after LK optimization
        std::sort(population.begin(), population.end());
        updateBestSolution();
    }

    // ==================== STATISTICS ====================

    struct GenerationStats {
        double bestDistance;
        double worstDistance;
        double meanDistance;
        double stdDevDistance;
        double delta;  // improvement from previous generation
        int generation;
    };

    std::vector<GenerationStats> statsHistory;

    GenerationStats calculateStats(int generation, double prevBest) {
        GenerationStats stats;
        stats.generation = generation;

        double sum = 0.0;
        double sumSq = 0.0;
        stats.bestDistance = std::numeric_limits<double>::max();
        stats.worstDistance = 0.0;

        for (const auto& ind : population) {
            sum += ind.distance;
            sumSq += ind.distance * ind.distance;
            if (ind.distance < stats.bestDistance) {
                stats.bestDistance = ind.distance;
            }
            if (ind.distance > stats.worstDistance) {
                stats.worstDistance = ind.distance;
            }
        }

        int n = population.size();
        stats.meanDistance = sum / n;
        double variance = (sumSq / n) - (stats.meanDistance * stats.meanDistance);
        stats.stdDevDistance = std::sqrt(std::max(0.0, variance));
        stats.delta = prevBest - stats.bestDistance;  // positive = improvement

        return stats;
    }

    const std::vector<GenerationStats>& getStatsHistory() const {
        return statsHistory;
    }

    void printStats(const GenerationStats& stats) {
        std::cout << "Gen " << stats.generation
                  << " | Best: " << stats.bestDistance
                  << " | Mean: " << stats.meanDistance
                  << " | Worst: " << stats.worstDistance
                  << " | StdDev: " << stats.stdDevDistance
                  << " | Delta: " << stats.delta << std::endl;
    }

    // ==================== MAIN EVOLUTION LOOP ====================

    /**
     * @brief Run the genetic algorithm
     * @param verbose If true, print progress
     * @return Best solution found
     */
    Individual run(bool verbose = true) {
        // Initialize population
        initializePopulation();

        // Statistics
        double prevBest = bestSolution.distance;
        int stagnationCounter = 0;
        int diversityInjections = 0;  // Count how many times we injected diversity
        double currentMutationRate = std::max(mutationRate, 0.25);
        const int stagnationThreshold = 10;   // AGGRESSIVE: only 10 generations without improvement
        const double maxMutationRate = 0.9;   // Higher max mutation when stuck
        const double minMutationRate = 0.10;  // Higher floor to keep noise alive

        if (verbose) {
            std::cout << "\n=== Starting GA Evolution ===" << std::endl;
            std::cout << "Population: " << populationSize
                      << " | Offspring: " << offspringCount
                      << " | Elitism: " << (100 * populationSize) / (populationSize + offspringCount) << "%"
                      << " | Max generations: " << maxGenerations << std::endl;
        }

        // LK scheduling: apply to a few best individuals periodically
        // Adjust frequency based on problem size for efficiency
        const int lkInterval = (tsp.dimension > 1000) ? std::max(200, tsp.dimension / 2) : std::max(100, tsp.dimension); // fewer calls for very large instances
        const int lkTopCount = (tsp.dimension > 1000) ? 1 : 3; // apply to fewer individuals for very large instances

        // Main evolution loop
        for (int gen = 1; gen <= maxGenerations; gen++) {
            // Shuffle to reduce positional bias before selection
            std::shuffle(population.begin(), population.end(), rng);

            std::vector<Individual> offspring;
            offspring.reserve(offspringCount);

            std::uniform_real_distribution<double> prob01(0.0, 1.0);

            // Create offspring
            for (int i = 0; i < offspringCount; i++) {
                // Select parents
                auto [p1Idx, p2Idx] = selectParents();

                // Crossover
                Individual child = mixedCrossover(population[p1Idx], population[p2Idx]);

                // Mutation (with adaptive rate)
                mutate(child, currentMutationRate);

                // Local search (2-opt) on child
                // Delay local search until burn-in to avoid early homogenization
                // Adjust probability based on problem size and generation
                double lsProb = 0.0;
                if (gen >= localSearchStartGen) {
                    // Reduce local search probability for larger instances to save time
                    lsProb = (tsp.dimension > 500) ? localSearchProbability * 0.5 : localSearchProbability;
                    // Further reduce if we're in the later stages to save time
                    if (gen > maxGenerations * 0.8) {
                        lsProb *= 0.7; // Reduce in final 20% of generations
                    }
                }
                if (prob01(rng) < lsProb) {
                    twoOptImprove(child);  // Just one pass to save time
                }

                offspring.push_back(child);
            }

            // GENERATIONAL REPLACEMENT WITH MERGE + ELITISM
            std::sort(population.begin(), population.end());
            int effectiveElite = eliteCount > 0 ? eliteCount : std::max(1, populationSize / 50);

            std::vector<Individual> combined;
            combined.reserve(population.size() + offspring.size());
            combined.insert(combined.end(), population.begin(), population.end());
            combined.insert(combined.end(), offspring.begin(), offspring.end());

            std::sort(combined.begin(), combined.end());

            // Scheduled immigrant injection before final truncation
            if (immigrantInterval > 0 && gen % immigrantInterval == 0) {
                int immigrantCount = std::max(1, (int)std::round(populationSize * immigrantRate));
                int protectCount = std::max(effectiveElite, populationSize / 10); // keep top 10% safe
                for (int i = 0; i < immigrantCount; i++) {
                    int replaceIdx = (int)combined.size() - 1 - i;
                    if (replaceIdx <= protectCount) break;
                    combined[replaceIdx] = generateRandomIndividual();
                }
                std::sort(combined.begin(), combined.end());
                diversityInjections++;
            }

            // Truncate to population size while keeping elites
            population.assign(combined.begin(), combined.begin() + std::min((int)combined.size(), populationSize));

            // Update best solution
            updateBestSolution();

            // Periodic LK polish on elites to escape local minima
            if (lkInterval > 0 && gen >= localSearchStartGen && gen % lkInterval == 0) {
                applyLKToTop(lkTopCount);
            }

            // Calculate and store statistics
            GenerationStats stats = calculateStats(gen, prevBest);
            statsHistory.push_back(stats);

            // Additional diversity injection when variance collapses
            if (stats.stdDevDistance < 1e-3) {
                int injectCount = populationSize / 4; // replace worst 25%
                int startIdx = populationSize - injectCount;
                for (int i = 0; i < injectCount; i++) {
                    int idx = startIdx + i;
                    population[idx] = generateRandomIndividual();
                }
                currentMutationRate = std::max(currentMutationRate, 0.5);
                stagnationCounter = 0;
                diversityInjections++;
            }

            // Check for stagnation (adaptive mutation + diversity injection)
            if (stats.delta < 1e-9) {
                stagnationCounter++;

                // After 10 generations stuck, take aggressive action
                if (stagnationCounter >= stagnationThreshold) {
                    // Increase mutation rate aggressively
                    currentMutationRate = std::min(maxMutationRate, currentMutationRate * 2.0);

                    // PROTECTED ELITE + DIVERSITY INJECTION
                    // Protect top 20% (40 individuals), replace bottom 30% (60 individuals)
                    int protectedCount = populationSize / 5;     // 20% = 40 protected
                    int injectCount = (populationSize * 3) / 10; // 30% = 60 to replace
                    int startIdx = populationSize - injectCount;

                    for (int i = 0; i < injectCount; i++) {
                        int targetIdx = startIdx + i;

                        // Alternate between random immigrants and heavily mutated good individuals
                        if (i % 2 == 0) {
                            // Random immigrant
                            population[targetIdx] = generateRandomIndividual();
                        } else {
                            // Take from protected elite and heavily mutate (5 mutations)
                            Individual mutant = population[i % protectedCount];
                            for (int m = 0; m < 5; m++) {
                                inversionMutation(mutant);
                            }
                            population[targetIdx] = mutant;
                        }
                    }

                    // Reset stagnation counter after injection
                    stagnationCounter = 0;
                    diversityInjections++;
                }

                // Hard restart after prolonged stall
                if (stagnationCounter >= restartThreshold) {
                    int keep = std::max(1, populationSize / 20); // keep top 5%
                    std::sort(population.begin(), population.end());
                    for (int i = keep; i < populationSize; i++) {
                        population[i] = generateRandomIndividual();
                    }
                    currentMutationRate = restartMutationRate;
                    stagnationCounter = 0;
                }
            } else {
                stagnationCounter = 0;
                // Decrease mutation rate if improving
                currentMutationRate = std::max(minMutationRate, currentMutationRate * 0.95);
            }

            prevBest = bestSolution.distance;

            // Print progress
            if (verbose && (gen % 100 == 0 || gen == 1)) {
                printStats(stats);
                if (tsp.optimalTourLength > 0) {
                    double gap = tsp.getGapFromOptimal(bestSolution.distance);
                    std::cout << "  Gap: " << gap << "% | Injections: " << diversityInjections
                              << " | MutRate: " << currentMutationRate << std::endl;
                }
            }

            // Early termination checks (if enabled)
            if (enableEarlyTermination && !forceFullGenerations) {
                // Check if gap from optimal is within target threshold
                if (tsp.optimalTourLength > 0) {
                    double gap = tsp.getGapFromOptimal(bestSolution.distance);
                    if (gap <= targetGap) {
                        if (verbose) {
                            std::cout << "\n*** Early termination: target gap (" << targetGap << "%) reached at generation " << gen << " (actual gap: " << gap << "%) ***" << std::endl;
                        }
                        break;
                    }
                }

                // Check if no improvement for specified generations
                if (stagnationCounter >= noImprovementLimit) {
                    if (verbose) {
                        std::cout << "\n*** Early termination: no improvement for " << noImprovementLimit << " generations at generation " << gen << " ***" << std::endl;
                    }
                    break;
                }
            }

            // Apply full 2-opt to best solution periodically, but adjust frequency based on instance size
            if (tsp.dimension >= 100) {
                if (tsp.dimension < 500 && gen % 500 == 0 && gen > 0) {
                    twoOptFull(population[0]);
                    std::sort(population.begin(), population.end());
                    updateBestSolution();
                }
                // For larger instances, apply 2-opt less frequently to save time
                else if (tsp.dimension >= 500 && gen % 1000 == 0 && gen > 0) {
                    twoOptFull(population[0]);
                    std::sort(population.begin(), population.end());
                    updateBestSolution();
                }
            }
        }

        // Final 2-opt on best solution
        twoOptFull(bestSolution);

        // Final LK polish on the best solution
        linKernighanOptimize(bestSolution);

        if (verbose) {
            std::cout << "\n=== GA Complete ===" << std::endl;
            printBestSolution();
        }

        return bestSolution;
    }
};

#endif // GENETIC_ALGORITHM_H
