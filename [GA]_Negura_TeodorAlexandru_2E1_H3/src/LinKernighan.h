#ifndef LIN_KERNIGHAN_H
#define LIN_KERNIGHAN_H

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include "TSPInstance.h"

/**
 * Lin-Kernighan Style Local Search for TSP
 *
 * This is a simplified but robust implementation that combines:
 * - Candidate lists (K nearest neighbors) for faster search
 * - Don't-look bits for efficiency
 * - Multiple 2-opt passes with smart candidate selection
 * - Validation to ensure tour integrity
 */
class LinKernighan {
private:
    const TSPInstance& tsp;
    int n;

    std::vector<int> tour;
    std::vector<int> position;
    double tourLength;

    static constexpr int K = 5;
    std::vector<std::vector<int>> candidates;
    std::vector<bool> dontLook;

public:
    LinKernighan(const TSPInstance& instance)
        : tsp(instance), n(instance.dimension) {

        tour.resize(n);
        position.resize(n);
        candidates.resize(n);
        dontLook.resize(n, false);

        computeCandidates();
    }

    void computeCandidates() {
        for (int i = 0; i < n; i++) {
            std::vector<std::pair<double, int>> distances;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    distances.push_back({tsp.getDistance(i, j), j});
                }
            }
            std::sort(distances.begin(), distances.end());

            int numCandidates = std::min(K, (int)distances.size());
            candidates[i].clear();
            for (int k = 0; k < numCandidates; k++) {
                candidates[i].push_back(distances[k].second);
            }
        }
    }

    void rebuildPosition() {
        for (int i = 0; i < n; i++) {
            position[tour[i]] = i;
        }
    }

    double calculateTourLength() const {
        double len = 0;
        for (int i = 0; i < n; i++) {
            len += tsp.getDistance(tour[i], tour[(i + 1) % n]);
        }
        return len;
    }

    /**
     * Validate tour integrity - ensure it's a valid permutation
     */
    bool validateTour() const {
        std::vector<bool> visited(n, false);
        for (int i = 0; i < n; i++) {
            if (tour[i] < 0 || tour[i] >= n) return false;
            if (visited[tour[i]]) return false;
            visited[tour[i]] = true;
        }
        return true;
    }

    /**
     * Apply 2-opt: reverse segment [i, j]
     */
    void reverse2opt(int i, int j) {
        while (i < j) {
            std::swap(tour[i], tour[j]);
            i++;
            j--;
        }
    }

    /**
     * Standard 2-opt improvement - safe and tested
     */
    bool twoOptImprove() {
        bool improved = false;

        for (int i = 0; i < n - 1; i++) {
            if (dontLook[tour[i]]) continue;

            int a = tour[i];
            int b = tour[i + 1];

            // Try swapping with edges involving candidates
            for (int c : candidates[a]) {
                int posC = position[c];
                int posD = (posC + 1) % n;

                // Skip adjacent or overlapping edges (2-opt requires non-adjacent edges)
                if (posC == i || posC == i + 1 || posD == i) continue;
                int d = tour[posD];
                if (d == a || d == b) continue;

                double oldDist = tsp.getDistance(a, b) + tsp.getDistance(c, d);
                double newDist = tsp.getDistance(a, c) + tsp.getDistance(b, d);

                if (newDist < oldDist - 1e-9) {
                    reverse2opt(i + 1, posC);
                    rebuildPosition();
                    tourLength -= (oldDist - newDist);

                    dontLook[a] = false;
                    dontLook[b] = false;
                    dontLook[c] = false;
                    dontLook[d] = false;

                    improved = true;
                    break;
                }
            }

            if (!improved) {
                dontLook[tour[i]] = true;
            } else {
                break;
            }
        }

        return improved;
    }

    /**
     * Full 2-opt pass (check all pairs, not just candidates) - optimized version
     * Only continues searching after an improvement if maxImprovementsPerPass allows more
     */
    bool twoOptFull() {
        bool improved = false;
        int improvementsThisPass = 0;
        const int maxImprovementsPerPass = 5; // Limit improvements per pass to avoid long runtimes

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 2; j < n; j++) {
                if (i == 0 && j == n - 1) continue;

                int a = tour[i];
                int b = tour[i + 1];
                int c = tour[j];
                int d = tour[(j + 1) % n];

                double oldDist = tsp.getDistance(a, b) + tsp.getDistance(c, d);
                double newDist = tsp.getDistance(a, c) + tsp.getDistance(b, d);

                if (newDist < oldDist - 1e-9) {
                    reverse2opt(i + 1, j);
                    rebuildPosition();
                    tourLength -= (oldDist - newDist);
                    improved = true;
                    improvementsThisPass++;

                    // Limit improvements per pass to avoid very long runtimes
                    if (improvementsThisPass >= maxImprovementsPerPass) {
                        break;
                    }
                }
            }
            if (improvementsThisPass >= maxImprovementsPerPass) {
                break;
            }
        }

        return improved;
    }

    /**
     * Or-opt: Move a segment of 1-3 cities to a better position
     * This is an additional move type that can help escape local optima
     */
    bool orOptImprove() {
        bool improved = false;

        // Try moving segments of size 1, 2, 3
        for (int segLen = 1; segLen <= 3 && !improved; segLen++) {
            for (int i = 0; i < n && !improved; i++) {
                // Segment is tour[i..i+segLen-1]
                if (i + segLen >= n) continue;

                int prevCity = tour[(i - 1 + n) % n];
                int segStart = tour[i];
                int segEnd = tour[i + segLen - 1];
                int nextCity = tour[(i + segLen) % n];

                // Current cost of having segment here
                double currentCost = tsp.getDistance(prevCity, segStart) +
                                     tsp.getDistance(segEnd, nextCity);

                // Cost if we remove the segment (connect prev to next directly)
                double removeCost = tsp.getDistance(prevCity, nextCity);

                // Try inserting the segment at each other position
                for (int j = i + segLen + 1; j < n; j++) {
                    int insertAfter = tour[j];
                    int insertBefore = tour[(j + 1) % n];

                    // Current cost at insertion point
                    double insertCurrentCost = tsp.getDistance(insertAfter, insertBefore);

                    // New cost with segment inserted
                    double insertNewCost = tsp.getDistance(insertAfter, segStart) +
                                           tsp.getDistance(segEnd, insertBefore);

                    double gain = (currentCost + insertCurrentCost) -
                                  (removeCost + insertNewCost);

                    if (gain > 1e-9) {
                        // Apply the move
                        std::vector<int> segment;
                        for (int k = 0; k < segLen; k++) {
                            segment.push_back(tour[i + k]);
                        }

                        // Remove segment
                        tour.erase(tour.begin() + i, tour.begin() + i + segLen);

                        // Find new position (j shifted because we removed)
                        int newJ = j - segLen;

                        // Insert segment
                        tour.insert(tour.begin() + newJ + 1, segment.begin(), segment.end());

                        rebuildPosition();
                        tourLength -= gain;
                        improved = true;
                        break;
                    }
                }
            }
        }

        return improved;
    }

    /**
     * Main optimization: 2-opt + Or-opt
     */
    std::vector<int> optimize(const std::vector<int>& inputTour) {
        tour = inputTour;
        n = tour.size();

        if (!validateTour()) {
            return inputTour;  // Return unchanged if invalid input
        }

        rebuildPosition();
        tourLength = calculateTourLength();
        std::fill(dontLook.begin(), dontLook.end(), false);

        bool improved = true;
        int iterations = 0;
        int maxIterations = n * 5;

        while (improved && iterations < maxIterations) {
            improved = false;
            iterations++;

            // Try candidate-based 2-opt
            if (twoOptImprove()) {
                improved = true;
                continue;
            }

            // Try Or-opt
            if (orOptImprove()) {
                improved = true;
                continue;
            }
        }

        // Final validation
        if (!validateTour()) {
            return inputTour;  // Return original if corrupted
        }

        return tour;
    }

    /**
     * Full optimization with exhaustive 2-opt
     */
    std::vector<int> optimizeFull(const std::vector<int>& inputTour) {
        tour = inputTour;
        n = tour.size();

        if (!validateTour()) {
            return inputTour;
        }

        rebuildPosition();
        tourLength = calculateTourLength();

        // Do limited full 2-opt passes - only if significant improvement is possible
        bool improved = true;
        int passes = 0;
        int maxPasses = (n > 500) ? 3 : 5; // Reduce passes for larger instances
        while (improved && passes < maxPasses) {
            improved = twoOptFull();
            passes++;
        }

        // Then apply the faster candidate-based optimization
        optimize(tour);

        // Final 2-opt cleanup with fewer passes
        improved = true;
        passes = 0;
        int finalPasses = (n > 500) ? 2 : 3; // Reduce final passes for larger instances
        while (improved && passes < finalPasses) {
            improved = twoOptFull();
            passes++;
        }

        if (!validateTour()) {
            return inputTour;
        }

        return tour;
    }

    double getTourLength() const {
        return tourLength;
    }
};

#endif // LIN_KERNIGHAN_H