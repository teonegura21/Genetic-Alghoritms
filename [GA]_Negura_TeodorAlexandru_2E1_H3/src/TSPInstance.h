#ifndef TSP_INSTANCE_H
#define TSP_INSTANCE_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <random>

// Constants for GEO distance calculation
const double PI = 3.141592653589793;
const double RRR = 6378.388; // Earth radius in km

/**
 * @brief Represents a TSP instance loaded from TSPLIB format
 */
class TSPInstance {
public:
    std::string name;
    std::string comment;
    int dimension;
    std::string edgeWeightType;
    std::vector<std::pair<double, double>> coordinates;
    std::vector<std::vector<double>> distanceMatrix;
    int optimalTourLength;

    TSPInstance() : dimension(0), optimalTourLength(-1) {}

    /**
     * @brief Load a TSP instance from a TSPLIB format file
     * @param filename Path to the .tsp file
     * @return true if loading was successful
     */
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        std::string line;
        bool readingCoords = false;

        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (line.empty()) continue;
            if (line == "EOF") break;

            if (readingCoords) {
                // Parse coordinate line: node_id x y
                std::istringstream iss(line);
                int id;
                double x, y;
                if (iss >> id >> x >> y) {
                    coordinates.push_back({x, y});
                }
            } else {
                // Parse header
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos) {
                    std::string key = line.substr(0, colonPos);
                    std::string value = line.substr(colonPos + 1);
                    
                    // Trim key and value
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                    if (key == "NAME") {
                        name = value;
                    } else if (key == "COMMENT") {
                        comment = value;
                    } else if (key == "DIMENSION") {
                        dimension = std::stoi(value);
                    } else if (key == "EDGE_WEIGHT_TYPE") {
                        edgeWeightType = value;
                    }
                } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                    readingCoords = true;
                }
            }
        }

        file.close();

        if (coordinates.size() != static_cast<size_t>(dimension)) {
            std::cerr << "Warning: Expected " << dimension << " cities but read " 
                      << coordinates.size() << std::endl;
            dimension = coordinates.size();
        }

        // Build distance matrix
        buildDistanceMatrix();

        return true;
    }

    /**
     * @brief Build the distance matrix based on edge weight type
     */
    void buildDistanceMatrix() {
        distanceMatrix.resize(dimension, std::vector<double>(dimension, 0.0));

        for (int i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                double dist;
                if (edgeWeightType == "GEO") {
                    dist = calculateGeoDistance(i, j);
                } else {
                    // Default to EUC_2D (Euclidean distance)
                    dist = calculateEuclideanDistance(i, j);
                }
                distanceMatrix[i][j] = dist;
                distanceMatrix[j][i] = dist;
            }
        }
    }

    /**
     * @brief Calculate Euclidean distance between two cities
     */
    double calculateEuclideanDistance(int i, int j) const {
        double dx = coordinates[i].first - coordinates[j].first;
        double dy = coordinates[i].second - coordinates[j].second;
        return std::round(std::sqrt(dx * dx + dy * dy));
    }

    /**
     * @brief Calculate geographical distance (for GEO type instances)
     * Uses the TSPLIB GEO distance calculation formula
     */
    double calculateGeoDistance(int i, int j) const {
        // Convert coordinates to radians (TSPLIB format: degrees.minutes)
        auto toRadians = [](double coord) {
            int deg = static_cast<int>(coord);
            double min = coord - deg;
            return PI * (deg + 5.0 * min / 3.0) / 180.0;
        };

        double lat1 = toRadians(coordinates[i].first);
        double lon1 = toRadians(coordinates[i].second);
        double lat2 = toRadians(coordinates[j].first);
        double lon2 = toRadians(coordinates[j].second);

        double q1 = std::cos(lon1 - lon2);
        double q2 = std::cos(lat1 - lat2);
        double q3 = std::cos(lat1 + lat2);

        return static_cast<int>(RRR * std::acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 0.5);
    }

    /**
     * @brief Get distance between two cities (0-indexed)
     */
    double getDistance(int i, int j) const {
        return distanceMatrix[i][j];
    }

    /**
     * @brief Calculate total tour length for a given tour
     * @param tour Vector of city indices (0-indexed)
     * @return Total distance of the tour
     */
    double calculateTourLength(const std::vector<int>& tour) const {
        if (tour.size() != static_cast<size_t>(dimension)) {
            throw std::invalid_argument("Tour size does not match number of cities");
        }

        double totalDistance = 0.0;
        const size_t tourSize = tour.size();
        for (size_t i = 0; i < tourSize - 1; i++) {
            totalDistance += distanceMatrix[tour[i]][tour[i + 1]];
        }
        // Return to starting city
        totalDistance += distanceMatrix[tour.back()][tour.front()];

        return totalDistance;
    }

    /**
     * @brief Validate that a tour visits each city exactly once
     */
    bool isValidTour(const std::vector<int>& tour) const {
        if (tour.size() != static_cast<size_t>(dimension)) {
            return false;
        }

        std::vector<bool> visited(dimension, false);
        for (int city : tour) {
            if (city < 0 || city >= dimension || visited[city]) {
                return false;
            }
            visited[city] = true;
        }
        return true;
    }

    /**
     * @brief Generate a random tour (random permutation of cities)
     */
    std::vector<int> generateRandomTour() const {
        std::vector<int> tour(dimension);
        for (int i = 0; i < dimension; i++) {
            tour[i] = i;
        }

        // Fisher-Yates shuffle using a local random number generator for thread safety
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int i = dimension - 1; i > 0; i--) {
            std::uniform_int_distribution<> dis(0, i);
            int j = dis(gen);
            std::swap(tour[i], tour[j]);
        }

        return tour;
    }

    /**
     * @brief Generate a greedy (nearest neighbor) tour
     * @param startCity Starting city index
     */
    std::vector<int> generateNearestNeighborTour(int startCity = 0) const {
        std::vector<int> tour;
        std::vector<bool> visited(dimension, false);

        int currentCity = startCity;
        tour.push_back(currentCity);
        visited[currentCity] = true;

        for (int i = 1; i < dimension; i++) {
            int nearestCity = -1;
            double nearestDistance = std::numeric_limits<double>::max();

            for (int j = 0; j < dimension; j++) {
                if (!visited[j] && distanceMatrix[currentCity][j] < nearestDistance) {
                    nearestDistance = distanceMatrix[currentCity][j];
                    nearestCity = j;
                }
            }

            tour.push_back(nearestCity);
            visited[nearestCity] = true;
            currentCity = nearestCity;
        }

        return tour;
    }

    /**
     * @brief Print instance information
     */
    void printInfo() const {
        std::cout << "=== TSP Instance ===" << std::endl;
        std::cout << "Name: " << name << std::endl;
        std::cout << "Comment: " << comment << std::endl;
        std::cout << "Dimension: " << dimension << " cities" << std::endl;
        std::cout << "Edge Weight Type: " << edgeWeightType << std::endl;
        if (optimalTourLength > 0) {
            std::cout << "Optimal Tour Length: " << optimalTourLength << std::endl;
        }
        std::cout << "===================" << std::endl;
    }

    /**
     * @brief Set the known optimal tour length
     */
    void setOptimalTourLength(int optimal) {
        optimalTourLength = optimal;
    }

    /**
     * @brief Get the gap percentage from optimal
     */
    double getGapFromOptimal(double tourLength) const {
        if (optimalTourLength <= 0) return -1.0;
        return ((tourLength - optimalTourLength) / optimalTourLength) * 100.0;
    }
};

#endif // TSP_INSTANCE_H
