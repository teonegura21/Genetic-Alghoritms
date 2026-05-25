# Hill Climbing Visualization Script
# This script creates comprehensive visualizations of First and Best Improvement algorithms

library(ggplot2)
library(dplyr)
library(gridExtra)
library(viridis)
library(scales)

# Set working directory to script location
setwd("C:/Users/TEODO/Desktop/Facultate/Cod/Alg Genetici/tema h1'")

# Read data
landscape <- read.csv("landscape.csv")
fi_traj <- read.csv("fi_trajectories.csv")
bi_traj <- read.csv("bi_trajectories.csv")
basins <- read.csv("basins.csv")

# Define theme for plots
theme_custom <- theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    axis.title = element_text(size = 11, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray70", fill = NA, linewidth = 0.5)
  )

# 1. FITNESS LANDSCAPE WITH LOCAL MAXIMA
print("Creating fitness landscape plot...")

# Identify local maxima from basins data
local_maxima_bi <- basins %>%
  filter(Algorithm == "BI") %>%
  distinct(EndPoint) %>%
  pull(EndPoint)

landscape_data <- landscape %>%
  mutate(IsLocalMax = Position %in% local_maxima_bi)

p1 <- ggplot(landscape_data, aes(x = Position, y = Fitness)) +
  geom_line(color = "#2c3e50", linewidth = 1.2) +
  geom_point(data = filter(landscape_data, IsLocalMax),
             aes(color = factor(Position)), size = 5, shape = 18) +
  geom_point(data = filter(landscape_data, !IsLocalMax),
             color = "#95a5a6", size = 2, alpha = 0.6) +
  scale_color_viridis_d(name = "Local Maxima", option = "C") +
  labs(
    title = "Fitness Landscape",
    subtitle = "f(x) = x³ - 60x² + 900x + 100",
    x = "Position (0-31)",
    y = "Fitness Value"
  ) +
  theme_custom +
  scale_x_continuous(breaks = seq(0, 31, 5)) +
  scale_y_continuous(labels = comma)

ggsave("landscape_plot.png", p1, width = 10, height = 6, dpi = 300)

# 2. FIRST IMPROVEMENT TRAJECTORIES - Selected examples
print("Creating First Improvement trajectory plot...")

# Select interesting trajectories (different end points)
selected_starts_fi <- basins %>%
  filter(Algorithm == "FI") %>%
  group_by(EndPoint) %>%
  slice(1:2) %>%  # Take first 2 examples of each endpoint
  ungroup() %>%
  pull(StartPoint)

fi_traj_selected <- fi_traj %>%
  filter(StartPoint %in% selected_starts_fi)

p2 <- ggplot() +
  geom_line(data = landscape, aes(x = Position, y = Fitness),
            color = "gray80", linewidth = 1, alpha = 0.5) +
  geom_path(data = fi_traj_selected,
            aes(x = Position, y = Fitness, group = StartPoint, color = factor(StartPoint)),
            linewidth = 1.2, alpha = 0.8,
            arrow = arrow(length = unit(0.2, "cm"), type = "closed")) +
  geom_point(data = fi_traj_selected %>% filter(Step == 0),
             aes(x = Position, y = Fitness, color = factor(StartPoint)),
             size = 4, shape = 21, fill = "white", stroke = 2) +
  scale_color_viridis_d(name = "Start Point", option = "D") +
  labs(
    title = "First Improvement: Search Trajectories",
    subtitle = "Arrows show path from start to local maximum",
    x = "Position (0-31)",
    y = "Fitness Value"
  ) +
  theme_custom +
  scale_x_continuous(breaks = seq(0, 31, 5)) +
  scale_y_continuous(labels = comma)

ggsave("fi_trajectories.png", p2, width = 12, height = 7, dpi = 300)

# 3. DETAILED TRAJECTORY PLOT - Show step-by-step jumps
print("Creating detailed jump visualization...")

# Select 4-6 interesting trajectories for detailed view
detailed_starts <- c(0, 5, 10, 15, 20, 25)
fi_detailed <- fi_traj %>%
  filter(StartPoint %in% detailed_starts)

p3 <- ggplot() +
  geom_line(data = landscape, aes(x = Position, y = Fitness),
            color = "gray70", linewidth = 0.8) +
  geom_segment(data = fi_detailed %>%
                 group_by(StartPoint) %>%
                 filter(Step < max(Step)) %>%
                 ungroup(),
               aes(x = Position, xend = lead(Position, order_by = Step),
                   y = Fitness, yend = lead(Fitness, order_by = Step),
                   color = factor(StartPoint)),
               arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
               linewidth = 1, alpha = 0.7) +
  geom_point(data = fi_detailed,
             aes(x = Position, y = Fitness, color = factor(StartPoint)),
             size = 3) +
  geom_label(data = fi_detailed %>% filter(Step == 0),
            aes(x = Position, y = Fitness, label = paste0("S", StartPoint)),
            size = 2.5, nudge_y = 150, fontface = "bold") +
  scale_color_viridis_d(name = "Start Point", option = "plasma") +
  labs(
    title = "First Improvement: Point-to-Point Jumps",
    subtitle = "Detailed view showing each step in the search",
    x = "Position (0-31)",
    y = "Fitness Value"
  ) +
  theme_custom +
  facet_wrap(~StartPoint, ncol = 3, labeller = label_both) +
  scale_y_continuous(labels = comma) +
  theme(strip.text = element_text(face = "bold", size = 9))

ggsave("fi_detailed_jumps.png", p3, width = 14, height = 10, dpi = 300)

# 4. BASIN OF ATTRACTION COMPARISON
print("Creating basin of attraction comparison...")

basins_summary <- basins %>%
  group_by(Algorithm, EndPoint) %>%
  summarise(BasinSize = n(), .groups = "drop") %>%
  left_join(landscape, by = c("EndPoint" = "Position")) %>%
  mutate(Label = paste0("x=", EndPoint, "\nf=", Fitness))

p4 <- ggplot(basins_summary, aes(x = factor(EndPoint), y = BasinSize, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = BasinSize), position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("BI" = "#3498db", "FI" = "#e74c3c"),
                    labels = c("Best Improvement", "First Improvement")) +
  labs(
    title = "Basin of Attraction Sizes",
    subtitle = "Number of starting points leading to each local maximum",
    x = "Local Maximum Position",
    y = "Basin Size (number of starting points)",
    fill = "Algorithm"
  ) +
  theme_custom +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave("basin_comparison.png", p4, width = 12, height = 6, dpi = 300)

# 5. SUCCESS RATE COMPARISON
print("Creating success rate comparison...")

# Identify global optimum (highest fitness)
global_opt <- landscape %>% filter(Fitness == max(Fitness)) %>% pull(Position)

success_data <- basins %>%
  mutate(Success = ifelse(EndPoint == global_opt, "Global Optimum", "Local Trap")) %>%
  group_by(Algorithm, Success) %>%
  summarise(Count = n(), .groups = "drop") %>%
  mutate(Percentage = Count / 32 * 100)

p5 <- ggplot(success_data, aes(x = Algorithm, y = Percentage, fill = Success)) +
  geom_bar(stat = "identity", position = "stack", width = 0.6) +
  geom_text(aes(label = sprintf("%d (%.1f%%)", Count, Percentage)),
            position = position_stack(vjust = 0.5),
            size = 4, fontface = "bold", color = "white") +
  scale_fill_manual(values = c("Global Optimum" = "#27ae60", "Local Trap" = "#c0392b")) +
  scale_x_discrete(labels = c("BI" = "Best Improvement", "FI" = "First Improvement")) +
  labs(
    title = "Algorithm Success Rate Comparison",
    subtitle = "Percentage reaching global optimum vs trapped in local maxima",
    x = "Algorithm",
    y = "Percentage of Starting Points (%)",
    fill = "Outcome"
  ) +
  theme_custom +
  coord_flip()

ggsave("success_rate.png", p5, width = 10, height = 6, dpi = 300)

# 6. STEPS TO CONVERGENCE
print("Creating convergence steps comparison...")

p6 <- ggplot(basins, aes(x = Algorithm, y = Steps, fill = Algorithm)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 3) +
  geom_jitter(width = 0.2, alpha = 0.4, size = 2) +
  scale_fill_manual(values = c("BI" = "#3498db", "FI" = "#e74c3c"),
                    labels = c("Best Improvement", "First Improvement")) +
  scale_x_discrete(labels = c("BI" = "Best Improvement", "FI" = "First Improvement")) +
  labs(
    title = "Steps to Convergence",
    subtitle = "Number of iterations until reaching local maximum",
    x = "Algorithm",
    y = "Number of Steps",
    fill = "Algorithm"
  ) +
  theme_custom +
  theme(legend.position = "none")

ggsave("convergence_steps.png", p6, width = 10, height = 6, dpi = 300)

# 7. ALGORITHM COMPARISON SIDE-BY-SIDE
print("Creating side-by-side comparison...")

# Create comparison plot for selected start points
comparison_starts <- c(0, 8, 16, 24)

fi_comp <- fi_traj %>% filter(StartPoint %in% comparison_starts) %>% mutate(Algorithm = "First Improvement")
bi_comp <- bi_traj %>% filter(StartPoint %in% comparison_starts) %>% mutate(Algorithm = "Best Improvement")

comp_data <- bind_rows(fi_comp, bi_comp)

p7 <- ggplot() +
  geom_line(data = landscape, aes(x = Position, y = Fitness),
            color = "gray80", linewidth = 0.8) +
  geom_path(data = comp_data,
            aes(x = Position, y = Fitness, color = Algorithm, group = interaction(StartPoint, Algorithm)),
            linewidth = 1.2, alpha = 0.7,
            arrow = arrow(length = unit(0.15, "cm"), type = "closed")) +
  geom_point(data = comp_data %>% filter(Step == 0),
             aes(x = Position, y = Fitness), size = 3, shape = 21, fill = "white") +
  scale_color_manual(values = c("Best Improvement" = "#3498db", "First Improvement" = "#e74c3c")) +
  facet_wrap(~StartPoint, ncol = 2, labeller = labeller(StartPoint = function(x) paste("Start:", x))) +
  labs(
    title = "Algorithm Comparison: Side-by-Side Trajectories",
    subtitle = "Blue = Best Improvement, Red = First Improvement",
    x = "Position (0-31)",
    y = "Fitness Value",
    color = "Algorithm"
  ) +
  theme_custom +
  scale_y_continuous(labels = comma) +
  theme(strip.text = element_text(face = "bold"))

ggsave("algorithm_comparison.png", p7, width = 14, height = 10, dpi = 300)

print("=== All visualizations created successfully! ===")
print("Generated files:")
print("  1. landscape_plot.png - Fitness landscape with local maxima")
print("  2. fi_trajectories.png - First Improvement trajectories")
print("  3. fi_detailed_jumps.png - Detailed point-to-point jumps")
print("  4. basin_comparison.png - Basin of attraction sizes")
print("  5. success_rate.png - Success rate comparison")
print("  6. convergence_steps.png - Steps to convergence")
print("  7. algorithm_comparison.png - Side-by-side comparison")
