# curvature_distortion.R
# Figure 3: layer-wise Curvature-Distortion curves (K=128, Issue #83 real data)
# x: curvature c (log scale, c=0 = Euclidean placed at leftmost)
# y: normalized Distortion_l(c)

suppressMessages({ library(ggplot2) })

theme_set(
  theme_classic(base_size = 7, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.4, colour = "black"),
      axis.ticks = element_line(linewidth = 0.4, colour = "black"),
      axis.text = element_text(size = 7),
      axis.title = element_text(size = 7),
      legend.title = element_blank(),
      legend.text = element_text(size = 7),
      legend.key.size = unit(3.0, "mm"),
      legend.position = "bottom",
      plot.margin = margin(3, 5, 3, 3, "pt")
    )
)

BLUE_DARK <- "#0F4D92"
RED <- "#B64342"
GREEN <- "#2E9E44"
BLACK <- "#272727"

c_vals <- c(0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10)

# K=128 Issue #83 main_summary.json distortion curves
dist <- data.frame(
  c = rep(c_vals, 3),
  layer = factor(rep(c("L0", "L1", "L2"), each = 9), levels = c("L0", "L1", "L2")),
  value = c(
    0.03515922954365548, 0.03516100556387154, 0.03516811535492724,
    0.0351770155118316, 0.035248719312496124, 0.03533953874834355,
    0.03552472337740152, 0.036101738667250555, 0.0370998240551487,
    0.028388117298469732, 0.028387652100532502, 0.028385790530801428,
    0.028383465031557163, 0.028364919409049946, 0.02834188262517362,
    0.02829628868670041, 0.02816326024351378, 0.027953523422096307,
    0.030070812428020472, 0.030070462161790743, 0.03006906196666432,
    0.03006731228976074, 0.030053337548401837, 0.030035925733845878,
    0.030001290320596133, 0.029898880375301407, 0.029733120410372696
  )
)

# x mapping: log10(c) for c>0; c=0 (Euclidean) placed at x = -2.6
dist$x <- ifelse(dist$c == 0, -2.6, log10(dist$c))

x_breaks <- c(-2.6, -2, -1, 0, 1)
x_labels <- c("0 (Eucl.)", "0.01", "0.1", "1", "10")

minima <- aggregate(value ~ layer, dist, min)
minima$x <- ifelse(minima$layer == "L0", -2.6, 1)

p <- ggplot(dist, aes(x = x, y = value, colour = layer)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = BLACK, linewidth = 0.3, alpha = 0.6) +
  geom_point(data = minima, aes(x = x, y = value), size = 2.4, shape = 21,
             fill = "white", stroke = 0.8) +
  scale_colour_manual(values = c(L0 = BLUE_DARK, L1 = RED, L2 = GREEN)) +
  scale_x_continuous(breaks = x_breaks, labels = x_labels, limits = c(-2.9, 1.3)) +
  scale_y_continuous(limits = c(0.025, 0.040)) +
  labs(x = "Curvature c", y = "Normalized distortion") +
  annotate("text", x = -2.62, y = 0.0376, label = "L0", colour = BLUE_DARK,
           size = 2.5, fontface = "bold", hjust = 0) +
  annotate("text", x = 1.05, y = 0.0276, label = "L1", colour = RED,
           size = 2.5, fontface = "bold", hjust = 0) +
  annotate("text", x = 1.05, y = 0.0294, label = "L2", colour = GREEN,
           size = 2.5, fontface = "bold", hjust = 0)

save_pub_r <- function(plot, filename, width_mm = 100, height_mm = 62, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot)
  dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()
  ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
  cat("saved", filename, "\n")
}

save_pub_r(p, "curvature_distortion", width_mm = 84, height_mm = 55)
