# codebook_geometry_split.R
# Split the 3-panel codebook geometry figure into three standalone panels.
# a: mean pairwise distance across unified codebook sizes
# b: K=128 NN distance + local density k=5
# c: K=128 effective rank + PC1 variance share

suppressMessages({
  library(ggplot2)
})

theme_set(
  theme_classic(base_size = 7, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.4, colour = "black"),
      axis.ticks = element_line(linewidth = 0.4, colour = "black"),
      axis.text = element_text(size = 7),
      axis.title = element_text(size = 7),
      legend.title = element_blank(),
      legend.text = element_text(size = 7),
      legend.key.size = unit(2.2, "mm"),
      legend.position = "bottom",
      plot.title = element_text(size = 7, face = "bold", hjust = 0.5,
                                margin = margin(b = 2)),
      panel.grid = element_blank(),
      plot.margin = margin(3, 5, 3, 3, "pt")
    )
)

BLUE_DARK <- "#0F4D92"
BLUE_MID <- "#3775BA"
BLUE_SOFT <- "#B4C0E4"
BLACK <- "#272727"
RED <- "#B64342"
TEAL <- "#42949E"

save_panel <- function(plot, filename, width_mm = 74, height_mm = 70, dpi = 600) {
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

# ---- Panel a: pairwise distance grouped bars ----
dfa <- data.frame(
  K = rep(c("K = 64", "K = 128", "K = 256"), each = 3),
  layer = rep(c("L0", "L1", "L2"), 3),
  value = c(0.1922, 0.1059, 0.0856,
            0.1946, 0.1060, 0.0839,
            0.1993, 0.1035, 0.0799)
)
dfa$layer <- factor(dfa$layer, levels = c("L0", "L1", "L2"))
dfa$K <- factor(dfa$K, levels = c("K = 64", "K = 128", "K = 256"))

pa <- ggplot(dfa, aes(x = K, y = value, fill = layer)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.62,
           colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c(L0 = BLUE_DARK, L1 = BLUE_MID, L2 = BLUE_SOFT)) +
  scale_y_continuous(limits = c(0, 0.26), expand = expansion(mult = c(0, 0.04))) +
  labs(y = "Mean pairwise distance") +
  theme(legend.position = c(0.80, 0.86),
        legend.background = element_blank(),
        legend.key.height = unit(0.30, "cm"),
        legend.spacing.y = unit(1.0, "mm"))

# ---- Panel b: NN distance (left) + local density (right) ----
dfb <- data.frame(
  layer = c("L0", "L1", "L2"),
  nn = c(0.1044, 0.0767, 0.0641),
  dens = c(8.47, 12.46, 14.96)
)
dfb$layer <- factor(dfb$layer, levels = c("L0", "L1", "L2"))
dens_span <- (0.115 - 0.05) / (17 - 6)
dfb$dens_y <- 0.05 + (dfb$dens - 6) * dens_span

pb <- ggplot(dfb, aes(x = layer)) +
  geom_line(aes(y = nn, group = 1), colour = BLUE_DARK, linewidth = 0.7) +
  geom_point(aes(y = nn), colour = BLUE_DARK, size = 1.7) +
  geom_line(aes(y = dens_y, group = 1),
            colour = RED, linewidth = 0.7, linetype = "dashed") +
  geom_point(aes(y = dens_y), colour = RED, size = 1.7, shape = 15) +
  scale_y_continuous(
    name = "Mean NN distance",
    limits = c(0.05, 0.115),
    sec.axis = sec_axis(~ 6 + (. - 0.05) / dens_span,
                        name = "Local density (k = 5)")
  ) +
  labs(x = NULL)

# ---- Panel c: effective rank (left) + PC1 variance (right) ----
dfc <- data.frame(
  layer = c("L0", "L1", "L2"),
  rank = c(15.65, 22.17, 23.64),
  pc1 = c(20.54, 8.36, 6.33)
)
dfc$layer <- factor(dfc$layer, levels = c("L0", "L1", "L2"))

pc <- ggplot(dfc, aes(x = layer)) +
  geom_line(aes(y = rank, group = 1), colour = TEAL, linewidth = 0.7) +
  geom_point(aes(y = rank), colour = TEAL, size = 1.7) +
  geom_text(aes(y = rank, label = sprintf("%.2f", rank)),
            colour = TEAL, size = 2.47, vjust = -1.0, hjust = -0.05, nudge_x = 0.02) +
  geom_line(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2), group = 1),
            colour = BLACK, linewidth = 0.7, linetype = "dotted") +
  geom_point(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2)),
             colour = BLACK, size = 1.7, shape = 17) +
  geom_text(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2),
                label = sprintf("%.2f%%", pc1),
                vjust = ifelse(pc1 == min(pc1), -1.3, 1.7),
                hjust = ifelse(pc1 == max(pc1), -0.05, 1.05)),
            colour = BLACK, size = 2.47) +
  scale_y_continuous(
    name = "Effective rank",
    limits = c(13, 26),
    sec.axis = sec_axis(~ 6.33 + (. - 13) * (25 - 2) / (26 - 13),
                        name = "Var. explained by PC1 (%)")
  ) +
  labs(x = NULL)

save_panel(pa, "codebook_geometry_a", width_mm = 47.6, height_mm = 45.1)
save_panel(pb, "codebook_geometry_b", width_mm = 47.6, height_mm = 34.9)
save_panel(pc, "codebook_geometry_c", width_mm = 47.6, height_mm = 34.9)
