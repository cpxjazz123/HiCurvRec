# codebook_geometry_r.R
# R version of the 3-panel codebook geometry figure, SS 1.2 evidence
# Panel a - mean pairwise distance across unified codebook sizes
# Panel b - K=128 NN distance down + local density k=5 up
# Panel c - K=128 effective rank up + PC1 variance share down

suppressMessages({
  library(ggplot2)
  library(patchwork)
})

theme_set(
  theme_classic(base_size = 6.5, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.text = element_text(size = 5.8),
      axis.title = element_text(size = 6.2),
      legend.title = element_blank(),
      legend.text = element_text(size = 5.8),
      legend.key.size = unit(2.4, "mm"),
      legend.position = "bottom",
      plot.title = element_text(size = 6.8, face = "bold", hjust = 0.5,
                                margin = margin(b = 1.5)),
      panel.grid = element_blank(),
      plot.margin = margin(2, 4, 2, 2, "pt")
    )
)

BLUE_DARK <- "#0F4D92"
BLUE_MID <- "#3775BA"
BLUE_SOFT <- "#B4C0E4"
BLACK <- "#272727"
RED <- "#B64342"
TEAL <- "#42949E"

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
           colour = "black", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.4f", value)),
            position = position_dodge(width = 0.8),
            size = 2.0, vjust = -0.55) +
  scale_fill_manual(values = c(L0 = BLUE_DARK, L1 = BLUE_MID, L2 = BLUE_SOFT)) +
  scale_y_continuous(limits = c(0, 0.26), expand = expansion(mult = c(0, 0.04))) +
  labs(y = "Mean pairwise distance", title = "Global scale") +
  theme(legend.position = c(0.82, 0.85),
        legend.background = element_blank())

# ---- Panel b: NN distance (left) + local density (right) ----
dfb <- data.frame(
  layer = c("L0", "L1", "L2"),
  nn = c(0.1044, 0.0767, 0.0641),
  dens = c(8.47, 12.46, 14.96)
)
dfb$layer <- factor(dfb$layer, levels = c("L0", "L1", "L2"))
dens_span <- (0.115 - 0.05) / (17 - 6)  # rescale density into nn axis range
dfb$dens_y <- 0.05 + (dfb$dens - 6) * dens_span

pb <- ggplot(dfb, aes(x = layer)) +
  geom_line(aes(y = nn, group = 1), colour = BLUE_DARK, linewidth = 0.6) +
  geom_point(aes(y = nn), colour = BLUE_DARK, size = 1.4) +
  geom_text(aes(y = nn, label = sprintf("%.4f", nn)),
            colour = BLUE_DARK, size = 2.0, vjust = 1.8) +
  geom_line(aes(y = dens_y, group = 1),
            colour = RED, linewidth = 0.6, linetype = "dashed") +
  geom_point(aes(y = dens_y), colour = RED, size = 1.4, shape = 15) +
  geom_text(aes(y = dens_y, label = sprintf("%.2f", dens)),
            colour = RED, size = 2.0, vjust = -1.0) +
  scale_y_continuous(
    name = "Mean NN distance",
    limits = c(0.05, 0.115),
    sec.axis = sec_axis(~ 6 + (. - 0.05) / dens_span,
                        name = "Local density (k = 5)")
  ) +
  labs(title = "Local structure", x = NULL)

# ---- Panel c: effective rank (left) + PC1 variance (right) ----
dfc <- data.frame(
  layer = c("L0", "L1", "L2"),
  rank = c(15.65, 22.17, 23.64),
  pc1 = c(20.54, 8.36, 6.33)
)
dfc$layer <- factor(dfc$layer, levels = c("L0", "L1", "L2"))

pc <- ggplot(dfc, aes(x = layer)) +
  geom_line(aes(y = rank, group = 1), colour = TEAL, linewidth = 0.6) +
  geom_point(aes(y = rank), colour = TEAL, size = 1.4) +
  geom_text(aes(y = rank, label = sprintf("%.2f", rank)),
            colour = TEAL, size = 2.0, vjust = -1.0) +
  geom_line(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2), group = 1),
            colour = BLACK, linewidth = 0.6, linetype = "dotted") +
  geom_point(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2)),
             colour = BLACK, size = 1.4, shape = 17) +
  geom_text(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2),
                label = sprintf("%.2f%%", pc1)),
            colour = BLACK, size = 2.0, vjust = 1.6) +
  scale_y_continuous(
    name = "Effective rank",
    limits = c(13, 26),
    sec.axis = sec_axis(~ 6.33 + (. - 13) * (25 - 2) / (26 - 13),
                        name = "Var. explained by PC1 (%)")
  ) +
  labs(title = "Spectral shape", x = NULL)

# ---- Compose ----
fig <- (pa | pb | pc) + plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9, face = "bold"),
        plot.tag.position = c(0, 1))

save_pub_r <- function(plot, filename, width_mm = 186, height_mm = 63, dpi = 600) {
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
  cat("saved svg/pdf/tiff\n")
}

save_pub_r(fig, "codebook_geometry_r")
