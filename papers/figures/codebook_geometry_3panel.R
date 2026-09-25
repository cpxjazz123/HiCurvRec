# codebook_geometry_3panel.R
# Figure 1: three-panel codebook geometry heterogeneity
# (a) global pairwise distance  (b) local NN + density  (c) spectral rank + PC1
# Font 7pt, canvas ~183mm wide (full two-column width), 600 dpi exports.

suppressMessages({ library(ggplot2); library(patchwork) })

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
      legend.key.height = unit(0.30, "cm"),
      legend.spacing.y = unit(1.0, "mm"),
      legend.position = "bottom",
      plot.margin = margin(3, 5, 3, 3, "pt")
    )
)

BLUE_DARK <- "#0F4D92"
BLUE_MID <- "#3775BA"
BLUE_SOFT <- "#B4C0E4"
BLACK <- "#272727"
RED <- "#B64342"
TEAL <- "#42949E"

# ---- (a) global: pairwise distance grouped bars ----
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
  labs(y = "Mean pairwise distance", x = NULL) +
  theme(legend.position = c(0.80, 0.86),
        legend.background = element_blank())

# ---- (b) local: NN distance + local density (K=128) ----
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
  geom_line(aes(y = dens_y, group = 1), colour = RED, linewidth = 0.7, linetype = "dashed") +
  geom_point(aes(y = dens_y), colour = RED, size = 1.7, shape = 15) +
  scale_y_continuous(
    name = "Mean NN distance",
    limits = c(0.05, 0.115),
    sec.axis = sec_axis(~ 6 + (. - 0.05) / dens_span,
                        name = "Local density (k = 5)")
  ) +
  labs(x = NULL)

# ---- (c) spectral: effective rank + PC1 variance (K=128) ----
dfc <- data.frame(
  layer = c("L0", "L1", "L2"),
  rank = c(15.65, 22.17, 23.64),
  pc1 = c(20.54, 8.36, 6.33)
)
dfc$layer <- factor(dfc$layer, levels = c("L0", "L1", "L2"))

pc <- ggplot(dfc, aes(x = layer)) +
  geom_line(aes(y = rank, group = 1), colour = TEAL, linewidth = 0.7) +
  geom_point(aes(y = rank), colour = TEAL, size = 1.7) +
  geom_line(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2), group = 1),
            colour = BLACK, linewidth = 0.7, linetype = "dotted") +
  geom_point(aes(y = 13 + (pc1 - 6.33) * (26 - 13) / (25 - 2)),
             colour = BLACK, size = 1.7, shape = 17) +
  scale_y_continuous(
    name = "Effective rank",
    limits = c(13, 26),
    sec.axis = sec_axis(~ 6.33 + (. - 13) * (25 - 2) / (26 - 13),
                        name = "Var. explained by PC1 (%)")
  ) +
  labs(x = NULL)

fig <- (pa | pb | pc) + plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9, face = "bold"),
        plot.tag.position = c(0, 1))

save_pub_r <- function(plot, filename, width_mm = 183, height_mm = 58, dpi = 600) {
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

save_pub_r(fig, "codebook_geometry_3panel")
