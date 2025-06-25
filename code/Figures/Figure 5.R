#TGI - how greeness coordinates need more spatial information
#skyview - how more covered coordinates need more spatial information
#solar - how more solar exposed coordinates need more spatial information
#slope_sd - how coordinates with more nearby changes in the ground might need more spatial information - this also include changes in height so we didn't analyze height.


load(file = "predicted_solar_sd.RData")
load(file = "predicted_skyview.RData")
load(file = "predicted_TGI.RData")
load(file = "predicted_solar.RData")

library(ggplot2)
library(rlang) # for the .data pronoun if needed

plot_grouped_lines <- function(
    data, 
    x, y, ymin, ymax, group_col,
    colors = NULL,
    xlab = NULL, ylab = NULL,
    legend_title = NULL
) {
  # If not already a factor, convert group_col to factor (no relabeling)
  if (!is.factor(data[[deparse(substitute(group_col))]])) {
    data[[deparse(substitute(group_col))]] <- factor(data[[deparse(substitute(group_col))]])
  }
  
  # Default: assign three distinct colors if not supplied
  if (is.null(colors)) {
    palette <- c("#4575b4", "#fdae61", "#d73027")
    levels_group <- levels(data[[deparse(substitute(group_col))]])
    colors <- setNames(palette[seq_along(levels_group)], levels_group)
  }
  
  ggplot(data, aes(x = {{x}}, y = {{y}}, color = {{group_col}}, group = {{group_col}})) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymin = {{ymin}}, ymax = {{ymax}}, fill = {{group_col}}), 
                alpha = 0.2, color = NA) +
    scale_color_manual(values = colors) +
    scale_fill_manual(values = colors) +
    labs(
      x = xlab %||% as_label(enquo(x)),
      y = ylab %||% as_label(enquo(y)),
      color = legend_title %||% as_label(enquo(group_col)),
      fill = legend_title %||% as_label(enquo(group_col))
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      legend.margin = margin(t = 0, r = 0, b = 0, l = 0),   # Reduce space around legend
      legend.box.margin = margin(t = 0, r = 0, b = 0, l = 0),# Reduce space around legend box
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2),      # Reduce outer plot margin
      axis.title.x = element_text(margin = margin(t = 5)),   # Shrink axis title spacing
      axis.title.y = element_text(margin = margin(r = 5)),   # Shrink axis title spacing
      panel.spacing = unit(0, "lines"),                    # Space between panels (for facets)
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
}

p_slope = plot_grouped_lines(
  data = df_p_slope,  x = size,  y = predicted,  ymin = conf.low,  
  ymax = conf.high,  group_col = slope_SD,  xlab = "",
  ylab = "",  legend_title = "Slope SD (°)")

p_solar = plot_grouped_lines(
  data = df_p_solar,  x = size,  y = predicted,  ymin = conf.low,  
  ymax = conf.high,  group_col = solar,  xlab = "",
  ylab = "",  legend_title = expression("Solar Radiation (W m"^{-2}*")"))

p_TGI = plot_grouped_lines(
  data = df_p_TGI,  x = size,  y = predicted,  ymin = conf.low,  
  ymax = conf.high,  group_col = TGI,  xlab = "",
  ylab = "",  legend_title = "TGI")

p_skyview = plot_grouped_lines(
  data = df_p_skyview,  x = size,  y = predicted,  ymin = conf.low,  
  ymax = conf.high,  group_col = skyview,  xlab = "",
  ylab = "",  legend_title = "Skyview (%)")

library(ggpubr)

# Arrange with individual legends
combined_plot <- ggarrange(
  p_solar, p_TGI, p_slope, p_skyview,
  nrow = 2, ncol = 2,labels = c("(a)", "(b)", "(c)", "(d)"), label.x = 0.2, label.y = 0.85,
  common.legend = FALSE, align = "hv",
  font.label = list(size = 12, face = "plain")
)

final_plot <- annotate_figure(
  combined_plot,
  left = text_grob("Square Error", rot = 90, size = 14, vjust = 1.5),
  bottom = text_grob("Size", size = 14, vjust = -0.5), 
)

print(final_plot)

ggsave("Figure 5.png", final_plot, dpi = 300, width=2500, height = 2000, units = "px", bg = "white")
