# Load necessary libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)

#get the name of files with the errors of each model. These files were created by the "/Users/ofir/Dropbox/pycharm_projects/Sulami_et_al_Ecology/code/MicroclimateModelAnalysis.py" file
files = dir("~/Dropbox/pycharm_projects/Sulami_et_al_Ecology", pattern = "error_maps_model_size.*.csv", full.names = T)


# Read each CSV into a list of data frames
df_list <- lapply(files, read.csv, stringsAsFactors = FALSE)

# Row‑bind them all together
all_errors <- do.call(rbind, df_list)

# organize the dataset
all_errors$factor_size = as.factor(all_errors$model_size)
all_errors = all_errors[all_errors$microhabitat!="all",]
all_errors$microhabitat = as.factor(all_errors$microhabitat)
all_errors$flight = as.factor(all_errors$flight)

# add an hour column
all_errors_hours <- all_errors %>%
  separate(flight,
           into = c("site", "date", "time", "idx"),
           sep   = "_") %>%
  mutate(
    # make sure 'time' is zero‑padded to 4 chars, e.g. "610" → "0610"
    time4     = str_pad(as.character(time), width = 4, side = "left", pad = "0"),
    # turn "0610" → "06:10"
    time_colon = str_replace(time4, "^(\\d{2})(\\d{2})$", "\\1:\\2"),
    # now parse the H:M string into a lubridate Period
    dt_period = hm(time_colon),
    # if you also want a full datetime, parse your date and add:
    date       = dmy(date),
    datetime   = date + dt_period,
    # round to the nearest hour
    datetime_hr = hour(round_date(datetime, unit = "hour"))
  )

# add daypart column
all_errors_hours = all_errors_hours %>%
  mutate(time_class = case_when(datetime_hr %in% c(6, 7) ~ "morning", datetime_hr %in% c(18, 17) ~ "evening",
                           datetime_hr >= 8  & datetime_hr <= 16   ~ "midday",
                           TRUE                      ~ NA_character_
  ))

# organize the new dataset
all_errors_hours$flight = all_errors$flight
all_errors_hours$time_class = as.factor(all_errors_hours$time_class)
all_errors_hours$microhabitat = as.factor(all_errors$microhabitat)
all_errors_hours$microhabitat_time = as.factor(paste(all_errors_hours$microhabitat, all_errors_hours$time_class))

library(mgcv)

model_micro_mse_time_1 = bam(MSE~s(model_size, k=8) + s(model_size, k=8, by=as.numeric(time_class=="morning")) + s(model_size, k=8, by=as.numeric(time_class=="evening")) + s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_2 = bam(MSE~s(model_size, k=8, by = time_class) + s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_3 = bam(MSE~s(model_size, k=8, by = time_class) + microhabitat+ s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_4 = bam(MSE~s(model_size, k=8, by = time_class) + microhabitat*time_class+ s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_5 = bam(MSE~s(model_size, k=8, by = time_class) + microhabitat+time_class+ s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_6 = bam(MSE~s(model_size, k=8, by = time_class) + s(model_size, k=8, by = microhabitat)+ s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_7 = bam(MSE~s(model_size, k=8, by = microhabitat_time) + s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_8 = bam(MSE~s(model_size, k=8) + time_class + s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)
model_micro_mse_time_9 = bam(MSE~ model_size*time_class+ s(flight, bs='re'), family = Gamma(link=log), data = all_errors_hours)

AIC(model_micro_mse_time_1, model_micro_mse_time_2, model_micro_mse_time_3, 
    model_micro_mse_time_4, model_micro_mse_time_5, model_micro_mse_time_6, 
    model_micro_mse_time_7, model_micro_mse_time_8, model_micro_mse_time_9)

summary(model_micro_mse_time_4)

library(DHARMa)
e = simulateResiduals(model_micro_mse_time_4)
plot(e)
plot(e$scaledResiduals~all_errors_hours$time_class)
library(ggeffects)
p = predict_response(model_micro_mse_time_4, terms = c("model_size", "microhabitat", "time_class"), margin = "empirical")
plot(p, show_data = T, jitter = 1) 
as.data.frame(p)

df_p = as.data.frame(p)
names(df_p) = c("model_size", "predicted", "std.error", "conf.low", "conf.high", "microhabitat", "time_class")

# Plot
library(ggplot2)

# Make a panel label for each facet
df_p$panel_letter <- c("A", "B", "C")[as.numeric(as.factor(df_p$time_class))]
all_errors_hours$panel_letter <- c("A", "B", "C")[as.numeric(as.factor(all_errors_hours$time_class))]

# Use the median x and maximum y for placing the label
panel_positions <- df_p %>%
  group_by(time_class) %>%
  summarise(
    x = min(model_size, na.rm = TRUE) + 0.1, # near left edge, adjust as needed
    y = max(predicted, na.rm = TRUE),         # near top
    panel_letter = unique(panel_letter)
  )

make_panel_plot <- function(df_pred, df_err) {
  ggplot(df_pred, aes(x = model_size, y = predicted, color = microhabitat, group = microhabitat)) +
    geom_point(
      data = df_err, 
      aes(x = model_size, y = MSE, color = microhabitat), 
      position = position_jitter(width = 1), 
      size = 2, alpha = 0.5
    ) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = microhabitat), 
                alpha = 0.2, color = NA, show.legend = FALSE) +
    scale_color_manual(values = c("open" = "orange", "shade" = "black")) +
    scale_fill_manual(values = c("open" = "orange", "shade" = "black")) +
    ylim(0, 60) +
    labs(x = NULL, y = NULL, color = "Microhabitat", fill = "Microhabitat") +  # No axis labels here
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black"),
      plot.title = element_blank(),
      legend.position = "none"
    )
}

time_classes <- unique(df_p$time_class)

# Make sure to keep the desired order
panel1_df_pred <- df_p %>% filter(time_class == time_classes[1])
panel1_df_err  <- all_errors_hours %>% filter(time_class == time_classes[1])

panel2_df_pred <- df_p %>% filter(time_class == time_classes[2])
panel2_df_err  <- all_errors_hours %>% filter(time_class == time_classes[2])

panel3_df_pred <- df_p %>% filter(time_class == time_classes[3])
panel3_df_err  <- all_errors_hours %>% filter(time_class == time_classes[3])

plot1 <- make_panel_plot(panel1_df_pred, panel1_df_err)
plot2 <- make_panel_plot(panel2_df_pred, panel2_df_err)
plot3 <- make_panel_plot(panel3_df_pred, panel3_df_err)

library(ggpubr)
# Define your desired panel labels in the correct order
panel_labels <- c("(a) morning", "(b) noon  ", "(c) evening")

# Combine plots with custom panel labels
final_plot <- ggarrange(
  plot1, plot2, plot3, 
  nrow = 1, ncol = 3,
  common.legend = TRUE, legend = "top",
  labels = panel_labels,
  font.label = list(size = 12, face = "plain"),  # size 12
  label.x = 0.35,  # tweak as needed for position
  label.y = 0.95, hjust = 0
)

# Add global axis labels
final_plot <- annotate_figure(
  final_plot,
  left = text_grob("Mean Square Error", rot = 90, vjust = 1, size = 14),
  bottom = text_grob("Tile Size (pixels)", size = 14)
)

print(final_plot)

ggsave("Figure_4.png", final_plot, dpi = 300, width=2500, height = 1000, units = "px", bg = "white")
