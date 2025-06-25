library(mgcv)
library(dplyr)
library(ggplot2)
library(ggeffects)

options(ggeffects_margin = "empirical")
data = read.csv("20_pixels_per_map_data.csv", stringsAsFactors = T)
data$se = data$errors^2

# solar radiation
#effect of solar radiation and size - see points_stats_solar.R for model selection
solar_data = data %>% filter(feature=="real_solar")
model_solar = bam(se~s(size, value)+ s(Map, bs='re') + s(x, y), data = solar_data, family = Gamma(link="log"))
summary(model_solar)
p = predict_response(model_solar, terms = c("size", "value[80, 400, 800]"))
df_p_solar =as.data.frame(p)
names(df_p_solar) = c("size", "predicted", "std.error", "conf.low", "conf.high", "solar")
save(df_p_solar, file = "predicted_solar.RData")

# TGI
#effect of solar radiation and size - see points_stats_solar.R for model selection
TGI_data = data %>% filter(feature=="TGI")
model_TGI = bam(se~s(size, k=8) + s(value)+ s(Map, bs='re') + s(x, y), data = TGI_data, family = Gamma(link="log"))
summary(model_TGI)
p = predict_response(model_TGI, terms = c("size", "value[-0.03, 0, 0.03]"))
df_p_TGI =as.data.frame(p)
names(df_p_TGI) = c("size", "predicted", "std.error", "conf.low", "conf.high", "TGI")
save(df_p_TGI, file = "predicted_TGI.RData")

# skyview
#effect of solar radiation and size - see points_stats_solar.R for model selection
skyview_data = data %>% filter(feature=="skyview")
model_skyview = bam(se~s(size, value)+ s(Map, bs='re') + s(x, y), data = skyview_data, family = Gamma(link="log"))
summary(model_skyview)
p = predict_response(model_skyview, terms = c("size", "value[0.2, 0.5, 0.8]"))
df_p_skyview =as.data.frame(p)
names(df_p_skyview) = c("size", "predicted", "std.error", "conf.low", "conf.high", "skyview")
save(df_p_skyview, file = "predicted_skyview.RData")


#effect of solar radiation and size - see points_stats_solar.R for model selection
slope_sd_data = data %>% filter(feature=="slope")
model_slope_sd = bam(se~s(size, value)+ s(Map, bs='re') + s(x, y), data = slope_sd_data, family = Gamma(link="log"))
summary(model_slope_sd)
p = predict_response(model_slope_sd, terms = c("size", "value [0.5, 5, 9.5]"), condition = c(x = mean(slope_sd_data$x), y = mean(slope_sd_data$y)))
df_p_slope =as.data.frame(p)
names(df_p_slope) = c("size", "predicted", "std.error", "conf.low", "conf.high", "slope_SD")
save(df_p_slope, file = "predicted_solar_sd.RData")

