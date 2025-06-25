library(dplyr)
library(nlme)
library(ggplot2)
data = read.csv("20_pixels_per_map_data.csv", stringsAsFactors = T)
data$se = data$errors^2
#analysis of slope sd
slope_sd_data = data %>% filter(feature=="slope")

ggplot(slope_sd_data, aes(x = size, y = se, color = value)) +
  geom_point() +
  labs(x = "Size", y = "SE", color = "Value") +
  theme_minimal()

# Step 1: Create 4 groups based on quartiles of 'value'
slope_sd_data <- slope_sd_data %>%
  mutate(value_group = ntile(value, 4)) %>%          # creates 4 groups numbered 1 to 4
  mutate(value_group = paste("Q", value_group, sep=""))  # label as Q1, Q2, Q3, Q4

# Step 2: Plot with separate smooths for each group
ggplot(slope_sd_data, aes(x = size, y = se, color = value_group)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE) +
  labs(x = "Size", y = "Errors", color = "Value Group") +
  theme_minimal()


hist(slope_sd_data$se)

library(mgcv)
model = bam(se~s(size, k=8) + value + s(Map, bs='re'), data = slope_sd_data, family = Gamma(link="log"))
model1 = bam(se~s(size, value)+ s(Map, bs='re'), data = slope_sd_data, family = Gamma(link="log"))
model2 = bam(se~s(size, k=8) + s(value)+ s(Map, bs='re'), data = slope_sd_data, family = Gamma(link="log"))
AIC(model, model1, model2)
summary(model1)


r = simulateResiduals(model1)
plot(r)
plot(r$scaledResiduals~slope_sd_data$size)
plot(r$scaledResiduals~slope_sd_data$value)
plot(r$fittedResiduals~slope_sd_data$size)
plot(r$fittedResiduals~slope_sd_data$value)
testSpatialAutocorrelation(r, x = slope_sd_data$x, y = slope_sd_data$y)

model1_car = bam(se~s(size, value)+ s(Map, bs='re') + s(x, y), data = slope_sd_data, family = Gamma(link="log"))
summary(model1_car)

AIC(model1, model1_car)

p = predict_response(model1_car, terms = c("size", "value"))
#plot(p, show_data=T, jitter = 1, dot_alpha = 0.25)
plot(p, dot_alpha = 0.25)
