library(dplyr)
library(nlme)
library(ggplot2)
data = read.csv("20_pixels_per_map_data.csv", stringsAsFactors = T)
data$se = data$errors^2
#analysis of solar radiation
solar_data = data %>% filter(feature=="real_solar")

ggplot(solar_data, aes(x = size, y = se, color = value)) +
  geom_point() +
  labs(x = "Size", y = "Errors", color = "Value") +
  theme_minimal()

# Step 1: Create 4 groups based on quartiles of 'value'
solar_data <- solar_data %>%
  mutate(value_group = ntile(value, 4)) %>%          # creates 4 groups numbered 1 to 4
  mutate(value_group = paste("Q", value_group, sep=""))  # label as Q1, Q2, Q3, Q4

# Step 2: Plot with separate smooths for each group
ggplot(solar_data, aes(x = size, y = se, color = value_group)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE) +
  labs(x = "Size", y = "Errors", color = "Value Group") +
  theme_minimal()

hist(abs(solar_data$se))

library(mgcv)
model = bam(se~s(size, k=8) + value + s(Map, bs='re'), data = solar_data, family = Gamma(link="log"))
model1 = bam(se~s(size, value)+ s(Map, bs='re'), data = solar_data, family = Gamma(link="log"))
model2 = bam(se~s(size, k=8) + s(value)+ s(Map, bs='re'), data = solar_data, family = Gamma(link="log"))
AIC(model, model1, model2)
summary(model1)


r = simulateResiduals(model1)
plot(r)
plot(r$scaledResiduals~solar_data$size)
plot(r$scaledResiduals~solar_data$value)
plot(r$fittedResiduals~solar_data$size)
plot(r$fittedResiduals~solar_data$value)
testSpatialAutocorrelation(r, x = solar_data$x, y = solar_data$y)

model1_car = bam(se~s(size, value)+ s(Map, bs='re') + s(x, y), data = solar_data, family = Gamma(link="log"))
summary(model1_car)

AIC(model1, model1_car)

p = predict_response(model1_car, terms = c("size", "value"))
plot(p, show_data=T, jitter = 1, alpha = 0.4)
