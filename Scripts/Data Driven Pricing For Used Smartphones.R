# Load necessary libraries
library(dplyr)
library(ggplot2)
library(psych)
library(caret)
library(corrplot)
library(car)
library(rpart)
library(rpart.plot)
library(RWeka)

# Load the dataset
device.df <- read.csv("used_device_data.csv")
str(device.df)

# Convert Categorical attributes into factors
device.df$device_brand <- as.factor(device.df$device_brand)
device.df$os <- as.factor(device.df$os)
device.df$X4g <- as.factor(device.df$X4g)
device.df$X5g <- as.factor(device.df$X5g)

# Checking levels in the factor to know if 34 levels will be an issue
levels(device.df$device_brand)
table(device.df$device_brand)

# Left the device brand as a factor for now lets revert to it later. 
str(device.df)

# check for missing values
colSums(is.na(device.df))

# Handling Missing Values by imputing with the median grouping by device_brand
device.df <- device.df %>%
  group_by(device_brand) %>%
  mutate(
    rear_camera_mp = ifelse(is.na(rear_camera_mp),
                            median(rear_camera_mp, na.rm = TRUE),
                            rear_camera_mp),
    front_camera_mp = ifelse(is.na(front_camera_mp),
                             median(front_camera_mp, na.rm = TRUE),
                             front_camera_mp),
    internal_memory = ifelse(is.na(internal_memory),
                             median(internal_memory, na.rm = TRUE),
                             internal_memory),
    ram = ifelse(is.na(ram),
                 median(ram, na.rm = TRUE),
                 ram),
    battery = ifelse(is.na(battery),
                     median(battery, na.rm = TRUE),
                     battery),
    weight = ifelse(is.na(weight),
                    median(weight, na.rm = TRUE),
                    weight)
  ) %>%
  ungroup() %>%
  group_by(os, release_year) %>%
  mutate(
    rear_camera_mp = ifelse(is.na(rear_camera_mp),
                            median(rear_camera_mp, na.rm = TRUE),
                            rear_camera_mp),
    front_camera_mp = ifelse(is.na(front_camera_mp),
                             median(front_camera_mp, na.rm = TRUE),
                             front_camera_mp),
    internal_memory = ifelse(is.na(internal_memory),
                             median(internal_memory, na.rm = TRUE),
                             internal_memory),
    ram = ifelse(is.na(ram),
                 median(ram, na.rm = TRUE),
                 ram),
    battery = ifelse(is.na(battery),
                     median(battery, na.rm = TRUE),
                     battery),
    weight = ifelse(is.na(weight),
                    median(weight, na.rm = TRUE),
                    weight)
  ) %>%
  ungroup() 

# Checking to see if NA's were handled properly
colSums(is.na(device.df))

# check for zero values
zero_check <- sapply(device.df, function(x) sum(x == 0, na.rm = TRUE))
zero_check

# 0's only present in front_camera_mp so we are leaving it for now cause they could be meaningful 0'S

# summary statistics
summary(device.df)
describe(device.df)

# Distribution exploration 
num_vars <- c("screen_size", "rear_camera_mp", "front_camera_mp",
              "internal_memory", "ram", "battery", "weight",
              "days_used", "normalized_used_price", "normalized_new_price")

for (var in num_vars) {
  print(
    ggplot(device.df, aes_string(x = var)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      theme_minimal() +
      labs(title = paste("Distribution of", var), x = var, y = "Count")
  )
}

# Boxplots for outliers detection
for (var in num_vars) {
  print(
    ggplot(device.df, aes_string(y = var)) +
      geom_boxplot(fill = "tomato") +
      theme_minimal() +
      labs(title = paste("Boxplot of", var), y = var)
  )
}

# Handling Outliers 

# Creating a new column to flag rear camera inconsistencies on manual inspection
device.df$rear_camera_flag <- ifelse(
  device.df$rear_camera_mp < 0.1 |
    (device.df$rear_camera_mp < 3 & 
       (device.df$internal_memory > 4 | device.df$ram > 1)),
  "Inconsistent",
  "Plausible"
)

# Replace inconsistent rear_camera_mp values with NA
device.df$rear_camera_mp[device.df$rear_camera_flag == "Inconsistent"] <- NA


# Impute the inconsistent values median value by grouping with brand for rear camera 
device.df <- device.df %>%
  group_by(device_brand) %>%
  mutate(rear_camera_mp = ifelse(is.na(rear_camera_mp),
                                 median(rear_camera_mp, na.rm = TRUE),
                                 rear_camera_mp)) %>%
  ungroup()

# Check to see the imputation of the impossible values is done correctly.
sum(is.na(device.df$rear_camera_mp))

# Creating a new column to flag front camera inconsistencies on manual inspection

device.df$front_camera_flag <- ifelse(
  device.df$front_camera_mp > device.df$rear_camera_mp |
    (device.df$front_camera_mp < 2 & 
       (device.df$internal_memory > 4 | device.df$ram > 1 | device.df$rear_camera_mp >= 5)),
  "Inconsistent",
  "Plausible"
)

# Replace inconsistent front_camera_mp values with NA
device.df$front_camera_mp[device.df$front_camera_flag == "Inconsistent"] <- NA

# Impute the inconsistent values with median value by grouping with release year for front camera 
device.df <- device.df %>%
  group_by(release_year) %>%
  mutate(front_camera_mp = ifelse(is.na(front_camera_mp),
                                 median(front_camera_mp, na.rm = TRUE),
                                 front_camera_mp)) %>%
  ungroup()

sum(is.na(device.df$front_camera_mp))

# Creating a new column to flag Internal memory inconsistencies on manual inspection
device.df$internal_memory_flag <- ifelse(
  device.df$internal_memory < device.df$ram | 
    (device.df$internal_memory > 64 &
       (device.df$release_year < 2015)) |
    (device.df$internal_memory < 1 & 
       (device.df$rear_camera_mp > 5 | device.df$front_camera_mp > 2 | device.df$ram > 0.5)),
  "Inconsistent",
  "Plausible"
)

# Replace inconsistent internal memory values with NA
device.df$internal_memory[device.df$internal_memory_flag == "Inconsistent"] <- NA

# Impute the inconsistent values with median value by grouping with brand for internal memory 
device.df <- device.df %>%
  group_by(device_brand) %>%
  mutate(internal_memory = ifelse(is.na(internal_memory),
                                  median(internal_memory, na.rm = TRUE),
                                  internal_memory)) %>%
  ungroup()

sum(is.na(device.df$internal_memory))

# Creating a new column to flag Ram inconsistencies on manual inspection
device.df$ram_flag <- ifelse(
  device.df$ram > device.df$internal_memory|
    (device.df$ram < 1 &
       (device.df$internal_memory >= 4 | device.df$rear_camera_mp > 5 )),
  "Inconsistent",
  "Plausible"
)

# Replace inconsistent ram values with NA
device.df$ram[device.df$ram_flag == "Inconsistent"] <- NA

# Impute the inconsistent values with median value by grouping with release year for ram
device.df <- device.df %>%
  group_by(release_year) %>%
  mutate(ram = ifelse(is.na(ram),
                      median(ram, na.rm = TRUE),
                      ram)) %>%
  ungroup()

sum(is.na(device.df$ram))

# Creating a new column to flag weight inconsistencies on manual inspection
device.df$weight_flag <- ifelse(
  device.df$weight > 350 & device.df$screen_size < 15,
  "Inconsistent",
  "Plausible"
)

# Replace inconsistent weight values with NA
device.df$weight[device.df$weight_flag == "Inconsistent"] <- NA

# Impute the inconsistent values with median value by grouping with brand for ram
device.df <- device.df %>%
  group_by(device_brand) %>%
  mutate(weight = ifelse(is.na(weight),
                      median(weight, na.rm = TRUE),
                      weight)) %>%
  ungroup()

sum(is.na(device.df$ram))

# Recheck the dataset for missing values and 0's 
colSums(is.na(device.df))
zero_check <- sapply(device.df, function(x) sum(x == 0, na.rm = TRUE))
zero_check

# Create a new variable called price tier to classify the devices based on their new price into low medium and high
median_price <- median(device.df$normalized_new_price, na.rm = TRUE)
device.df$price_tier <- ifelse(device.df$normalized_new_price <= median_price,"Low", "High")
device.df$price_tier <- as.factor(device.df$price_tier)

# Partition Data for training and testing
set.seed(2048)
train_index <- createDataPartition(device.df$normalized_used_price, p = 0.7, list = FALSE)
train_data <- device.df[train_index,]
test_data <- device.df[-train_index,]

# Remove flags from the test and training data
train_data <- train_data %>%
  select(-rear_camera_flag, -front_camera_flag, -internal_memory_flag, -ram_flag, -weight_flag)
test_data <- test_data %>%
  select(-rear_camera_flag, -front_camera_flag, -internal_memory_flag, -ram_flag, -weight_flag)

# Modelling
# Linear Regression
# correlation check
numeric_vars <- device.df %>%
  select(screen_size, rear_camera_mp, front_camera_mp, internal_memory,
         ram, battery, weight, days_used, release_year)

cor_matrix <- cor(numeric_vars, use = "complete.obs")
print(round(cor_matrix, 2))
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Linear Regression (Formula was based on correlation)
linear_model <- lm(normalized_used_price ~ screen_size + rear_camera_mp + front_camera_mp + 
     internal_memory + ram + days_used + release_year +
     device_brand + os + X4g + X5g, data = train_data)

summary(linear_model)
VIF_values <- vif(linear_model)
print(VIF_values)

# Predictions
lm_predictions <- predict(linear_model, newdata = test_data)

# Prediction evaluation metrics
test_rmse_lm  <- sqrt(mean((lm_predictions - test_data$normalized_used_price)^2))
test_mae_lm  <- mean(abs(lm_predictions - test_data$normalized_used_price))
test_rmse_lm 
test_mae_lm

# Regression Tree 
set.seed(123)
reg_tree <- rpart( normalized_used_price ~ ., data = train_data, method = "anova")

# Visualize the tree
rpart.plot(reg_tree, main = "Regression Tree for Used Price", cex = 0.9)

summary(reg_tree)

# Tree Pruning to account for overfitting
print(reg_tree$cptable)

# Choose optimal CP (one that minimizes xerror)
optimal_cp <- reg_tree$cptable[which.min(reg_tree$cptable[,"xerror"]),"CP"]

# Prune tree with optimal CP
pruned_tree <- prune(reg_tree, cp = optimal_cp)

# Plot pruned tree
rpart.plot(pruned_tree, main = "Pruned Regression Tree")

tree_predictions <- predict(pruned_tree, newdata = test_data)

# Test RMSE
test_rmse_tree <- sqrt(mean((tree_predictions - test_data$normalized_used_price)^2))
test_rmse_tree

# Test MAE
test_mae_tree <- mean(abs(tree_predictions - test_data$normalized_used_price))
test_mae_tree

# Important Predictors
pruned_tree$variable.importance

# Visual Plot for Important predictors
var_imp <- data.frame(
  Variable = names(pruned_tree$variable.importance),
  Importance = pruned_tree$variable.importance
)

# Plot
ggplot(var_imp, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance in Regression Tree",
       x = "Variable",
       y = "Importance Score")


# Model Tree
model_tree <- M5P(normalized_used_price ~ ., data = train_data)

# Summary of model tree
summary(model_tree)
print(model_tree)

# Predictions 
model_predictions <- predict(model_tree, newdata = test_data)

# Evaluate model performance
test_rmse_tree <- sqrt(mean((model_predictions - test_data$normalized_used_price)^2))
test_mae_tree <- mean(abs(model_predictions - test_data$normalized_used_price))

test_rmse_tree
test_mae_tree


###################################################################################

# Classification Modelling

# Fit the Tree 
class_tree <- rpart(price_tier ~ screen_size + rear_camera_mp + front_camera_mp +
  internal_memory + ram + battery + weight +
  release_year + days_used + device_brand + os + X4g + X5g,
data = train_data,
method = "class"
)

# Plot the tree
rpart.plot(class_tree, main = "Classification Tree for Price Tier", cex = 0.9)

# Predictions on test data
class_tree_predictions <- predict(class_tree, newdata = test_data, type = "class")

# Confusion matrix and metrics
conf_matrix_tree <- confusionMatrix(class_tree_predictions, test_data$price_tier)
conf_matrix_tree

# Variable importance
var_imp_tree <- data.frame(
  Variable = names(class_tree$variable.importance),
  Importance = class_tree$variable.importance
)

# Plot Variable Importance
ggplot(var_imp_tree, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance in Classification Tree",
       x = "Variable",
       y = "Importance Score")

