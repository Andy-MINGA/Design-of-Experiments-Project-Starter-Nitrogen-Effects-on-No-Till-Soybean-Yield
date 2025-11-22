# Load necessary packages
# install.packages("lme4")
# install.packages("ggplot2")
# install.packages("performance")

library(lme4)
library(ggplot2)

install.packages("performance")
library(performance)  # for diagnostic checks
library(car)       # for Levene's test

# Load CSV Data
df <- read.csv("C:/Users/EGMC/Desktop/Fall 2025 Classes/Desin of Experiments/Project/NEW DATA/AllenEtal.StarterN.allplots.csv",
               stringsAsFactors = TRUE)

# Check first rows
head(df)

str(df)         # Check variable types
table(df$block) # Check block levels
table(df$treatment) # Check treatment levels
table(df$siteyr) # Check site-year levels


library(lme4)

# Mixed-effects model
model_mixed <- lmer(yield.kgha ~ treatment + (1 | siteyr) + (1 | siteyr:block), data = df)
summary(model_mixed)


# ------------------------------
# 3. Fit the mixed-effects model
# ------------------------------
# Model: yield.kgha ~ treatment + (1 | siteyr) + (1 | siteyr:block)
# Explanation:
# - yield.kgha: response variable (crop yield)
# - treatment: fixed effect (the main factor we want to test)
# - (1 | siteyr): random effect of site-year (accounts for environmental differences)
# - (1 | siteyr:block): random effect of blocks nested within site-year
model_mixed <- lmer(yield.kgha ~ treatment + (1 | siteyr) + (1 | siteyr:block), data = df)

# Inspect the model summary
summary(model_mixed)

# ------------------------------
# 4. Extract residuals and fitted values
# ------------------------------
res <- resid(model_mixed)      # residuals
fitted_vals <- fitted(model_mixed)  # fitted values

# ------------------------------
# 5. Check linearity and homoscedasticity
# ------------------------------
# Residuals vs Fitted plot
# Random scatter around zero indicates linearity and constant variance
plot(fitted_vals, res,
     xlab = "Fitted values",
     ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, col = "red")

# ------------------------------
# 6. Check normality of residuals
# ------------------------------
# Histogram
hist(res, main = "Histogram of Residuals", xlab = "Residuals")

# Q-Q plot
qqnorm(res, main = "Q-Q Plot of Residuals")
qqline(res, col = "red")

# Shapiro-Wilk test for normality
shapiro.test(res)  # p > 0.05 suggests approximate normality

# ------------------------------
# 7. Check random effects
# ------------------------------
# Random effects for site-year
ranef_siteyr <- ranef(model_mixed)$siteyr[,1]
qqnorm(ranef_siteyr, main = "Q-Q Plot: Site-Year Random Effects")
qqline(ranef_siteyr, col = "red")

# Random effects for blocks nested within site-year
ranef_block <- ranef(model_mixed)$`siteyr:block`[,1]
qqnorm(ranef_block, main = "Q-Q Plot: Block Random Effects")
qqline(ranef_block, col = "red")


# Now, we get the results in terms of p-values
# We install fist the package
install.packages("lmerTest", repos = "https://cloud.r-project.org/")

#Load the package
library(lmerTest)

# Refit the model using lmerTest
model_mixed <- lmer(yield.kgha ~ treatment +
                      (1 | siteyr) +
                      (1 | siteyr:block),
                    data = df)
summary(model_mixed)





# -----------------------------
# Simulate RCBD experiment dataset
# -----------------------------
# Goal: Mimic real experimental design with specific site-years, blocks, and treatments
# without using real fitted values.

# Load necessary libraries
library(dplyr)
library(lme4)

# Set seed for reproducibility
set.seed(123)

# -----------------------------
# 1. Define the experiment structure
# -----------------------------
siteyrs <- c("NY20","NY21","NY22","WI21","WI22")    # 5 site-years
blocks_per_site <- c(4,5,5,4,4)                      # Number of blocks per site-year
treatments <- c("Control","FM","PL","SN")            # 4 treatments per block

# -----------------------------
# 2. Define hypothetical parameters
# -----------------------------
intercept <- 2600                     # Baseline yield (kg/ha)
treatment_effects <- c(Control=0, FM=50, PL=75, SN=100)  # Hypothetical treatment effects
sigma_site <- 500                      # SD for site-year random effect
sigma_block <- 50                      # SD for block random effect
sigma_resid <- 400                     # SD for residual error

# -----------------------------
# 3. Initialize empty dataframe
# -----------------------------
sim_data <- data.frame()

# -----------------------------
# 4. Loop over site-years, blocks, and treatments to simulate yields
# -----------------------------
for(i in 1:length(siteyrs)){
  site <- siteyrs[i]                  # Current site-year
  n_blocks <- blocks_per_site[i]      # Number of blocks in current site-year
  
  # Simulate random effect for site-year
  site_effect <- rnorm(1, mean=0, sd=sigma_site)
  
  # Loop over blocks in current site-year
  for(b in 1:n_blocks){
    # Simulate random effect for block nested within site-year
    block_effect <- rnorm(1, mean=0, sd=sigma_block)
    
    # Loop over treatments in current block
    for(trt in treatments){
      # Simulate residual error for individual observation
      resid <- rnorm(1, mean=0, sd=sigma_resid)
      
      # Calculate simulated yield
      yield <- intercept + treatment_effects[trt] + site_effect + block_effect + resid
      
      # Add observation to the dataframe
      sim_data <- rbind(sim_data, data.frame(
        siteyr = site,
        block = b,
        treatment = trt,
        yield.kgha = yield
      ))
    }
  }
}

# -----------------------------
# 5. Convert variables to factors
# -----------------------------
sim_data$siteyr <- as.factor(sim_data$siteyr)
sim_data$block <- as.factor(sim_data$block)
sim_data$treatment <- as.factor(sim_data$treatment)

# -----------------------------
# 6. Preview simulated dataset
# -----------------------------
head(sim_data)

# -----------------------------
# 7. Save to CSV
# -----------------------------
write.csv(sim_data, file = "simulated_RCBD_data.csv", row.names = FALSE)


# =========================================================
# CHECKING MODEL ASSUMPTIONS FOR SIMULATED RCBD DATA
# =========================================================

# Load necessary library
library(lme4)   # For fitting linear mixed-effects models

# -----------------------------
# 1. Fit the mixed-effects model
# -----------------------------
# Fixed effect: treatment
# Random effects: site-year and block nested within site-year
model_sim <- lmer(
  yield.kgha ~ treatment + (1 | siteyr) + (1 | siteyr:block),
  data = sim_data
)

# -----------------------------
# 2. Extract residuals and fitted values
# -----------------------------
resid_sim <- residuals(model_sim)       # Residuals from the model
fitted_sim <- fitted(model_sim)         # Predicted/fitted values

# -----------------------------
# 3. Check residual normality
# -----------------------------
# Q-Q plot: residuals should approximately follow a straight line
qqnorm(resid_sim, main="Q-Q Plot of Residuals")
qqline(resid_sim, col="red")   # Add reference line

# Histogram for residuals (optional)
hist(resid_sim, breaks=15, main="Histogram of Residuals", xlab="Residuals")

# -----------------------------
# 4. Check homoscedasticity (constant variance)
# -----------------------------
# Plot residuals vs fitted values
plot(fitted_sim, resid_sim,
     main="Residuals vs Fitted",
     xlab="Fitted Values",
     ylab="Residuals")
abline(h=0, col="red")  # Reference line at 0

# -----------------------------
# 5. Check random effects normality
# -----------------------------

# 5a. Site-year random effects
ranef_site <- ranef(model_sim)$siteyr[,1]  # Extract site-year random effects
qqnorm(ranef_site, main="Q-Q Plot: Site-Year Random Effects")
qqline(ranef_site, col="red")




# -----------------------------
# Pairwise Comparisons of Treatments
# -----------------------------
# We are using the previously fitted mixed-effects model:
# model_mixed <- lmer(yield.kgha ~ treatment + (1 | siteyr) + (1 | siteyr:block), data = df)

# 1. Load the emmeans package for estimated marginal means (EMMs)
# If not installed, first run: install.packages("emmeans")
library(emmeans)

# 2. Compute estimated marginal means for treatments
emms <- emmeans(model_mixed, specs = "treatment")
# emms contains the predicted means of yield for each treatment,
# adjusted for the random effects of site-year and block

# 3. Perform pairwise comparisons between treatments
# Using Tukey method to adjust for multiple comparisons
pairwise_results <- contrast(emms, method = "pairwise", adjust = "tukey")
# contrast() calculates the differences between all treatment pairs

# 4. View the summary of pairwise comparisons
summary(pairwise_results)
# This will show:
# - The estimated difference between each pair of treatments
# - Standard errors
# - Confidence intervals
# - Adjusted p-values

# 5. Convert results to a data frame for saving/exporting
pairwise_df <- as.data.frame(summary(pairwise_results))
head(pairwise_df)  # preview first few rows

# 6. Save pairwise results to CSV
# write.csv(pairwise_df, "pairwise_comparisons.csv", row.names = FALSE)

