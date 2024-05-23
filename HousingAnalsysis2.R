
#------------------- DATA IMPORTATION AND CLEANING-----------------------------------
# Read the data from CSV file
housing_data_raw = read.csv("Housing.csv", header = TRUE)

#Load necessary library
library(ggplot2)
library(dplyr)
library(scales)
library(leaps)
library(ISLR)
library(randomForest)
library(ggplot2)
library(tree)
library(AUC)
library(e1071)
library(corrplot)
library(lmtest)
library(glmnet)

# Convert 'yes'/'no' to 1/0 for specified columns
housing_data_raw <- housing_data_raw %>%
  mutate(mainroad = ifelse(mainroad == "yes", 1, 0),
         guestroom = ifelse(guestroom == "yes", 1, 0),
         basement = ifelse(basement == "yes", 1, 0),
         hotwaterheating = ifelse(hotwaterheating == "yes", 1, 0),
         airconditioning = ifelse(airconditioning == "yes", 1, 0),
         prefarea = ifelse(prefarea == "yes", 1,0))

# Transform 'furnishingstatus' from categorical to numerical
housing_data_raw$furnishingstatus_num <- as.integer(factor(housing_data_raw$furnishingstatus, levels = c("unfurnished", "semi-furnished", "furnished"), labels = c(1, 2, 3)))

# Print the updated dataframe to verify changes
head(housing_data_raw)

#Remove 'furnishingstatus' from the dataframe
housing_data <- housing_data_raw[,-13]

# Print the updated data frame to verify changes
print(head(housing_data))

# Split the data into training and testing sets
set.seed(123) # for reproducibility
index <- sample(1:nrow(housing_data), size = 0.8 * nrow(housing_data)) # 80% for training
train_data <- housing_data[index, ]
test_data <- housing_data[-index, ]


#--------------------------- Best subset -------------------------------

###Firtst trial with the Exhaustive model

HousingSubset_Exhaustive <- regsubsets(price~.,train_data)
HousingSubset_Exhaustive_summary <- summary(HousingSubset_Exhaustive)
#plotting RSS, adjusted R2, C_p, BIC to check which model is best
par(mfrow=c(2,2))
par(mar = c(4, 4, 1, 1))
plot(HousingSubset_Exhaustive_summary$rss, type='l',xlab='Number of variables',ylab='RSS')
plot(HousingSubset_Exhaustive_summary$adjr2, type='l',xlab='Number of variables',ylab='Adjusted R2')
plot(HousingSubset_Exhaustive_summary$cp, type='l',xlab='Number of variables',ylab='Cp')
plot(HousingSubset_Exhaustive_summary$bic, type='l',xlab='Number of variables',ylab='BIC')

## Interpretation 
min_BIC_index_ex<- (which.min(HousingSubset_Exhaustive_summary$bic)) # minimum value for BIC
(coef(HousingSubset_Exhaustive,min_BIC_index_ex)) #model with 1 predictors
max_ADJR2_index_ex <- (which.max(HousingSubset_Exhaustive_summary$adjr2)) # maximum value for adjr2
(coef(HousingSubset_Exhaustive,max_ADJR2_index_ex)) #model with 4 predictors
#All charts are indicating that 8 variables should be included for best performing model
# bedrooms, mainroad, guestroom, and furnishing status have been suggested as variables to be omitted

###Second trial with the Forward Selection model

HousingSubset_Forward <- regsubsets(price~.,train_data,method ="forward")
HousingSubset_Forward_summary <- summary(HousingSubset_Forward)
#plotting RSS, adjusted R2, C_p, BIC to check which model is best
par(mfrow=c(2,2))
par(mar = c(4, 4, 1, 1))
plot(HousingSubset_Forward_summary$rss, type='l',xlab='Number of variables',ylab='RSS')
plot(HousingSubset_Forward_summary$adjr2, type='l',xlab='Number of variables',ylab='Adjusted R2')
plot(HousingSubset_Forward_summary$cp, type='l',xlab='Number of variables',ylab='Cp')
plot(HousingSubset_Forward_summary$bic, type='l',xlab='Number of variables',ylab='BIC')

## Interpretation 
min_BIC_index_fwd<- (which.min(HousingSubset_Forward_summary$bic)) # minimum value for BIC
(coef(HousingSubset_Forward,min_BIC_index_ex)) #model with 1 predictors
max_ADJR2_index_fwd <- (which.max(HousingSubset_Forward_summary$adjr2)) # maximum value for adjr2
(coef(HousingSubset_Forward,max_ADJR2_index_ex)) #model with 4 predictors

#Similar interpretation to Exhaustive model

###Third trial with the Backward Selection model

HousingSubset_Backward <- regsubsets(price~.,train_data,method ="backward")
HousingSubset_Backward_summary <- summary(HousingSubset_Backward)
#plotting RSS, adjusted R2, C_p, BIC to check which model is best
par(mfrow=c(2,2))
par(mar = c(4, 4, 1, 1))
plot(HousingSubset_Backward_summary$rss, type='l',xlab='Number of variables',ylab='RSS')
plot(HousingSubset_Backward_summary$adjr2, type='l',xlab='Number of variables',ylab='Adjusted R2')
plot(HousingSubset_Backward_summary$cp, type='l',xlab='Number of variables',ylab='Cp')
plot(HousingSubset_Backward_summary$bic, type='l',xlab='Number of variables',ylab='BIC')

## Interpretation 
min_BIC_index_fwd<- (which.min(HousingSubset_Forward_summary$bic)) # minimum value for BIC
(coef(HousingSubset_Forward,min_BIC_index_ex)) #model with 1 predictors
max_ADJR2_index_fwd <- (which.max(HousingSubset_Forward_summary$adjr2)) # maximum value for adjr2
(coef(HousingSubset_Forward,max_ADJR2_index_ex)) #model with 4 predictors

##Similar interpretation to Exhaustive model and forward selection model


###------> Multilinear regression model using all variables 

## Creating the multilinear regression model using all variables 
#Multiple Linear Regression -
Multi_Reg_all=lm(price~.,data=train_data)
print(summary(Multi_Reg_all))

# Predictions with the test data
yhat_reg_all <- predict(Multi_Reg_all, test_data)

# Plot the predictions against the actual prices
par(mfrow=c(1,1))
plot(yhat_reg_all, test_data$price, xlab = "Predicted Price", ylab = "Actual Price", col = "blue", main = "Prediction accuracy for the multilinear regression with all variables")
abline(0, 1)

#Calculate the Mean Squared Error (MSE)
mse_reg_all <- mean((test_data$price-yhat_reg_all)**2)

# Calculate the Mean Absolute Error (MAE)
mae_reg_all <- mean(abs(test_data$price - yhat_reg_all))

# Calculate the Root Mean Squared Error (RMSE)
rmse_reg_all <- sqrt(mse_reg_all)

# Calculate the Mean Error (ME)
me_reg_all <- mean(test_data$price - yhat_reg_all)

# Printing the RMSE and ME
print(paste("MSE:", mse_reg_all))
print(paste("MAE:", mae_reg_all))
print(paste("RMSE:", rmse_reg_all))
print(paste("ME:", me_reg_all))


###------> Multilinear regression model using suggested variables 

## Creating the multilinear regression model using the suggested variables  
#Multiple Linear Regression -
Multi_Reg_subset=lm(price~area+bathrooms+stories+basement+hotwaterheating+airconditioning+parking+prefarea,data=train_data)
print(summary(Multi_Reg_subset))

# Now make predictions with the test data
yhat_reg_subset <- predict(Multi_Reg_subset, test_data)

# Plot the predictions against the actual prices
plot(yhat_reg_subset, test_data$price, xlab = "Predicted Price", ylab = "Actual Price", col = "blue", main = "Prediction accuracy for the multilinear regression with all variables")
abline(0, 1)

#Calculate the Mean Squared Error (MSE)
mse_reg_subset <- mean((test_data$price-yhat_reg_subset)**2)

# Calculate the Mean Absolute Error (MAE)
mae_reg_subset <- mean(abs(test_data$price - yhat_reg_subset))

# Calculate the Root Mean Squared Error (RMSE)
rmse_reg_subset <- sqrt(mse_reg_subset)

# Calculate the Mean Error (ME)
me_reg_subset <- mean(test_data$price - yhat_reg_subset)

# Printing the RMSE and ME
print(paste("MSE:", mse_reg_subset))
print(paste("MAE:", mae_reg_subset))
print(paste("RMSE:", rmse_reg_subset))
print(paste("ME:", me_reg_subset))


## the analysis shows that although using the suggested variables using regsubset
# the remaining variables have a high statitistical significance lower than 1%
# the model with all the variables provides a lower MSE error in generalization 


### ------> Multilinear regression model using all variables except bathroom 
#Multiple Linear Regression without bedroom only
Multi_Reg_subset2 =lm(price~area+bathrooms+stories+mainroad+guestroom+basement+hotwaterheating+airconditioning+parking+prefarea+furnishingstatus_num,data=train_data)
print(summary(Multi_Reg_subset2))

# Predictions with the test data
yhat_reg_subset2 <- predict(Multi_Reg_subset2, test_data)

#Calculate the Mean Squared Error (MSE)
mse_reg_subset2 <- mean((test_data$price-yhat_reg_subset2)**2)

# Calculate the Mean Absolute Error (MAE)
mae_reg_subset2 <- mean(abs(test_data$price - yhat_reg_subset2))

# Calculate the Root Mean Squared Error (RMSE)
rmse_reg_subset2 <- sqrt(mse_reg_subset2)

# Calculate the Mean Error (ME)
me_reg_subset2 <- mean(test_data$price - yhat_reg_subset2)

# Printing the RMSE and ME
print(paste("MSE:", mse_reg_subset2))
print(paste("MAE:", mae_reg_subset2))
print(paste("RMSE:", rmse_reg_subset2))
print(paste("ME:", me_reg_subset2))

###### --------- Evaluation of best multilinear regression model -------------------------------------------########

# Create vector of mse, mae, rmse, and me values
mse_values <- c(mse_reg_subset, mse_ridge_reg_new, mse_reg_all)
mae_values <- c(mae_reg_subset, mae_ridge_reg_new, mae_reg_all)
rmse_values <- c(rmse_reg_subset, rmse_ridge_reg_new, rmse_reg_all)
me_values <- c(me_reg_subset, me_ridge_reg_new, me_reg_all)
adjustedR_values<- c(0.691,0.7083,0.7082)

# Combine the metrics into a data frame for easier plotting
metrics_df <- data.frame(
  Model = rep(c("subset", "Ridge reg", "all-variable"), times=4),
  Value = c( mse_values, mae_values, rmse_values, me_values),
  Metric = rep(c("MSE", "MAE", "RMSE", "ME" ), each=3)
)

ggplot(metrics_df, aes(x=Model, y=Value, fill=Metric)) + 
  geom_bar(stat="identity", position="dodge") +
  geom_text(aes(label=ifelse(Metric == "MSE", paste0(round(Value / 1e12, 3), " T"), round(Value, 2))), vjust=-0.3, position=position_dodge(width=0.3)) +
  facet_wrap(~Metric, scales = 'free', nrow=1) +
  theme_light() +
  labs(title="Metric Comparison Across Multivariate Linear Regression  and Ridge Regression Models", x="Model", y="Value") +
  scale_fill_brewer(palette="Pastel1")

#######--------------- ANALYSIS OF LINEAR REGRESSION ASSUMPTIONS-----------------------
# 1. Multicollinearity 
# Compute the correlation matrix
cor_matrix <- cor(housing_data[-1])

# Visualize the correlation matrix using a heatmap
corrplot(cor_matrix, method = "number")


# 2. Linearity 
# Perform the Harvey Collier Test 
hc_test <- harvtest(Multi_Reg_all)

# Print the results
print(hc_test)

# 3. Heteroscedasticity and Homoscedasticity
par(mfrow=c(3,1))
# Calculate fitted values for the training data
fitted_values <- fitted(Multi_Reg_all)

# Calculate residuals for the training data
residuals <- residuals(Multi_Reg_all)

# Plot the fitted values against the actual prices
plot(train_data$price, fitted_values, xlab = "Actual Price", ylab = "Fitted Price", col = "red", main = "Fitted vs Actual Prices")
abline(0, 1, col = "green") # Add a 45-degree line

# Plot the residuals
plot(residuals, xlab = "Observation", ylab = "Residuals", col = "red", main = "Residuals of the Model")
abline(h = 0, col = "green") # Add a horizontal line at 0

# Create a residuals vs. fitted values plot
plot(fitted_values, residuals, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted Values")
abline(h = 0, col = "red", lwd = 2) # Add a horizontal line at 0

# Adding a lowess line to check for patterns
lines(lowess(fitted_values, residuals), col = "blue", lwd = 2)

# Perform the Breusch-Pagan test
bptest_result <- bptest(Multi_Reg_all)

# Output the result
print(bptest_result)


########-----------------------------Ridge regression-------###### 

#Ridge regression without transformation of price 
x_train = model.matrix(price~.,train_data)[, -1] # Prepares predictor matrix excluding intercept
y_train = train_data$price # Response variable vector

x_test = model.matrix(price~.,test_data)[, -1] # Prepares predictor matrix excluding intercept
y_test = test_data$price # Response variable vector

# Here, we define a lambda sequence from 10^8 to 10^-4.
grid = 10^seq(8, -4, length = 100)

#Now let's fit ridge regression again with lambda=100
#use predict function to make predictions based on input x from the test set
ridgeModelCV=glmnet(x_train,y_train,alpha=0,lambda=grid)

set.seed(123)
par(mfrow=c(1,1))
cv.out=cv.glmnet(x_train,y_train,alpha=0)
plot(cv.out, main = "Cross validation on ridge regression without transformation")
(best_lambda=cv.out$lambda.min)

yhat_ridge_reg=predict(ridgeModelCV,s=best_lambda,newx = x_test)

#Calculate the Mean Squared Error (MSE)
mse_ridge_reg <- mean((test_data$price-yhat_ridge_reg)**2)

# Calculate the Mean Absolute Error (MAE)
mae_ridge_reg <- mean(abs(test_data$price - yhat_ridge_reg))

# Calculate the Root Mean Squared Error (RMSE)
rmse_ridge_reg<- sqrt(mse_ridge_reg)

# Calculate the Mean Error (ME)
me_ridge_reg <- mean(test_data$price - yhat_ridge_reg)


#Transformation and ridge regression 

#RIdge regression with dependent variable transformation

x_train_new = model.matrix(price~.,train_data)[, -1] # Prepares predictor matrix excluding intercept
y_train_new = log (train_data$price) # Response variable vector

x_test_new = model.matrix(price~.,test_data)[, -1] # Prepares predictor matrix excluding intercept
y_test_new = log(test_data$price) # Response variable vector

# Here, we define a lambda sequence from 10^8 to 10^-4.
grid = 10^seq(8, -4, length = 100)

#Now let's fit ridge regression again with lambda=100
#use predict function to make predictions based on input x from the test set
ridgeModelCV=glmnet(x_train_new,y_train_new,alpha=0,lambda=grid)

set.seed(123)
cv.out=cv.glmnet(x_train_new,y_train_new,alpha=0)
plot(cv.out, main = "Cross validation on ridge regression with log transformation")
(best_lambda=cv.out$lambda.min)

yhat_ridge_reg_new=exp(predict(ridgeModelCV,s=best_lambda,newx = x_test_new))

#Calculate the Mean Squared Error (MSE)
mse_ridge_reg_new <- mean((test_data$price-yhat_ridge_reg_new)**2)

# Calculate the Mean Absolute Error (MAE)
mae_ridge_reg_new <- mean(abs(test_data$price - yhat_ridge_reg_new))

# Calculate the Root Mean Squared Error (RMSE)
rmse_ridge_reg_new<- sqrt(mse_ridge_reg_new)

# Calculate the Mean Error (ME)
me_ridge_reg_new <- mean(test_data$price - yhat_ridge_reg_new)

###### --------- RANDOM FOREST MODEL -------------------------------------------########
## We selected the random forest model as it has higher capability to operate with non-linear data
# and is an ensemble model that is less prone to overfitting 
# We also optimize for the hyper-parameter "mtry" indicating the best through 
# training 12 RF models at different values of "mtry" and using the MSE to get the best model


# Initialize a vector to store the MSE values for each mtry
mse_values_RF <- numeric(11)  # testing 12 different values (from 2 to 13)
mae_values_RF <- numeric(11)
me_values_RF <- numeric(11)
rmse_values_RF <- numeric(11)

# Loop over the mtry values from 2 to 12
set.seed(123)
for (mtry_val in 2:12) {
  # Train the Random Forest model with the current mtry value
  RF <- randomForest(price~., train_data, mtry = mtry_val, importance = TRUE)
  
  # Make predictions on the test set
  y_hat_RF <- predict(RF, test_data)
  
  # Calculate and print the MSE,MAE,RMSE, and ME for the current mtry value
  mse <- mean((y_hat_RF - test_data$price)^2)
  mae <- mean(abs(test_data$price - y_hat_RF))
  rmse <- sqrt(mse)
  me<-mean(y_hat_RF - test_data$price)
  
  mse_values_RF[mtry_val - 1] <- mse # Store the MSE in the vector
  mae_values_RF[mtry_val - 1] <- mae # Store the MAE in the vector
  rmse_values_RF[mtry_val - 1]<- rmse# Store the RMSE in the vector
  me_values_RF[mtry_val - 1]<- me# Store the ME in the vector
  
  cat("MSE for Random Forest with mtry =", mtry_val, ":", mse, "\n")
  cat("MAE for Random Forest with mtry =", mtry_val, ":", mae, "\n")
  cat("RMSE for Random Forest with mtry =", mtry_val, ":", rmse, "\n")
  cat("ME for Random Forest with mtry =", mtry_val, ":", me, "\n")
  
  if (mtry_val == 9) {
    # Extract and print variable importance scores
    importance_values <- importance(RF)  # Extract importance values
    print(importance_values)  # Print the raw importance values
    
    # Storing the yhat, MSE, RMSE, ME, and MAE values 
    yhat_RF <- y_hat_RF
    mse_RF <- mse
    rmse_RF <- rmse 
    me_RF <- me
    mae_RF <- mae 
    
    # Creating a data frame for easier plotting
    importance_df <- data.frame(
      Variable = rownames(importance_values), 
      Importance = importance_values[, "%IncMSE"] 
    )
    
    # Ordering the data for better plotting
    importance_df <- importance_df[order(-importance_df$Importance), ]
    
    # Plotting variable importance using ggplot2
    ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_bar(stat = "identity") +
      coord_flip() +  # Flips the axes for easier reading of variable names
      xlab("Variable") +
      ylab("Importance") +
      ggtitle("Variable Importance in Random Forest Model")
    
    
  }
}

# Plot the MSE values against the mtry values to visualize the results
plot(2:12, mse_values_RF, type = "b", col = "blue", pch = 19, xlab = "mtry value", ylab = "MSE",
     main = "MSE and MAE vs. mtry for Random Forest", ylim = range(mse_values_RF))

# Add a legend for MSE
legend("topright", inset=.05, legend = "MSE", col = "blue", pch = 19, bty = "n")

# Superimpose the new plot for MAE values on a different scale
par(new = TRUE)  # Allows adding a new plot on top of the existing one

# Plot MAE values with a secondary axis; no need for x-axis labels, ticks, or line since they overlay the existing plot
plot(2:12, mae_values_RF, type = "b", col = "red", pch = 17, xaxt = "n", yaxt = "n", xlab = "", ylab = "", ylim = range(mae_values_RF), xlim = range(2:12))

# Add secondary y-axis for MAE values
axis(side = 4, at = pretty(range(mae_values)), las = 1)  # Ensure that 'at' uses the range of mae_values. 'las = 1' makes labels horizontal.
mtext("MAE", side = 4, line = 3)  # Add label for the secondary y-axis

# Add a legend for MAE, avoiding overlap with MSE legend
legend("topright", inset=.1, legend = "MAE", col = "red", pch = 17, bty = "n")

# -------------------------------------

# Plot the RMSE values against the mtry values to visualize the results
plot(2:12, rmse_values_RF, type = "b", col = "purple", pch = 19, xlab = "mtry value", ylab = "RMSE",
     main = "RMSE and ME vs. mtry for Random Forest", ylim = range(rmse_values_RF))

# Add a legend for RMSE
legend("topright", inset=.05, legend = "RMSE", col = "purple", pch = 19, bty = "n")

# Superimpose the new plot for ME values on a different scale
par(new = TRUE)  # Allows adding a new plot on top of the existing one

# Plot ME values with a secondary axis; no need for x-axis labels, ticks, or line since they overlay the existing plot
plot(2:12, me_values_RF, type = "b", col = "green", pch = 17, xaxt = "n", yaxt = "n", xlab = "", ylab = "", ylim = range(me_values_RF), xlim = range(2:12))

# Add secondary y-axis for ME values
axis(side = 4, at = pretty(range(me_values_RF)), las = 1)  # Ensure that 'at' uses the range of mae_values. 'las = 1' makes labels horizontal.
mtext("ME", side = 4, line = 3)  # Add label for the secondary y-axis

# Add a legend for ME, avoiding overlap with MSE legend
legend("topright", inset=.0, legend = "ME", col = "green", pch = 17, bty = "n")

#------------------



## the best Random forest model has outperformed the multilinear regression model 
# Generally, all MSE errors from the random forest were better compared to the multilinear regression model 


###### --------- SVM  -------------------------------------------########

########## Radial SVM
# Build a SVM with radial base kernel
# Optimize the cost,gamma and  degree parameter
SV.cost <- 2^(2:16) #lower values would give in combination with gamma unfeasible solutions
SV.gamma <-  2^(-8:8)

#Add the cost, gamma and degree to the ranges list
(tuninglist <- tune(svm,price ~ ., 
                    data = train_data, kernel = 'radial',
                    ranges=list(cost = SV.cost,
                                gamma = SV.gamma)))

#Build the radial SVM
SV_radial <- svm(price ~ ., 
                 data = train_data,
                 type = "eps-regression", 
                 kernel = 'radial', 
                 cost = tuninglist$best.parameters$cost,
                 gamma = tuninglist$best.parameters$gamma,
                 probability=TRUE)


#Make predictions with classes
yhat_svm <- predict(SV_radial,test_data)

# Calculate  MSE 
mse_svm <- mean((yhat_svm - test_data$price)**2)

# Calculate the Mean Absolute Error (MAE)
mae_svm <- mean(abs(test_data$price - yhat_svm))

# Calculate the Root Mean Squared Error (RMSE)
rmse_svm <- sqrt(mse_svm)

# Calculate the Mean Error (ME)
me_svm <- mean(test_data$price - yhat_svm)

# Plot the predictions against the actual prices for radial svm
plot(yhat_svm, test_data$price, xlab = "Predicted Price", ylab = "Actual Price", col = "blue", main = "Prediction accuracy for Radial SVM")
abline(0, 1)

###---------------------------------- Evaluation charts for the report 


par(mfrow=c(3,1))

# Plot the predictions against the actual prices for the multilinear regression 
plot(yhat_reg_subset2, test_data$price, xlab = "Predicted Price", ylab = "Actual Price", col = "blue", main = "Prediction accuracy for MLR model (all-variables)")
abline(0, 1)

# Plot the predictions against the actual prices for the random forest
plot(yhat_RF, test_data$price, xlab = "Predicted Price", ylab = "Actual Price", col = "blue", main = "Prediction accuracy for RF model ('mtry' = 9)")
abline(0, 1)

# Plot the predictions against the actual prices for radial svm
plot(yhat_svm, test_data$price, xlab = "Predicted Price", ylab = "Actual Price", col = "blue", main = "Prediction accuracy for  Radial SVM (cost = 16, gamma =0.00390625 )")
abline(0, 1)


## plotting the prediction vs actual price shows that price is actually closer to the line for RF than for both 
# RF is more versatile, with more prediction space 
# Best models are RF and SVM 

# Create vector of mse, mae, rmse, and me values
mse_values_models <- c(mse_reg_all,mse_ridge_reg_new, mse_RF,mse_svm)
mae_values_models <- c(mae_reg_all,mae_ridge_reg_new,  mae_RF,mae_svm )
rmse_values_models <- c(rmse_reg_all, rmse_ridge_reg_new, rmse_RF,rmse_svm )
me_values_models <- c(me_reg_all, me_ridge_reg_new, me_RF, me_svm )

# Combine the metrics into a data frame for easier plotting
metrics_df <- data.frame(
  Model = rep(c("MLR all-variables","Ridge Reg","RF", "Radial SVM"), times=4),
  Value = c(mse_values_models, mae_values_models, rmse_values_models, me_values_models),
  Metric = rep(c("MSE", "MAE", "RMSE", "ME"), each=4)
)


ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = ifelse(Metric == "MSE", paste0(round(Value / 1e9, 2), " B"), round(Value, 2))),
            vjust = -0.3, position = position_dodge(width = 0.9), size = 3) +
  facet_wrap(~Metric, scales = 'free', nrow = 1) +
  theme_light() +
  labs(title = "Metric Comparison Across MLR model (all-variables), Ridge Regression, RF model, and Radial SVM",
       x = "Model", y = "Value") +
  scale_fill_brewer(palette = "Pastel1") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))















