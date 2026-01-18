# Rossmann Store Sales Prediction: A Deep Learning Approach

This project implements a Deep Learning model to forecast daily sales for 1,115 Rossmann stores across Germany. The primary objective is to predict 6 weeks of sales for these stores while accounting for seasonal factors, promotions, and store-specific features.

## Key Technique: Entity Embeddings
The standout feature of this implementation is the use of **Entity Embeddings** for categorical data. Unlike standard One-Hot Encoding, Entity Embeddings:
* Map categorical variables (like `Store ID`, `DayOfWeek`, and `Month`) into a continuous, low-dimensional vector space.
* Allow the model to learn complex relationships between entities (e.g., similar sales patterns between specific stores or holidays).
* Significantly reduce the dimensionality of the input space while capturing more meaningful information.

[Image of neural network entity embedding layer architecture]

## Data Engineering & Preprocessing
* **Target Transformation:** Applied log transformation $y' = \ln(1 + \text{Sales})$ to normalize the target distribution and stabilize the training process.
* **Missing Value Imputation:** Handled missing values in `CompetitionDistance` using the median and filled other competitive features with zeros to signify no competition.
* **Feature Extraction:** Engineered temporal features including `Year`, `Month`, `Day`, `DayOfWeek`, and `WeekOfYear`.
* **Competition Analysis:** Calculated the duration of competition in months to assess market impact over time.
* **Scaling:** Normalized continuous features using `StandardScaler` to ensure all inputs contribute equally to the gradient descent.

## Model Architecture
The model is a hybrid Multi-Input Artificial Neural Network built with **TensorFlow/Keras**:
