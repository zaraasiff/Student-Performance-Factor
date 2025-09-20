📘 Student Score Prediction
📌 Objective

The objective of this project is to predict student exam scores based on study-related factors (such as hours studied).
This project demonstrates the use of Linear Regression and Polynomial Regression to model and analyze real-world data.

📊 Dataset

Dataset used: Student Performance Factors (Kaggle)

It contains details such as:

Hours Studied

Sleep Hours

Attendance

Participation

Exam Score (Target Variable)

For simplicity in this project, we primarily focused on:

Feature: Hours_Studied

Target: Exam_Score

🛠️ Steps Performed
1. Data Preprocessing

Imported dataset (StudentPerformanceFactors.csv)

Checked missing values and data types

Selected Hours_Studied as input (X) and Exam_Score as target (Y)

2. Train-Test Split

Data divided into 80% Training and 20% Testing using train_test_split

3. Linear Regression

Trained a Linear Regression model

Evaluated with MSE, RMSE, and R² Score

Visualized Actual vs Predicted values

4. Polynomial Regression (Bonus)

Applied PolynomialFeatures (degree=2)

Trained and compared performance with Linear Regression

Visualized predictions as a curve

📈 Results
🔹 Linear Regression

Shows a straight line fit to the data

Performance measured with R², RMSE

🔹 Polynomial Regression

Captures non-linear patterns in data

Provided better fit (higher R² and lower error compared to Linear Regression)

🖼️ Visualizations

Scatter plots of Actual vs Predicted

Comparison plot between Linear Regression line and Polynomial Regression curve

⚙️ Libraries Used

pandas → Data handling

numpy → Mathematical operations

matplotlib → Visualization

scikit-learn → Model building and evaluation
