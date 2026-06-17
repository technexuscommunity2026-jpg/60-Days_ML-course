# Regression Model from Scratch – Easy Explanation

## Introduction

In this notebook, we build a **Linear Regression model from scratch** without using any machine learning library such as **Scikit-Learn**. The purpose is to understand how Linear Regression works internally instead of simply using built-in functions.

---

# Importing the Kaggle Dataset

```python
!pip install kagglehub
```

### Explanation

* Installs the **kagglehub** package.
* This package allows us to download datasets directly from Kaggle.

Think of it as installing an application before using it.

---

```python
import kagglehub

path = kagglehub.dataset_download("andonians/random-linear-regression")

print(path)
```

### Explanation

* Imports the KaggleHub library.
* Downloads the Linear Regression dataset.
* Stores the dataset on your computer.
* Prints the location where the dataset is saved.

---

# Importing Important Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Explanation

### Pandas (`pandas`)

* Reads CSV files.
* Stores data in tables (DataFrames).
* Helps clean and analyze data.

### NumPy (`numpy`)

* Performs mathematical calculations.
* Works efficiently with arrays and numerical values.

### Matplotlib (`matplotlib`)

* Draws graphs and charts.
* Helps visualize the dataset.

---

# Loading the Dataset

```python
train_data = pd.read_csv(...)
test_data = pd.read_csv(...)
```

## Explanation

This code loads the dataset into Python.

There are two datasets:

* **Training Data** – Used to teach the model.
* **Testing Data** – Used to evaluate the model after training.

---

# Displaying the First Five Rows

```python
train_data.head()
```

## Explanation

Displays the first five rows of the dataset.

This helps us understand:

* Column names
* Sample values
* Structure of the dataset

---

# Statistical Summary

```python
train_data.describe()
```

## Explanation

Shows statistical information such as:

* Count
* Mean
* Standard Deviation
* Minimum value
* Maximum value

This provides a quick overview of the dataset.

---

# Dataset Information

```python
train_data.info()
```

## Explanation

Displays:

* Number of rows
* Number of columns
* Data types
* Missing values
* Memory usage

This helps us understand the structure of the dataset before training.

---

# Checking Missing Values

```python
train_data.isnull().sum()
```

## Explanation

Checks whether any values are missing in each column.

Missing values can reduce the performance of a machine learning model, so they should be handled before training.

---

# Visualizing the Dataset

```python
plt.scatter(train_data['x'], train_data['y'])
```

## Explanation

Creates a scatter plot.

* Every dot represents one data point.
* It helps us observe the relationship between the input (`x`) and the output (`y`).

If the points roughly form a straight line, Linear Regression is an appropriate algorithm.

---

# Removing Missing Values

```python
train_data.dropna(inplace=True)
```

## Explanation

Removes all rows containing missing values.

This ensures the model is trained using clean and complete data.

---

# Creating the Loss Function

```python
def Loss_Function(m, c, x, y):
```

## Explanation

The **Loss Function** measures how wrong the model's predictions are.

* Small loss → Better prediction
* Large loss → Poor prediction

The goal of training is to reduce the loss as much as possible.

---

## Predicting Values

```python
y_pred = m * x + c
```

### Explanation

This is the equation of a straight line.

Where:

* **m** = Slope of the line
* **c** = Intercept
* **x** = Input value
* **y_pred** = Predicted output

The model uses this equation to predict output values.

---

## Calculating Error

```python
error = y - y_pred
```

### Explanation

Calculates the difference between:

* Actual value (`y`)
* Predicted value (`y_pred`)

Smaller error means better predictions.

---

## Mean Squared Error (MSE)

```python
np.mean(error**2)
```

### Explanation

* Squares every error so negative values become positive.
* Finds the average of all squared errors.

This value is called **Mean Squared Error (MSE)**.

A lower MSE means a better-performing model.

---

# Gradient Descent

```python
Gradient_Descent(...)
```

## Explanation

Gradient Descent is the learning algorithm.

Its job is to improve the regression line by updating:

* Slope (`m`)
* Intercept (`c`)

It keeps updating these values until the prediction error becomes very small.

---

## Predict Output

```python
y_pred = m * x + c
```

### Explanation

Uses the current values of the slope and intercept to predict output values.

---

## Calculate Error

```python
error = y - y_pred
```

### Explanation

Measures the difference between predicted values and actual values.

---

## Calculate Gradients

```python
m_grad
c_grad
```

### Explanation

Gradients tell us:

* Which direction to move.
* How much to update the slope and intercept.

Think of standing on a hill.

The gradient tells you the fastest way to reach the bottom of the hill, where the loss is minimum.

---

## Update Parameters

```python
m = m - learning_rate * m_grad
c = c - learning_rate * c_grad
```

### Explanation

Updates the slope and intercept.

Each update improves the regression line and reduces the prediction error.

---

# Display Final Parameters

```python
print("Slope:", m)
print("Intercept:", c)
```

## Explanation

Displays the final values of:

* Slope (`m`)
* Intercept (`c`)

These values define the best-fit regression line.

---

# Plotting the Loss Curve

```python
plot_loss_curve()
```

## Explanation

Draws a graph between:

* Slope (`m`)
* Loss

This graph helps us understand how the loss changes for different slope values.

---

## Generate Different Slopes

```python
m_values = np.linspace(-1000, 1000, 200)
```

### Explanation

Creates many different slope values.

Each slope is tested to determine the amount of prediction error.

---

## Calculate Loss

```python
loss = np.mean((y - y_pred) ** 2)
```

### Explanation

Calculates the Mean Squared Error for each slope value.

---

## Draw the Loss Graph

```python
plt.plot(...)
```

### Explanation

Draws the Loss Curve.

The graph usually looks like a **U-shaped bowl**.

The lowest point represents the minimum loss and therefore the best slope.

---

# Plotting the Regression Line

```python
plt.scatter(...)
plt.plot(...)
```

## Explanation

This graph contains:

* **Blue dots** → Original data points
* **Red line** → Regression line

If the red line passes close to most of the blue dots, the model has learned the relationship correctly.

---

# Complete Workflow

```text
Load Dataset
      ↓
Import Libraries
      ↓
Explore the Dataset
      ↓
Check Missing Values
      ↓
Remove Missing Values
      ↓
Visualize the Data
      ↓
Create the Loss Function
      ↓
Apply Gradient Descent
      ↓
Find the Best Slope and Intercept
      ↓
Plot the Regression Line
      ↓
Evaluate the Model
```

---

# Key Terms

| Term             | Meaning                                          |
| ---------------- | ------------------------------------------------ |
| Dataset          | Collection of data used for training and testing |
| Pandas           | Reads and manages datasets                       |
| NumPy            | Performs mathematical calculations               |
| Matplotlib       | Creates graphs and visualizations                |
| Scatter Plot     | Shows the relationship between input and output  |
| Loss Function    | Measures the prediction error                    |
| Gradient Descent | Reduces the prediction error step by step        |
| Slope (m)        | Controls the steepness of the regression line    |
| Intercept (c)    | Determines where the line crosses the y-axis     |
| Regression Line  | The best-fit line used for prediction            |

---

# Conclusion

This notebook demonstrates how to build a **Linear Regression model from scratch**. Instead of relying on Scikit-Learn, it manually implements every important step, including prediction, error calculation, the loss function, gradient descent, parameter updates, and plotting the final regression line. This approach helps beginners understand the complete working of Linear Regression in a simple and practical way.
