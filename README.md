# Simple Linear Regression

## Use of the Jupyter Notebook

You can use [Google Colab](https://colab.research.google.com/) or any other notebook environment for a better experience.

This project demonstrates a simple linear regression model using Python and scikit-learn to predict a continuous target variable (e.g., salary or CTC) based on a single feature (e.g., years of experience).

## 📁 Files Included

- `Simple Linear Regression Template.py`: The Python script with all the code for data preprocessing, model training, prediction, and visualization.
- `README.md`: This file.

## 📌 Steps Performed

1. **Importing Libraries**  
   - NumPy, Pandas, Matplotlib
   - `LinearRegression` from `sklearn.linear_model`

2. **Loading the Dataset**  
   - Using `pandas.read_csv()` to load a CSV file.

3. **Splitting the Dataset**  
   - Train-test split using `train_test_split` from `sklearn.model_selection`

4. **Training the Model**  
   - Creating a `LinearRegression()` object and fitting it to training data.

5. **Predicting Results**  
   - Using `regressor.predict()` to make predictions on test data.

6. **Visualizing the Results**  
   - Scatter plots and regression lines for both training and test sets using Matplotlib.

## 🛠 Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

Install the requirements using:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## 📊 Output

The output includes:
- A regression line showing the predicted relationship.
- Red dots representing the actual data points.
- The model learns and visualizes the relationship between experience and salary (or any other dependent variable).

## 📬 Author

- You can customize this section with your name and contact details if needed.

---

*This project is a beginner-friendly introduction to machine learning using simple linear regression.*
