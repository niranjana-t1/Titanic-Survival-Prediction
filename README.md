# Titanic-Survival-Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster using passenger information such as age, gender, passenger class, fare, and family details.

## Project Overview

This project applies machine learning techniques to analyze historical passenger data from the Titanic dataset and predict survival outcomes.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training and evaluation
* Prediction interface using Streamlit

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

## Dataset Features

The dataset contains the following important features:

* Passenger Class (Pclass)
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked

Target Variable:

* Survived (0 = Not Survived, 1 = Survived)

## Data Preprocessing

The following preprocessing steps were performed:

* Removed Cabin column due to excessive missing values
* Filled missing Age values using mean
* Filled missing Embarked values using mode
* Converted categorical variables into numerical format

## Exploratory Data Analysis

Some important observations from the dataset:

* Female passengers had higher survival rates
* First-class passengers survived more frequently
* Moderate family sizes showed better survival probability
* Passengers embarking from Southampton had higher survival counts

## Machine Learning Models Used

### Logistic Regression

Used as the primary model because the problem is a binary classification task.

### Random Forest

Used for comparison and to capture more complex patterns in the data.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

## Prediction System

A simple web application was built using Streamlit where users can enter passenger details and receive survival predictions.

## How to Run the Project

1. Clone the repository
2. Install required libraries
3. Run the Streamlit application:

```bash
streamlit run streamlit.py
```

## Project Type
Tutorial-guided beginner machine learning project for learning purposes.

## Conclusion
This project demonstrates how machine learning can be used to analyze historical data and generate meaningful predictions. It also highlights the practical implementation of machine learning using a deployed prediction interface.





