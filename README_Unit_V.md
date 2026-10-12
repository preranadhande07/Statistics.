# Unit V: Exploratory Data Analysis for Data Science & Artificial Intelligence

> **GitHub-ready study notes | Simple explanations | Formulas | Python practicals | Case study | Viva and revision questions**

**Source:** *Exploratory Data Analysis for Data Science & Artificial Intelligence — Unit V*, Dr. S. Satyanarayana, Sanjivani University. These notes follow the uploaded Unit V material and organize its concepts into a practical study guide.

---

## Table of Contents

1. [Unit Overview](#1-unit-overview)
2. [What Is Exploratory Data Analysis (EDA)?](#2-what-is-exploratory-data-analysis-eda)
3. [The Data Science Lifecycle](#3-the-data-science-lifecycle)
4. [First Five Pandas Commands](#4-first-five-pandas-commands)
5. [Data Cleaning](#5-data-cleaning)
6. [Data Transformation and Preprocessing](#6-data-transformation-and-preprocessing)
7. [Train-Test Split and Data Leakage](#7-train-test-split-and-data-leakage)
8. [Feature Selection](#8-feature-selection)
9. [Supervised Learning](#9-supervised-learning)
10. [Model Evaluation Metrics](#10-model-evaluation-metrics)
11. [Algorithms, Overfitting, and Validation](#11-algorithms-overfitting-and-validation)
12. [Unsupervised Learning](#12-unsupervised-learning)
13. [EDA in HR, Telecom, Healthcare, and Banking](#13-eda-in-hr-telecom-healthcare-and-banking)
14. [Complete Telecom Churn Case Study](#14-complete-telecom-churn-case-study)
15. [Ten Hands-On Python Labs](#15-ten-hands-on-python-labs)
16. [Mini Projects and Capstone Ideas](#16-mini-projects-and-capstone-ideas)
17. [Formula Sheet](#17-formula-sheet)
18. [Common Mistakes](#18-common-mistakes)
19. [Viva Questions with Short Answers](#19-viva-questions-with-short-answers)
20. [Practice and Bloom’s Taxonomy Questions](#20-practice-and-blooms-taxonomy-questions)
21. [Final Revision Checklist](#21-final-revision-checklist)
22. [References](#22-references)

---

## 1. Unit Overview

Unit V covers the practical process of exploring data, preparing it for machine learning, selecting useful features, building supervised models, applying unsupervised learning, and interpreting results.

### Main topics

- Exploratory Data Analysis (EDA)
- Data cleaning and handling missing values
- Feature scaling, encoding, and engineering
- Train-test splitting
- Feature selection and dimensionality reduction
- Supervised learning: regression and classification
- Model evaluation metrics
- Common machine-learning algorithms
- Overfitting, underfitting, and cross-validation
- Unsupervised learning and clustering
- Domain-based case studies in HR, Telecom, Healthcare, and Banking

### The complete EDA-to-model pipeline

**Raw data → Explore → Clean → Transform → Select features → Train model → Evaluate → Interpret and act**

The important idea is to understand the data before selecting or training a model. A complicated algorithm cannot reliably fix incorrect, duplicated, or poorly understood data.

### Learning outcomes

After studying this unit, you should be able to:

1. Explore a dataset using summary statistics and visualizations.
2. Identify and handle missing values, duplicates, incorrect types, and outliers.
3. Transform data into a form suitable for machine learning.
4. Select useful features.
5. Train and evaluate supervised learning models.
6. Apply clustering and other unsupervised methods.
7. Explain findings using a practical domain-based case study.

---

## 2. What Is Exploratory Data Analysis (EDA)?

### Definition

**Exploratory Data Analysis (EDA)** is the process of using summary statistics and visualizations to understand a dataset’s main characteristics, including distributions, relationships, unusual observations, and data-quality problems, before building a model.

EDA is associated with statistician **John Tukey**.

### Why is EDA important?

EDA helps us:

- Understand the size and structure of a dataset.
- Identify missing values and duplicate rows.
- Find outliers and unusual values.
- Understand the distribution of each variable.
- Discover relationships between variables.
- Check whether the target classes are balanced.
- Identify possible data leakage.
- Choose suitable preprocessing methods and models.

**Example:** Before building a model to predict whether a telecom customer will leave, we should explore customer tenure, monthly charges, contract type, missing values, and the proportion of customers who leave.

### Three levels of EDA

| Level | Meaning | Questions answered | Common tools |
|---|---|---|---|
| Univariate | Study one variable | How is this variable distributed? | Mean, median, histogram, box plot |
| Bivariate | Study two variables | How are these two variables related? | Scatter plot, correlation, grouped box plot |
| Multivariate | Study several variables together | How do multiple variables interact? | Pair plot, correlation heatmap, PCA |

**Examples**

- **Univariate:** Find the average age of customers.
- **Bivariate:** Check whether monthly charges differ between customers who leave and customers who stay.
- **Multivariate:** Examine how tenure, monthly charges, contract type, and customer churn are related.

### Common EDA visualizations

| Visualization | Main use |
|---|---|
| Histogram | Understand a numerical variable’s distribution |
| Box plot | Compare distributions and flag possible outliers |
| Scatter plot | Explore the relationship between two numerical variables |
| Bar chart | Compare categories or class counts |
| Correlation heatmap | View correlations among numerical variables |
| Pair plot | Explore relationships among several variables |

### Five first-look questions

When receiving a new dataset, ask:

1. How many rows and columns are present?
2. What does each column represent, and what is its data type?
3. Are there missing or duplicated records?
4. What do the distributions and summary statistics look like?
5. Is the target variable balanced, and are there suspicious features?

---

## 3. The Data Science Lifecycle

A typical lifecycle is:

1. **Define the problem:** Decide what needs to be understood or predicted.
2. **Collect data:** Obtain data from files, databases, sensors, or other sources.
3. **Explore and process data:** Inspect, clean, visualize, and transform the data.
4. **Select features:** Keep the variables useful for the task.
5. **Build a model:** Use supervised or unsupervised learning as appropriate.
6. **Evaluate:** Measure performance on data not used to train the model.
7. **Deploy and monitor:** Use the model in practice and monitor its performance.

EDA and processing are central because errors made at this stage can affect every later stage.

**Key point:** “Garbage in, garbage out” means that poor-quality input data can lead to poor results even when a strong algorithm is used.

---

## 4. First Five Pandas Commands

The following commands provide a quick overview of a dataset.

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.shape)             # Number of rows and columns
print(df.head())            # First five rows
print(df.info())            # Column names, types, and non-null counts
print(df.describe())        # Summary statistics for numeric columns
print(df.isnull().sum())    # Missing values in each column
```

| Command | Purpose |
|---|---|
| `df.shape` | Shows `(rows, columns)` |
| `df.head()` | Displays the first five rows by default |
| `df.info()` | Shows column data types and non-null counts |
| `df.describe()` | Shows count, mean, standard deviation, minimum, quartiles, and maximum for numeric columns |
| `df.isnull().sum()` | Counts missing values in each column |

Other useful commands:

```python
print(df.columns)              # Column names
print(df.dtypes)                # Data types
print(df.duplicated().sum())    # Number of duplicate rows
print(df["Churn"].value_counts())  # Count of each target class
print(df["Churn"].value_counts(normalize=True))  # Class proportions
```

---

## 5. Data Cleaning

### Definition

**Data cleaning** is the process of detecting and correcting or handling missing values, duplicates, incorrect data types, inconsistent labels, and suspicious observations.

Cleaning makes data more reliable for analysis and modelling.

### 5.1 Missing values

A missing value is an entry that has not been recorded or is unavailable.

#### Common approaches

- **Delete rows:** Can be reasonable when only a small number of records are missing and removal is appropriate.
- **Delete columns:** Consider when a column contains too much missing information and is not essential.
- **Mean imputation:** Replace missing numerical values with the mean; often considered for roughly symmetric data.
- **Median imputation:** Replace missing numerical values with the median; useful for skewed data or when outliers are present.
- **Mode imputation:** Replace missing categorical values with the most frequent category.
- **Model-based imputation:** Estimate missing values using other variables, for example KNN or regression imputation.

Example:

```python
# Mean imputation for a roughly symmetric numeric column
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Median imputation for a skewed numeric column
df["Income"] = df["Income"].fillna(df["Income"].median())

# Mode imputation for a categorical column
df["Contract"] = df["Contract"].fillna(df["Contract"].mode()[0])
```

**Important:** The right approach depends on why values are missing, how much data is missing, the variable’s distribution, and the problem being solved.

A value such as `0` can sometimes be a code for missing data rather than a real measurement. For example, a zero blood-pressure value should be checked against the dataset’s documentation rather than automatically treated as valid.

### 5.2 Duplicates

Duplicate rows can inflate counts and cause the model to learn repeated examples.

```python
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
```

Do not automatically remove rows that merely look similar. Confirm whether they are genuinely duplicate records.

### 5.3 Incorrect data types

Examples include:

- A numeric amount stored as text, such as `"1,200"`.
- A date stored as a string.
- A numerical column containing invalid text values.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["JoinDate"] = pd.to_datetime(
    df["JoinDate"], errors="coerce"
)
```

`errors="coerce"` converts values that cannot be parsed into missing values, which should then be inspected and handled.

### 5.4 Inconsistent categories

Values such as `"Male"`, `"male"`, and `"M"` may represent the same category. Standardize labels before encoding them.

```python
df["Gender"] = df["Gender"].replace({
    "male": "Male",
    "M": "Male",
    "female": "Female",
    "F": "Female"
})
```

Only apply mappings that are correct for the dataset.

### 5.5 Outliers

An **outlier** is an observation unusually far from the rest of the data. It may be a data-entry error or a genuine rare event.

#### IQR method

1. Find the first quartile, \(Q_1\).
2. Find the third quartile, \(Q_3\).
3. Calculate \(IQR = Q_3-Q_1\).
4. Lower fence: \(Q_1-1.5(IQR)\).
5. Upper fence: \(Q_3+1.5(IQR)\).
6. Values outside the fences are flagged as potential outliers.

**Worked example**

Suppose \(Q_1=30\) and \(Q_3=70\).

- \(IQR=70-30=40\)
- Lower fence \(=30-1.5(40)=-30\)
- Upper fence \(=70+1.5(40)=130\)

A value of 140 is flagged as a potential outlier. It should be investigated before deciding what to do with it.

#### Z-score method

\[
z=\frac{x-\mu}{\sigma}
\]

A value with \(|z|>3\) is sometimes flagged for investigation. This rule is a guide, not proof that a value is incorrect.

#### What can be done with outliers?

- Correct the value if it is a verified data-entry error.
- Keep it if it is a valid observation.
- Cap or winsorize it when justified.
- Remove it only when there is a defensible reason.

Example code to identify IQR outliers:

```python
q1 = df["MonthlyCharges"].quantile(0.25)
q3 = df["MonthlyCharges"].quantile(0.75)
iqr = q3 - q1

outliers = df[
    (df["MonthlyCharges"] < q1 - 1.5 * iqr) |
    (df["MonthlyCharges"] > q3 + 1.5 * iqr)
]

print("Potential outliers:", len(outliers))
```

### Cleaning checklist

- [ ] Check missing values.
- [ ] Check duplicate records.
- [ ] Verify data types.
- [ ] Standardize category labels.
- [ ] Investigate outliers and impossible values.
- [ ] Record the cleaning decisions.
- [ ] Apply the same learned preprocessing to training and test data without leaking test information.

---

## 6. Data Transformation and Preprocessing

### Definition

**Data transformation** changes the format, scale, or representation of data so it is suitable for analysis and machine learning.

### 6.1 Feature scaling

Features can have very different numerical ranges. For example, age may range from 18 to 80, while income may range from thousands to millions. Some algorithms are affected by these differences.

#### Standardization

\[
z=\frac{x-\mu}{\sigma}
\]

It subtracts the mean and divides by the standard deviation. The transformed feature has mean near 0 and standard deviation near 1 when calculated on the fitting data.

Often useful for models such as SVM, k-NN, PCA, and neural networks.

#### Min-max scaling

\[
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

This maps values to the range \([0,1]\), provided the maximum and minimum differ.

#### Robust scaling

\[
x'=\frac{x-\text{median}}{IQR}
\]

This uses the median and interquartile range and can be more resistant to outliers.

| Method | Main idea | Often useful when |
|---|---|---|
| Standardization | Centre and scale by standard deviation | Standard-scaled features are appropriate for the model |
| Min-max scaling | Map to a fixed range | A bounded range such as 0–1 is desired |
| Robust scaling | Use median and IQR | Outliers may strongly affect mean and standard deviation |

### 6.2 Encoding categorical variables

Machine-learning algorithms often need categories represented numerically.

#### Label encoding

Maps categories to integer values. It is appropriate for genuinely ordered categories.

Example:

| Education level | Encoded value |
|---|---:|
| Low | 0 |
| Medium | 1 |
| High | 2 |

#### One-hot encoding

Creates a separate 0/1 column for each category. It is often appropriate for categories without a natural order.

Example:

| Payment method | Card | Cash | Transfer |
|---|---:|---:|---:|
| Card | 1 | 0 | 0 |
| Cash | 0 | 1 | 0 |
| Transfer | 0 | 0 | 1 |

Assigning arbitrary numbers to unordered categories can incorrectly imply an order. For example, coding payment methods as 0, 1, and 2 does not mean one payment method is greater than another.

```python
df = pd.get_dummies(
    df,
    columns=["PaymentMethod"],
    drop_first=True,
    dtype=int
)
```

#### Target and frequency encoding

- **Target encoding:** Replaces a category with a statistic calculated from the target.
- **Frequency encoding:** Replaces a category with how often it occurs.

These methods can be useful for high-cardinality categories, but target encoding must be implemented carefully to avoid leakage.

### 6.3 Binning

Binning groups a continuous variable into ranges.

Example: Age may be grouped into young, middle-aged, and senior categories.

Binning can make patterns easier to interpret or capture certain non-linear patterns, but it also loses information and can create arbitrary boundaries.

### 6.4 Transformations

Log or Box-Cox transformations may help reduce strong skewness in suitable numerical data. The transformation should match the data and model assumptions.

### 6.5 Feature engineering

**Feature engineering** means creating useful new variables from existing data.

Examples:

- Average monthly spend = `TotalCharges / tenure`, when tenure is positive.
- Tenure groups based on months with a company.
- Transaction frequency over a selected time period.
- Ratios that represent meaningful business quantities.

Good features can improve a model even when the algorithm itself remains unchanged.

### 6.6 Train/test split

A common split is 80% training and 20% testing, or 70% training and 30% testing.

- **Training set:** Used to fit the model.
- **Test set:** Held aside to estimate performance on unseen data.

For imbalanced classification, stratified splitting can preserve the target-class proportions in both sets.

---

## 7. Train-Test Split and Data Leakage

### What is data leakage?

**Data leakage** occurs when information that should not be available during training is used to build or evaluate the model. It can make test scores look better than the model’s true performance.

### Example of leakage

If the mean and standard deviation for scaling are calculated using the complete dataset before splitting, information from the test set influences preprocessing.

### Correct order

1. Split the data into training and test sets.
2. Fit the imputer, encoder, and scaler using training data only.
3. Apply the fitted transformations to the test data.
4. Train the model on the transformed training data.
5. Evaluate once on the held-out test set.

A scikit-learn `Pipeline` helps apply the same transformations consistently.

**Rule to remember:** Fit preprocessing on training data; use the fitted preprocessing to transform test data.

---

## 8. Feature Selection

### Definition

**Feature selection** is the process of choosing a subset of the most useful original input variables for a model.

### Why is it needed?

Too many irrelevant or redundant features can:

- Add noise.
- Increase training time.
- Increase the risk of overfitting.
- Make a model harder to explain.
- Create problems related to high dimensionality.

### Three families of feature selection

| Family | How it works | Examples | Main advantage / limitation |
|---|---|---|---|
| Filter | Scores features independently of a particular model | Correlation, chi-square, ANOVA F, mutual information | Fast; may miss feature interactions |
| Wrapper | Searches feature subsets by model performance | Recursive Feature Elimination (RFE), forward/backward selection | Can find useful subsets but is computationally expensive |
| Embedded | Selects features as part of model fitting | Lasso (L1), tree/forest feature importance | Efficient, but tied to the model and its assumptions |

#### Filter examples

- Correlation can rank numerical features by their linear relationship with a numerical target.
- Chi-square can be used for suitable non-negative categorical/count-like feature representations in classification.
- ANOVA F can score numerical features against a categorical class target.
- Mutual information can measure broader statistical dependence.

Feature scores are not proof of causation.

### Redundant features

If two features have a correlation of \(r=0.95\), they may contain highly overlapping information. Depending on the task and model, keeping both may add little value and can make interpretation more difficult.

### Feature selection vs. dimensionality reduction

These are different:

- **Feature selection:** Keeps some original features.
- **Dimensionality reduction:** Creates a smaller set of new representations.

### Principal Component Analysis (PCA)

PCA creates new, uncorrelated variables called **principal components**. Each component is a linear combination of the original features. Components are ordered by the amount of variance they explain.

PCA can reduce dimensions and help visualize data, but the new components may be harder to interpret than the original variables.

### Example with scikit-learn

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=3)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

Fit the selector on training data only. Do not select features using the test labels.

---

## 9. Supervised Learning

### Definition

**Supervised learning** learns a relationship between input features \(X\) and a known target \(y\) using labelled examples. The learned model predicts the target for new examples.

Two main types are regression and classification.

### Regression

Regression predicts a numerical value.

Examples:

- Predicting house prices.
- Predicting monthly customer charges.
- Predicting energy consumption.

### Classification

Classification predicts a category or class.

Examples:

- Predicting whether a customer will leave: yes/no.
- Classifying an email as spam or not spam.
- Flagging a transaction as potentially fraudulent.

| Regression | Classification |
|---|---|
| Target is numerical | Target is categorical |
| Example: predict price | Example: predict churn |
| Metrics: MAE, RMSE, \(R^2\) | Metrics: precision, recall, F1, ROC-AUC, confusion matrix |

### Common supervised algorithms

| Algorithm | Basic idea | Strength / consideration |
|---|---|---|
| Linear Regression | Fits a weighted linear relationship | Simple numerical baseline |
| Logistic Regression | Estimates class probabilities | Often interpretable baseline for classification |
| Decision Tree | Uses feature-based splits | Easy to explain; a deep tree can overfit |
| Random Forest | Combines predictions from many trees | Strong general-purpose choice for tabular data |
| k-Nearest Neighbours (k-NN) | Uses nearby examples to predict | Requires appropriate scaling; prediction can be slow |
| Support Vector Machine (SVM) | Finds a separating boundary with a large margin | Can work well in high-dimensional data; scaling is often important |
| Naive Bayes | Uses Bayes’ rule with conditional-independence assumptions | Fast and useful for some text tasks |

The best algorithm depends on the data, target, performance needs, and interpretability requirements.

---

## 10. Model Evaluation Metrics

A model should be judged with metrics that match the task and the cost of errors.

### 10.1 Regression metrics

Let \(y_i\) be the actual value and \(\hat{y}_i\) the predicted value.

#### Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

It measures the average absolute prediction error. Lower is generally better.

#### Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE penalizes large errors more heavily than MAE. It is expressed in the target’s units.

#### Coefficient of determination (\(R^2\))

\[
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
\]

It compares the model’s squared errors with the baseline of predicting the mean. On a typical held-out regression set, a larger value can indicate a better fit, but \(R^2\) can be negative when the model performs worse than that baseline.

### 10.2 Confusion matrix

For binary classification:

| | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | True Positive (TP) | False Negative (FN) |
| Actual negative | False Positive (FP) | True Negative (TN) |

- **TP:** Correctly predicted positive.
- **TN:** Correctly predicted negative.
- **FP:** Predicted positive, but actually negative.
- **FN:** Predicted negative, but actually positive.

### 10.3 Classification metrics

#### Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

The fraction of all predictions that are correct.

#### Precision

\[
Precision=\frac{TP}{TP+FP}
\]

Of all cases predicted positive, what fraction were actually positive?

#### Recall

\[
Recall=\frac{TP}{TP+FN}
\]

Of all actual positive cases, what fraction did the model find?

#### F1-score

\[
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
\]

F1 is the harmonic mean of precision and recall.

### Worked classification example

Suppose a churn model gives:

- TP = 160
- FP = 40
- FN = 30
- TN = 270

Total = 500.

- Accuracy = \((160+270)/500=0.86\), or 86%.
- Precision = \(160/(160+40)=0.80\), or 80%.
- Recall = \(160/(160+30)\approx0.84\), or 84%.
- F1 is approximately 0.82.

### Why accuracy can mislead

Suppose only 2% of bank transactions are fraudulent. A model that predicts “not fraud” for every transaction would be 98% accurate but would catch no fraud.

For imbalanced data, examine precision, recall, F1, and suitable threshold-independent metrics as well as accuracy.

### ROC-AUC and decision threshold

- **ROC curve:** Shows the trade-off between true-positive rate and false-positive rate across decision thresholds.
- **ROC-AUC:** Summarizes ranking discrimination across thresholds.
- **Decision threshold:** The probability cutoff used to classify an observation as positive.

Changing the threshold can change precision and recall. The right threshold depends on the costs of false positives and false negatives.

### Choosing metrics for the domain

- **Healthcare screening:** Recall may be especially important because missing a high-risk patient can be costly.
- **Banking fraud:** Precision matters when false alarms cause legitimate transactions to be blocked, but recall matters too because missed fraud is costly.
- **Telecom churn:** Precision and recall should be considered alongside the cost and benefit of customer-retention offers.
- **HR attrition:** Evaluate performance and audit fairness; predictions should not be treated as unquestionable decisions about employees.

### Python example

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print(classification_report(y_test, y_pred, zero_division=0))
```

---

## 11. Algorithms, Overfitting, and Validation

### 11.1 Decision tree impurity

Decision trees choose splits that make the resulting groups more pure.

#### Gini impurity

\[
Gini=1-\sum_{i=1}^{k}p_i^2
\]

#### Entropy

\[
Entropy=-\sum_{i=1}^{k}p_i\log_2(p_i)
\]

Here, \(p_i\) is the proportion of class \(i\) in the node. A node containing only one class has Gini impurity and entropy equal to zero.

**Worked example:** A node has 8 “yes” and 2 “no” observations. The class proportions are 0.8 and 0.2.

\[
Gini=1-(0.8^2+0.2^2)=1-0.68=0.32
\]

\[
Entropy=-[0.8\log_2(0.8)+0.2\log_2(0.2)]\approx0.72
\]

### 11.2 Underfitting and overfitting

- **Underfitting:** The model is too simple to learn important patterns. Training and test performance are both poor. This is often associated with high bias.
- **Overfitting:** The model learns details and noise specific to the training data. Training performance is good but test performance is worse. This is often associated with high variance.
- **Generalization:** The ability to perform well on new, unseen data.

The goal is a model that captures useful patterns without memorizing the training examples.

### Ways to reduce overfitting

- Limit tree depth.
- Use regularization.
- Use cross-validation to compare model settings.
- Collect more representative data where possible.
- Remove leakage and review feature quality.
- Use ensemble methods where appropriate.
- Use early stopping for algorithms that support it.

### 11.3 k-fold cross-validation

Cross-validation estimates model performance more reliably than relying on a single train/validation split.

For **k-fold cross-validation**:

1. Divide the training data into \(k\) folds.
2. Train on \(k-1\) folds.
3. Validate on the remaining fold.
4. Repeat until every fold has served as the validation fold.
5. Average the validation scores.

For example, in 5-fold cross-validation, the model is trained and validated five times using different held-out folds.

### 11.4 Hyperparameters and tuning

Hyperparameters are settings chosen before or during the model-selection process rather than learned directly as ordinary model parameters.

Examples:

- Maximum depth of a decision tree.
- Number of neighbours in k-NN.
- Number of trees in a Random Forest.
- Regularization strength.

`GridSearchCV` tests specified combinations. `RandomizedSearchCV` samples combinations from a search space.

**Important:** Tune using training data and cross-validation. Do not repeatedly tune against the final test set.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 5, 8, None],
    "n_estimators": [100, 200]
}

search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best CV score:", search.best_score_)
```

Use a scoring metric that matches the actual task.

---

## 12. Unsupervised Learning

### Definition

**Unsupervised learning** finds patterns or structure in data without a known target label.

Main tasks include:

- Clustering similar observations.
- Reducing the number of dimensions.
- Detecting unusual observations or anomalies.

### 12.1 Clustering

Clustering groups observations based on similarity.

Example: A telecom company can group customers into segments such as long-term loyal customers, price-sensitive customers, and customers who may be at risk of leaving.

### 12.2 Common clustering algorithms

| Algorithm | Basic idea | Important notes |
|---|---|---|
| k-Means | Assigns points to the nearest of \(k\) centroids and updates the centroids | Must choose \(k\); scaling is often important; works best for suitable cluster shapes |
| Hierarchical clustering | Builds a hierarchy of nested clusters, often shown as a dendrogram | Helps explore nested groupings; choices of distance/linkage matter |
| DBSCAN | Finds dense groups and labels low-density points as noise | Can identify irregular shapes and noise; sensitive to parameters and density variation |

### 12.3 k-Means objective

k-Means minimizes the within-cluster sum of squares:

\[
WCSS=\sum_{j=1}^{k}\sum_{x\in C_j}\|x-\mu_j\|^2
\]

where \(C_j\) is cluster \(j\) and \(\mu_j\) is its centroid.

Basic steps:

1. Choose \(k\) initial centroids.
2. Assign each point to its nearest centroid.
3. Recalculate the centroid of each cluster.
4. Repeat assignment and recalculation until the assignments or centroids stop changing substantially.

### 12.4 Elbow method

1. Fit k-Means for several values of \(k\).
2. Record WCSS for each value.
3. Plot WCSS against \(k\).
4. Look for the point where the reduction in WCSS starts to flatten—the “elbow.”

WCSS decreases as \(k\) increases, so the goal is not simply to select the largest \(k\).

### 12.5 Silhouette score

The silhouette score evaluates how well observations fit within their assigned cluster compared with nearby clusters. Higher values generally indicate better separation, but the score should be considered with domain meaning and the structure of the data.

### 12.6 Dimensionality reduction

- **PCA:** Linear transformation into principal components; useful for compact representation and visualization.
- **t-SNE / UMAP:** Non-linear methods commonly used to visualize high-dimensional data in two or three dimensions. Visual distances should be interpreted carefully.

### 12.7 Anomaly detection

Anomaly detection identifies observations that differ substantially from typical patterns.

Methods mentioned in the unit include:

- Isolation Forest.
- DBSCAN noise points.
- Z-score-based flags such as \(|z|>3\), where appropriate.

Applications include potentially fraudulent banking transactions, unusual health measurements, and equipment faults. A flagged anomaly is not automatically an error or fraud; it needs context and validation.

### Example: k-Means in Python

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = df[["tenure", "MonthlyCharges", "TotalCharges"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Compare several k values
wcss = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

print("WCSS:", wcss)

# Example segmentation with k=3
km = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)

print(df["cluster"].value_counts())
```

For a proper evaluation, fit scaling and clustering on the intended analysis data and document the choices. Do not use classification accuracy to evaluate a clustering result when there are no known labels.

---

## 13. EDA in HR, Telecom, Healthcare, and Banking

The EDA pipeline stays broadly similar across domains, but the target variable, important features, and cost of errors change.

| Domain | Typical target | Useful features | Main cautions |
|---|---|---|---|
| HR | Employee attrition | Tenure, salary, satisfaction, overtime, promotions | Class imbalance, fairness, privacy, explainability |
| Telecom | Customer churn | Tenure, monthly charges, contract, complaints, usage | Missing values, churn imbalance |
| Healthcare | Diagnosis or risk | Age, vital signs, laboratory values, medical history | Missing/impossible values, clinical validation, recall |
| Banking | Loan default or fraud | Income, balance, credit history, transaction features | Rare events, precision/recall trade-offs, fairness, regulatory explainability |

### Typical findings

#### HR — Attrition

EDA may reveal that employees who leave have lower tenure or satisfaction and higher overtime. Useful views include box plots of satisfaction by attrition and correlation heatmaps. Patterns are associations, not proof that a feature causes attrition.

#### Telecom — Churn

Customers who leave may have shorter tenure, month-to-month contracts, or higher monthly charges. Grouped statistics and a chi-square test for contract type versus churn can help investigate associations.

#### Healthcare — Risk

Some measurements may be skewed, and a zero may represent a missing value rather than a real measurement. The data dictionary and clinical context must be checked. Recall may be important for screening, but model results require proper clinical validation.

#### Banking — Default or fraud

The positive class may be rare. EDA should inspect class balance, unusual transactions, and relationships among features. Explainability and fairness matter because decisions can affect people’s access to financial services.

### Five EDA moves common to all four domains

1. Profile the dataset: shape, types, and missing values.
2. Examine the target balance.
3. Compare feature distributions by target.
4. Check correlations and categorical associations.
5. Investigate outliers, impossible values, and leakage.

---

## 14. Complete Telecom Churn Case Study

### 14.1 Problem statement

A telecom company wants to understand customer churn and predict which customers may leave. The unit demonstrates the full pipeline using a synthetic dataset.

**Important:** Synthetic data is generated for learning and does not represent actual company customers.

### 14.2 Case-study stages

1. Create or load the dataset.
2. Explore the data and calculate the churn rate.
3. Compare churners and non-churners.
4. Rank features using an ANOVA F filter.
5. Split the data into training and test sets.
6. Train a Random Forest classifier.
7. Evaluate its predictions.
8. Apply k-Means to identify customer segments.
9. Interpret the results for a possible retention campaign.

### 14.3 Complete Python example

Save as `eda_case_study.py`. Install the required libraries first with `pip install numpy pandas scikit-learn`.

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Create a reproducible synthetic Telecom dataset
np.random.seed(42)
n = 1000

tenure = np.random.randint(1, 73, n)
monthly = np.round(np.random.uniform(20, 120, n), 2)
total = np.round(
    tenure * monthly * np.random.uniform(0.8, 1.0, n), 2
)
contract = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])

# Create a churn probability from selected features
logit = (
    -2.0
    + 0.04 * (50 - tenure)
    + 0.02 * (monthly - 70)
    + 0.9 * (contract == 0)
)
probability = 1 / (1 + np.exp(-logit))
churn = (np.random.uniform(0, 1, n) < probability).astype(int)

df = pd.DataFrame({
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "Churn": churn
})

# 2. First-look EDA
print("Dataset shape:", df.shape)
print("Churn rate:", round(df["Churn"].mean() * 100, 1), "%")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print("\nMean values grouped by churn:")
print(
    df.groupby("Churn")[["tenure", "MonthlyCharges"]]
      .mean()
      .round(1)
)

# 3. Separate features and target
X = df.drop(columns="Churn")
y = df["Churn"]

# 4. Feature scoring (fit on training data after split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)

feature_scores = pd.Series(
    selector.scores_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nANOVA F feature scores:")
print(feature_scores.round(2))

# 5. Train a supervised classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, pred), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("Classification report:")
print(classification_report(y_test, pred, zero_division=0))

# 6. Unsupervised customer segmentation
# Scale the clustering features because k-Means uses distances.
cluster_features = df[["tenure", "MonthlyCharges", "TotalCharges"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

print("\nChurn rate by cluster:")
print(df.groupby("cluster")["Churn"].mean().round(2))
```

### 14.4 How to interpret results

The uploaded unit’s illustrative output reports:

- 1,000 synthetic records and a churn rate of 31.4%.
- Mean tenure is lower for churners than for non-churners.
- Mean monthly charges are higher for churners in this generated dataset.
- Tenure receives the highest ANOVA F feature score in the example.
- The Random Forest reaches about 73% test accuracy in the shown run.
- Churn recall is about 0.46 in the displayed classification report.
- The k-Means result includes a segment with a churn rate around 0.47 and another around 0.17.

The numbers are examples from the unit, not guaranteed outputs from every version of code or library. The dataset is synthetic and the results should not be interpreted as real-world evidence.

### 14.5 Business interpretation

- Shorter tenure may be a useful signal for churn prediction.
- A segment with higher observed churn could be considered for further analysis and a retention experiment.
- Accuracy alone is insufficient; recall on the churn class is also important.
- The company should test whether a retention campaign actually improves outcomes before assuming that the model’s segments are useful.

### 14.6 Adapting the case study to other domains

| Domain | Target | Supervised goal | Unsupervised goal |
|---|---|---|---|
| Telecom | Churn | Predict customer cancellation | Customer segmentation |
| HR | Attrition | Predict employee attrition for careful analysis | Employee group exploration |
| Healthcare | Diagnosis/risk | Estimate risk with appropriate clinical validation | Explore patient subgroups |
| Banking | Default/fraud | Flag potential risk | Explore spending groups or anomalies |

Change the dataset, target column, preprocessing, and evaluation metric according to the actual domain. Do not simply reuse a metric without considering the consequences of each type of error.

---

## 15. Ten Hands-On Python Labs

These labs follow the themes in the Unit V source. They can be performed in Jupyter Notebook, Google Colab, or VS Code. Use a dataset with the columns mentioned in each lab or adjust the code to match your own column names.

### Lab 1 — First-look EDA

**Objective:** Understand the size, types, missingness, and target balance.

```python
import pandas as pd

df = pd.read_csv("churn.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["Churn"].value_counts(normalize=True))
```

**Deliverable:** A short note on dataset size, column types, missing values, and target balance.

### Lab 2 — Cleaning

**Objective:** Check duplicates, parse a numeric column, and identify potential outliers.

```python
df = df.drop_duplicates()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

q1 = df["MonthlyCharges"].quantile(0.25)
q3 = df["MonthlyCharges"].quantile(0.75)
iqr = q3 - q1

outliers = df[
    (df["MonthlyCharges"] < q1 - 1.5 * iqr) |
    (df["MonthlyCharges"] > q3 + 1.5 * iqr)
]

print("Potential outliers:", len(outliers))
```

**Deliverable:** A cleaned dataset and a record of each cleaning decision. Median imputation is an example choice; verify that it is appropriate for your data.

### Lab 3 — Visual EDA

**Objective:** Make univariate and bivariate plots.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x="tenure", kde=True)
plt.title("Distribution of Tenure")
plt.show()

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.show()

numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

**Deliverable:** Three plots and one observation from each. Correlation does not prove causation.

### Lab 4 — Encoding and scaling

**Objective:** Encode categories and scale numerical data without leakage.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_encoded = pd.get_dummies(
    df,
    columns=["PaymentMethod"],
    drop_first=True,
    dtype=int
)

X = df_encoded.drop(columns="Churn")
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Deliverable:** Training and test matrices, plus an explanation of why the scaler is fitted on training data only.

> If the dataset contains other categorical columns, encode them appropriately too. For a mixed-type production dataset, a `ColumnTransformer` and `Pipeline` are usually safer than manually applying one scaler to every column.

### Lab 5 — Feature selection

**Objective:** Rank features with an ANOVA F filter and inspect Random Forest feature importance.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)

scores = pd.Series(
    selector.scores_,
    index=X_train.columns
).sort_values(ascending=False)

print("ANOVA F scores:")
print(scores)

forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)

importance = pd.Series(
    forest.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("Random Forest feature importance:")
print(importance.head(10))
```

**Deliverable:** Compare the rankings and discuss why different methods may produce different results. Run this after appropriate encoding, and fit feature selection using training data only.

### Lab 6 — Classification model and metrics

**Objective:** Train a classifier and evaluate it.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
```

**Deliverable:** Confusion matrix, precision, recall, F1, and a short interpretation of the error costs.

### Lab 7 — Cross-validation and tuning

**Objective:** Estimate generalization and tune a model hyperparameter.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

base_model = RandomForestClassifier(random_state=42)

scores = cross_val_score(
    base_model, X_train, y_train,
    cv=5, scoring="f1"
)
print("Five-fold F1 scores:", scores.round(3))
print("Mean CV F1:", scores.mean().round(3))

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"max_depth": [3, 5, 8, None]},
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)
```

**Deliverable:** Fold scores and the best hyperparameter. Keep the final test set out of the tuning process.

### Lab 8 — Regression metrics

**Objective:** Predict a numerical target and measure the errors.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Replace "NumericTarget" with an appropriate numerical target.
X_reg = df.drop(columns=["NumericTarget"])
y_reg = df["NumericTarget"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=42
)

reg = LinearRegression()
reg.fit(Xr_train, yr_train)
predictions = reg.predict(Xr_test)

rmse = np.sqrt(mean_squared_error(yr_test, predictions))
print("MAE:", mean_absolute_error(yr_test, predictions))
print("RMSE:", rmse)
print("R-squared:", r2_score(yr_test, predictions))
```

**Deliverable:** MAE, RMSE, and \(R^2\), with one sentence explaining what each says about the predictions. Encode categorical features before using this model.

### Lab 9 — Clustering and the elbow method

**Objective:** Explore customer segments without target labels.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = df[["tenure", "MonthlyCharges", "TotalCharges"]]
X_scaled = StandardScaler().fit_transform(features)

wcss = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

print("WCSS for k=1..7:", wcss)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)

print(df["cluster"].value_counts())
print(df.groupby("cluster")[["tenure", "MonthlyCharges"]].mean())
```

**Deliverable:** WCSS values, a reasoned choice of \(k\), and a plain-language profile for each cluster. If you use the target later to describe cluster churn rates, do not use it as an input to the clustering if the goal is unsupervised segmentation.

### Lab 10 — End-to-end pipeline

**Objective:** Put preprocessing and model training into a reusable pipeline.

For a dataset that contains only numeric features, a simple example is:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

pipe.fit(X_train, y_train)
print("Test accuracy:", round(pipe.score(X_test, y_test), 3))
```

A Random Forest does not usually require feature scaling, so the scaler here mainly demonstrates the pipeline concept. For mixed numerical and categorical data, use a `ColumnTransformer` with appropriate imputers, encoders, and model steps. Use scaling when the chosen model needs it.

**Deliverable:** A reusable pipeline, held-out evaluation, and a short report describing the most useful findings and limitations.

---

## 16. Mini Projects and Capstone Ideas

### Project 1 — HR Attrition EDA Report (Foundational)

**Goal:** Explore an employee dataset to understand patterns associated with attrition.

Tasks:
- Profile columns and missing values.
- Compare employees who leave and stay.
- Examine tenure, satisfaction, salary, overtime, and promotions.
- Create relevant plots and a correlation heatmap.
- Summarize findings without claiming causation.

**Tools:** Pandas, Seaborn, Matplotlib.

**Outcome:** An insight report that can help HR ask better questions.

### Project 2 — Telecom Churn Predictor (Foundational)

**Goal:** Predict customer churn.

Tasks:
- Clean and encode data.
- Select features.
- Train a classification model.
- Report confusion matrix, precision, recall, and F1.
- Discuss a suitable decision threshold and retention costs.

**Outcome:** A churn model and a cost-aware evaluation report.

### Project 3 — Healthcare Risk Screening (Intermediate)

**Goal:** Explore a health dataset and build a risk-screening prototype.

Tasks:
- Check missing data and impossible values.
- Impute missing values appropriately.
- Train a classification model.
- Evaluate recall and precision.
- Explain the cost of false negatives and false positives.
- Clearly state that a student model is not a substitute for clinical validation.

**Outcome:** A carefully evaluated educational screening prototype.

### Project 4 — Customer Segmentation (Intermediate)

**Goal:** Group customers without pre-existing segment labels.

Tasks:
- Scale suitable numerical features.
- Compare k values using the elbow method and silhouette score.
- Profile each cluster.
- Give clusters descriptive business names only after examining their properties.
- Recommend a possible action for each segment and how it could be tested.

**Outcome:** A segmentation report.

### Project 5 — Banking Default/Fraud Model with Fairness Review (Advanced)

**Goal:** Build and evaluate a model for a rare financial-risk event.

Tasks:
- Explore class imbalance and outliers.
- Build an appropriate baseline.
- Evaluate precision, recall, and other relevant metrics.
- Inspect feature importance or use an explanation method such as SHAP.
- Audit performance across relevant groups when appropriate and lawful.
- Discuss limitations and avoid treating the model as an unquestionable decision-maker.

**Outcome:** An explainable risk-model report with fairness considerations.

### Project 6 — Multi-Domain EDA Engine (Advanced Capstone)

**Goal:** Create a reusable tool that explores and models datasets from HR, Telecom, Healthcare, or Banking.

Possible features:
- Automatically report dataset shape, types, missing values, and target balance.
- Generate selected plots.
- Offer configurable cleaning and preprocessing.
- Perform feature selection.
- Train and cross-validate a suitable supervised model.
- Apply clustering when useful.
- Produce a domain-aware summary with metrics and limitations.

**Outcome:** A portfolio project demonstrating the full EDA-to-model workflow.

---

## 17. Formula Sheet

| Topic | Formula / rule |
|---|---|
| Interquartile range | \(IQR=Q_3-Q_1\) |
| IQR lower fence | \(Q_1-1.5(IQR)\) |
| IQR upper fence | \(Q_3+1.5(IQR)\) |
| Z-score | \(z=(x-\mu)/\sigma\) |
| Standardization | \(z=(x-\mu)/\sigma\) |
| Min-max scaling | \(x'=(x-x_{\min})/(x_{\max}-x_{\min})\) |
| Robust scaling | \(x'=(x-\text{median})/IQR\) |
| MAE | \(\frac{1}{n}\sum |y_i-\hat y_i|\) |
| RMSE | \(\sqrt{\frac{1}{n}\sum(y_i-\hat y_i)^2}\) |
| \(R^2\) | \(1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}\) |
| Accuracy | \((TP+TN)/(TP+TN+FP+FN)\) |
| Precision | \(TP/(TP+FP)\) |
| Recall | \(TP/(TP+FN)\) |
| F1 | \(2PR/(P+R)\) |
| Gini impurity | \(1-\sum p_i^2\) |
| Entropy | \(-\sum p_i\log_2(p_i)\) |
| k-Means WCSS | \(\sum_j\sum_{x\in C_j}\|x-\mu_j\|^2\) |

---

## 18. Common Mistakes

1. **Starting modelling without EDA:** First check structure, missingness, distributions, and target balance.
2. **Deleting every outlier:** Investigate whether it is an error or a genuine rare observation.
3. **Using the mean for every missing numerical value:** Consider skewness, outliers, and why values are missing.
4. **Encoding nominal categories as ordered numbers:** Use an appropriate encoding such as one-hot encoding when categories have no natural order.
5. **Scaling before splitting:** Split first; fit the scaler on training data only.
6. **Selecting features using all data:** Fit feature selection on training data, not the held-out test labels.
7. **Using accuracy alone on imbalanced data:** Also inspect precision, recall, F1, and relevant curves.
8. **Using the test set to tune the model repeatedly:** Use validation or cross-validation for model selection.
9. **Clustering without scaling distance-based features:** Features with large ranges can dominate distances.
10. **Choosing \(k\) only because WCSS is lowest:** WCSS decreases as \(k\) rises; use the elbow, silhouette, and domain meaning.
11. **Treating association as causation:** A relationship in EDA does not prove that one variable causes another.
12. **Treating a synthetic case-study score as a real-world benchmark:** Test on representative real data before drawing practical conclusions.
13. **Ignoring fairness and explainability:** Consider the consequences of model decisions, especially in HR, Healthcare, and Banking.
14. **Using the same metric for every problem:** Match the evaluation to the target and error costs.

---

## 19. Viva Questions with Short Answers

### Q1. What is EDA?
EDA is the process of exploring data using statistics and visualizations before modelling.

### Q2. Who is associated with EDA?
John Tukey.

### Q3. Name the three levels of EDA.
Univariate, bivariate, and multivariate analysis.

### Q4. What is data cleaning?
Finding and handling missing values, duplicates, incorrect types, inconsistent categories, and suspicious values.

### Q5. What is imputation?
Replacing missing values with reasonable estimates, such as the mean, median, mode, or model-based estimates.

### Q6. What is an outlier?
An observation that is unusually far from the rest of the data.

### Q7. What is feature scaling?
Changing numerical feature scales so that their ranges are more comparable.

### Q8. What is the difference between standardization and min-max scaling?
Standardization uses the mean and standard deviation; min-max scaling maps values to a chosen range such as 0 to 1.

### Q9. What is one-hot encoding?
It creates a separate binary column for each category.

### Q10. What is feature engineering?
Creating useful new variables from existing data.

### Q11. What is data leakage?
Using information in training or preprocessing that would not legitimately be available at prediction time, causing overly optimistic evaluation.

### Q12. What is feature selection?
Choosing a subset of useful original input variables.

### Q13. Name the three feature-selection families.
Filter, wrapper, and embedded methods.

### Q14. What is PCA?
A dimensionality-reduction method that creates ordered principal components from linear combinations of original features.

### Q15. What is supervised learning?
Learning from labelled examples with known target values.

### Q16. What is the difference between regression and classification?
Regression predicts a numerical value; classification predicts a category.

### Q17. What is a confusion matrix?
A table comparing actual and predicted classes using TP, TN, FP, and FN.

### Q18. Define precision and recall.
Precision measures how many predicted positives are correct; recall measures how many actual positives are found.

### Q19. What is overfitting?
When a model learns training-specific noise and performs worse on unseen data.

### Q20. What is cross-validation?
A method that repeatedly trains and validates on different folds to estimate performance more reliably.

### Q21. What is unsupervised learning?
Finding structure in data without a known target label.

### Q22. What is k-Means?
A clustering algorithm that groups points around \(k\) centroids by minimizing within-cluster squared distances.

### Q23. What is the elbow method?
A method that plots WCSS against the number of clusters and looks for where improvement starts to flatten.

### Q24. What is anomaly detection?
Identifying observations that differ substantially from typical patterns.

### Q25. Why can accuracy be misleading?
A model can achieve high accuracy by predicting only the majority class when the data are imbalanced.

### Q26. Why does the domain matter?
The cost of mistakes and the appropriate metric can differ between Healthcare, Banking, Telecom, and HR.

---

## 20. Practice and Bloom’s Taxonomy Questions

The source organizes questions across Bloom’s Taxonomy levels: **L1 Remember, L2 Understand, L3 Apply, L4 Analyze, L5 Evaluate, L6 Create.**

### Day 1 — EDA basics

- **L1:** Define EDA and name its originator.
- **L1:** List the three levels of EDA.
- **L2:** Explain univariate versus bivariate analysis with examples.
- **L3:** List the first five Pandas commands and explain their use.
- **L4:** Explain how skipping EDA can cause a model to fail.
- **L5:** Evaluate the statement “A good model makes EDA unnecessary.”
- **L6:** Design an EDA checklist for a new Banking dataset.

### Day 2 — Cleaning

- **L1:** List four common data-quality problems.
- **L1:** Match mean, median, and mode imputation to suitable types of data.
- **L2:** Explain why duplicates can bias a model.
- **L3:** For \(Q_1=30\) and \(Q_3=70\), calculate IQR fences and check whether 140 is flagged.
- **L4:** Explain the risks of mean-imputing a strongly skewed income variable.
- **L5:** Evaluate whether to drop or impute a column with 60% missing values.
- **L6:** Design a reusable function to report missing values, duplicates, and type problems.

### Day 3 — Transformation

- **L1:** Name three scaling methods.
- **L1:** Distinguish label encoding from one-hot encoding.
- **L2:** Explain data leakage.
- **L3:** Choose encodings for an ordered contract variable and an unordered payment-method variable.
- **L4:** Explain why a scaler should be fitted only on training data.
- **L5:** Evaluate one-hot encoding a column with 1,000 distinct identifiers.
- **L6:** Design a preprocessing pipeline for mixed numerical and categorical data.

### Day 4 — Feature selection

- **L1:** Name the three families of feature selection with examples.
- **L1:** State what PCA produces.
- **L2:** Explain feature selection versus PCA.
- **L3:** Choose a suitable filter method for a categorical feature and for a numerical feature versus a class target.
- **L4:** Explain why wrapper methods can be more computationally expensive.
- **L5:** Evaluate using PCA when a model must be explained to a loan applicant.
- **L6:** Design a feature-selection strategy combining a filter and an embedded method.

### Day 5 — Supervised learning and metrics

- **L1:** Define supervised learning and distinguish regression from classification.
- **L1:** Write the formulas for precision, recall, and F1.
- **L2:** Explain why accuracy can mislead on imbalanced data.
- **L3:** Calculate RMSE and \(R^2\) for a small set of actual and predicted values.
- **L3:** Compute precision, recall, and F1 from a confusion matrix.
- **L4:** Explain whether recall or precision should be emphasized for a particular use case.
- **L5:** Evaluate a fraud model with high accuracy but very low recall.
- **L6:** Design a metric and threshold plan for a Banking fraud detector.

### Day 6 — Algorithms and validation

- **L1:** Name four supervised algorithms and one strength of each.
- **L1:** Write Gini and entropy formulas.
- **L2:** Explain underfitting versus overfitting.
- **L3:** Calculate Gini impurity for a node with 6 positive and 4 negative samples.
- **L4:** Explain why a deep decision tree may overfit.
- **L5:** Evaluate a model with 99% training accuracy and 70% test accuracy.
- **L6:** Design a cross-validated experiment to tune a Random Forest.

### Day 7 — Unsupervised learning

- **L1:** Define unsupervised learning and name its main tasks.
- **L1:** Write the k-Means WCSS objective.
- **L2:** Explain the elbow method.
- **L3:** Given WCSS values 90, 50, 28, 24, and 22 for \(k=1,\ldots,5\), discuss where the elbow may be.
- **L4:** Explain why scaling matters before k-Means.
- **L5:** Explain why classification accuracy is not the usual measure for clustering without labels.
- **L6:** Design an anomaly-detection approach for Banking transactions and explain how you would validate it.

### Day 8 — Domain-specific EDA

- **L1:** State a typical target for HR, Telecom, Healthcare, and Banking.
- **L1:** List the five common EDA moves.
- **L2:** Explain why Healthcare screening may prioritize recall.
- **L3:** Suggest two bivariate plots for HR attrition.
- **L4:** Analyze fairness risks in an HR model.
- **L5:** Evaluate deploying an unexplained model for loan approval.
- **L6:** Design an EDA plan for a Banking default dataset.

### Day 9 — Full case study

- **L1:** Name the main stages of the case-study pipeline.
- **L2:** Explain what cluster-wise churn rates reveal for a retention campaign.
- **L3:** Adapt the pipeline to an HR attrition dataset.
- **L4:** Suggest ways to improve low recall on the churn class.
- **L5:** Evaluate whether 73% accuracy is good given the dataset’s churn rate.
- **L6:** Design an end-to-end notebook that outputs a domain-specific insight report.

---

## 21. Final Revision Checklist

Before an exam or practical, make sure you can:

- [ ] Define EDA and explain why it comes before modelling.
- [ ] Distinguish univariate, bivariate, and multivariate analysis.
- [ ] Use the five first-look Pandas commands.
- [ ] Handle missing values, duplicates, incorrect types, and outliers.
- [ ] Calculate IQR fences and a z-score.
- [ ] Explain standardization, min-max scaling, and robust scaling.
- [ ] Distinguish label encoding and one-hot encoding.
- [ ] Explain feature engineering and data leakage.
- [ ] Describe filter, wrapper, and embedded feature selection.
- [ ] Distinguish feature selection from PCA.
- [ ] Explain supervised learning, regression, and classification.
- [ ] Calculate MAE, RMSE, \(R^2\), accuracy, precision, recall, and F1.
- [ ] Explain the confusion matrix, Gini impurity, and entropy.
- [ ] Distinguish underfitting from overfitting.
- [ ] Explain k-fold cross-validation and hyperparameter tuning.
- [ ] Describe k-Means, hierarchical clustering, DBSCAN, and the elbow method.
- [ ] Explain how EDA changes across HR, Telecom, Healthcare, and Banking.
- [ ] Walk through the Telecom churn case study and discuss its limitations.
- [ ] Keep preprocessing and feature selection free from test-data leakage.

---

## 22. References

The uploaded Unit V notes list the following further reading:

1. John W. Tukey, *Exploratory Data Analysis*, Addison-Wesley.
2. Peter Bruce and Andrew Bruce, *Practical Statistics for Data Scientists*, O’Reilly.
3. Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani, *An Introduction to Statistical Learning*, Springer.
4. Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, O’Reilly.
5. Wes McKinney, *Python for Data Analysis*, O’Reilly.
6. Official documentation for Pandas, Seaborn, and scikit-learn.

---

## Final Summary

Unit V connects data exploration with practical machine learning. First, understand the dataset through EDA. Then clean and transform it, select useful features, and split the data carefully to avoid leakage. Use supervised learning when labels are available and unsupervised learning to explore groups or anomalies without labels. Finally, evaluate the results with metrics that reflect the actual problem, interpret findings in their domain, and document limitations.

**Remember:** Explore first, model second, and interpret the results carefully.
