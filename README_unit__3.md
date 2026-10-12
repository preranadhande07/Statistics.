# Statistics Notes for Data Science & Artificial Intelligence

## Detailed Unit Notes, Important Formulas, Worked Examples, and Python Practical

> **Source note:** The PDF used to prepare these notes is titled
> *Inferential Statistics for Data Science & Artificial Intelligence*
> and identifies the material as **Unit III**. These notes expand that
> material into a GitHub-ready study guide. They do not claim to be Unit
> I. If your syllabus labels a different chapter as Unit I, use the
> correct Unit I PDF to make a matching README.

------------------------------------------------------------------------

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Basic Terms](#2-basic-terms)
3.  [Random Variables and Probability
    Distributions](#3-random-variables-and-probability-distributions)
4.  [Expectation, Variance, and Standard
    Deviation](#4-expectation-variance-and-standard-deviation)
5.  [Normal Distribution](#5-normal-distribution)
6.  [Standard Normal Distribution and
    Z-Score](#6-standard-normal-distribution-and-z-score)
7.  [Sampling Distribution and Standard
    Error](#7-sampling-distribution-and-standard-error)
8.  [Central Limit Theorem](#8-central-limit-theorem)
9.  [Student's t-Distribution](#9-students-t-distribution)
10. [Point Estimation and Estimator
    Properties](#10-point-estimation-and-estimator-properties)
11. [Maximum Likelihood Estimation](#11-maximum-likelihood-estimation)
12. [Confidence Intervals for a Population
    Mean](#12-confidence-intervals-for-a-population-mean)
13. [Estimating a Population
    Proportion](#13-estimating-a-population-proportion)
14. [Worked Example: Battery
    Lifetimes](#14-worked-example-battery-lifetimes)
15. [Python Practical](#15-python-practical)
16. [Python Code Explained](#16-python-code-explained)
17. [Applications in Data Science and
    AI](#17-applications-in-data-science-and-ai)
18. [Common Mistakes to Avoid](#18-common-mistakes-to-avoid)
19. [Quick Revision Sheet](#19-quick-revision-sheet)
20. [Practice Questions](#20-practice-questions)
21. [Source](#21-source)

------------------------------------------------------------------------

## 1. Introduction

**Inferential statistics** is the branch of statistics that uses
information from a sample to make estimates, predictions, or decisions
about a larger population.

Studying every member of a population is often expensive,
time-consuming, or impossible. Instead, we collect data from a sample
and use statistical methods to draw conclusions about the population.
Because a sample may not perfectly represent the population, inferential
statistics also measures and communicates uncertainty.

### Example

A company manufactures thousands of batteries. It may not be practical
to test every battery. The company tests a sample of 10 batteries and
uses their lifetimes to estimate the average lifetime of all batteries
produced.

### Main purposes

-   Estimate unknown population values.
-   Measure uncertainty in an estimate.
-   Construct confidence intervals.
-   Compare groups or treatments.
-   Support decisions based on data.
-   Draw conclusions from surveys, experiments, and machine-learning
    evaluations.

### Descriptive statistics versus inferential statistics

  -----------------------------------------------------------------------
  Descriptive statistics              Inferential statistics
  ----------------------------------- -----------------------------------
  Summarizes the data that has been   Uses sample data to make
  collected.                          conclusions about a population.

  Examples include mean, median,      Examples include estimation,
  charts, and standard deviation.     confidence intervals, and
                                      hypothesis tests.

  Describes the observed sample.      Generalizes beyond the observed
                                      sample, with uncertainty.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. Basic Terms

### Population

The **population** is the complete group of people, objects, or
observations we want to study.

**Example:** All batteries manufactured by a factory in one month.

### Sample

A **sample** is a smaller group selected from the population for
measurement or analysis.

**Example:** Ten batteries selected from that month's production.

### Parameter

A **parameter** is a numerical value that describes a population. It is
usually fixed, but often unknown.

Common symbols:

-   Population mean: `μ`
-   Population variance: `σ²`
-   Population standard deviation: `σ`
-   Population proportion: `p`

### Statistic

A **statistic** is a numerical value calculated from sample data. It is
used to describe the sample or estimate a population parameter.

Common symbols:

-   Sample mean: `x̄`
-   Sample variance: `s²`
-   Sample standard deviation: `s`
-   Sample proportion: `p̂`

### Random sample

A random sample is selected using a process that gives population
members a defined chance of being chosen. Random sampling can help
reduce selection bias, although it does not automatically remove every
possible source of bias.

### Sampling error

Sampling error is the difference between a sample statistic and the
corresponding population parameter that occurs because only part of the
population was observed.

### Important distinction

A **parameter** describes the population; a **statistic** describes the
sample. For example, `μ` is the true population mean, while `x̄` is the
mean calculated from the sample.

------------------------------------------------------------------------

## 3. Random Variables and Probability Distributions

### Random variable

A **random variable** assigns a numerical value to each possible outcome
of a random experiment. It is commonly represented by a capital letter
such as `X`.

**Example:** If `X` is the number of defective items in a batch, its
value depends on the observed batch.

### Discrete random variable

A discrete random variable takes countable values, such as
`0, 1, 2, 3, ...`.

Examples: - Number of defective products. - Number of customers arriving
in one hour. - Number of heads in several coin tosses.

A **probability mass function (PMF)** gives the probability that a
discrete random variable takes a particular value:

`p(x) = P(X = x)`

For a valid PMF: - `p(x) ≥ 0` for every possible value. - The
probabilities of all possible values add up to `1`.

### Continuous random variable

A continuous random variable can take any value within an interval, such
as a measurement of time, temperature, or battery lifetime.

A **probability density function (PDF)** describes the distribution of a
continuous random variable. For a valid PDF: - `f(x) ≥ 0`. - The total
area under the density curve is `1`. - The probability of an interval is
the area under the curve over that interval.

`P(a ≤ X ≤ b) = ∫[a to b] f(x) dx`

For a continuous random variable, the probability at one exact point is
zero: `P(X = a) = 0`. Probabilities are calculated over intervals.

### Cumulative distribution function (CDF)

The **CDF** gives the probability that the random variable is less than
or equal to a value:

`F(x) = P(X ≤ x)`

The CDF is useful for finding cumulative probabilities and probabilities
over ranges.

For example:

`P(a < X ≤ b) = F(b) − F(a)`

### Quick comparison

  -----------------------------------------------------------------------
  Feature                 Discrete variable       Continuous variable
  ----------------------- ----------------------- -----------------------
  Values                  Countable               Any value within a
                                                  range

  Main function           PMF                     PDF

  Example                 Number of defective     Lifetime of a battery
                          items                   

  Probability             Probability assigned to Area under the density
                          individual values       curve
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Expectation, Variance, and Standard Deviation

### Expected value / mean

The **expected value** is the long-run average value predicted by a
probability distribution. It is represented by `E[X]`. For a population
distribution, it is often written as `μ`.

For a discrete random variable:

`E[X] = Σ x P(X = x)`

For a continuous random variable:

`E[X] = ∫ x f(x) dx`

The mean describes the centre or average of a distribution.

### Variance

Variance measures how spread out values are around the mean.

`Var(X) = E[(X − μ)²]`

For a population, variance is represented by `σ²`.

A larger variance means the observations tend to be more spread out from
the mean. Variance is expressed in squared units.

### Standard deviation

Standard deviation is the square root of variance:

`σ = √Var(X)`

It measures spread in the same units as the original data, so it is
often easier to interpret than variance.

### Sample formulas

For observations `x₁, x₂, ..., xₙ`, the sample mean is:

`x̄ = (x₁ + x₂ + ... + xₙ) / n`

The usual sample variance formula is:

`s² = Σ(xᵢ − x̄)² / (n − 1)`

The sample standard deviation is:

`s = √s²`

The `n − 1` denominator is used in the usual unbiased estimator of
population variance when estimating the population variance from a
sample.

### Small example

For the values `2, 4, 6`:

-   Mean: `(2 + 4 + 6) / 3 = 4`
-   Deviations from the mean: `−2, 0, 2`
-   Squared deviations: `4, 0, 4`
-   Sum of squared deviations: `8`

The population variance for these three values considered as the whole
population is `8/3`. The sample variance, treating them as a sample, is
`8/(3 − 1) = 4`.

------------------------------------------------------------------------

## 5. Normal Distribution

A **normal distribution** is a continuous probability distribution with
a symmetric, bell-shaped curve. It is widely used to model measurements
and sampling distributions.

Notation:

`X ~ N(μ, σ²)`

Here: - `μ` is the mean. - `σ²` is the variance. - `σ` is the standard
deviation.

### Main properties

1.  The curve is bell-shaped and symmetric about the mean.
2.  Mean = median = mode.
3.  The mean determines the centre of the distribution.
4.  The standard deviation determines how spread out the curve is.
5.  The total area under the curve equals `1`.
6.  The tails extend in both directions and approach, but do not touch,
    the horizontal axis.

### Empirical rule: 68--95--99.7

For an approximately normal distribution:

-   About **68%** of observations lie between `μ − σ` and `μ + σ`.
-   About **95%** lie between `μ − 2σ` and `μ + 2σ`.
-   About **99.7%** lie between `μ − 3σ` and `μ + 3σ`.

### Example

Suppose test scores are approximately normally distributed with mean
`70` and standard deviation `5`.

-   About 68% of scores lie from `65` to `75`.
-   About 95% lie from `60` to `80`.
-   About 99.7% lie from `55` to `85`.

These percentages are approximations and rely on the normal-distribution
assumption.

------------------------------------------------------------------------

## 6. Standard Normal Distribution and Z-Score

### Standard normal distribution

The **standard normal distribution** is a normal distribution with mean
`0` and standard deviation `1`.

`Z ~ N(0, 1)`

A value from a normal distribution can be converted to a standard score,
or **z-score**, to show its position relative to the mean.

### Z-score formula

`z = (x − μ) / σ`

Where: - `x` is the observed value. - `μ` is the population mean. - `σ`
is the population standard deviation.

### Interpretation

-   `z = 0`: the value equals the mean.
-   `z > 0`: the value is above the mean.
-   `z < 0`: the value is below the mean.
-   `z = 2`: the value is two standard deviations above the mean.
-   `z = −1`: the value is one standard deviation below the mean.

### Example

A measurement is `85`, the mean is `75`, and the standard deviation is
`5`.

`z = (85 − 75) / 5 = 2`

Therefore, the measurement is two standard deviations above the mean.

### Z-table

A standard normal table commonly provides the cumulative probability to
the left of a z-score:

`P(Z ≤ z)`

To find the probability above a z-score:

`P(Z > z) = 1 − P(Z ≤ z)`

To find the probability between two z-scores:

`P(a < Z ≤ b) = P(Z ≤ b) − P(Z ≤ a)`

### Common critical z-values

  Confidence level     Critical z-value (approximately)
  ------------------ ----------------------------------
  90%                                             1.645
  95%                                             1.960
  99%                                             2.576

For a two-sided confidence interval, the critical value leaves half of
the remaining probability in each tail.

------------------------------------------------------------------------

## 7. Sampling Distribution and Standard Error

### Sampling distribution

A **sampling distribution** is the probability distribution of a
statistic calculated from repeated samples of the same size drawn from a
population.

For example, imagine repeatedly selecting samples of 10 batteries and
calculating the mean lifetime for each sample. The distribution of those
sample means is the sampling distribution of the sample mean.

### Standard error

The **standard error (SE)** measures the variability of a statistic from
sample to sample. It is not the same as the standard deviation of
individual observations.

When the population standard deviation `σ` is known, the standard error
of the sample mean is:

`SE(x̄) = σ / √n`

Where: - `σ` is the population standard deviation. - `n` is the sample
size.

When `σ` is unknown, the sample standard deviation `s` is often used to
estimate the standard error:

`Estimated SE(x̄) = s / √n`

### Effect of sample size

As sample size increases, the standard error of the mean generally
decreases. Increasing sample size improves precision, although it does
not automatically remove bias caused by poor sampling or measurement.

If sample size becomes four times larger, the standard error becomes
about half as large, because `√(4n) = 2√n`.

------------------------------------------------------------------------

## 8. Central Limit Theorem

The **Central Limit Theorem (CLT)** states that, under suitable
conditions, the sampling distribution of the sample mean becomes
approximately normal as the sample size increases, even if the original
population distribution is not normal.

For independent observations from a population with finite mean `μ` and
finite variance `σ²`, the sample mean has:

-   Mean: `μ`
-   Standard error: `σ / √n`

### Why it matters

The CLT helps us use normal-based methods to estimate means and
calculate probabilities for sample averages, provided the conditions are
appropriate.

### Important caution

A sample size of `n ≥ 30` is sometimes used as a rough classroom
guideline, but it is not a universal rule. Strong skewness, outliers,
dependence, and other features can require a larger sample or a
different method. The sample should also be collected using a suitable
process.

------------------------------------------------------------------------

## 9. Student's t-Distribution

### Definition

Student's **t-distribution** is used for inference about a population
mean when the population standard deviation is unknown and must be
estimated using the sample standard deviation.

It is especially useful for small samples when the population is
approximately normally distributed.

### t-statistic

`t = (x̄ − μ) / (s / √n)`

Where: - `x̄` is the sample mean. - `μ` is the population mean under
consideration. - `s` is the sample standard deviation. - `n` is the
sample size.

### Degrees of freedom

For a one-sample t procedure:

`df = n − 1`

Degrees of freedom affect the shape of the t-distribution and the
critical value used.

### Properties

-   It is symmetric around zero.
-   It has heavier tails than the standard normal distribution for small
    degrees of freedom.
-   Its shape depends on the degrees of freedom.
-   As degrees of freedom increase, it approaches the standard normal
    distribution.

### Choosing between z and t

  -----------------------------------------------------------------------
  Situation                           Typical method
  ----------------------------------- -----------------------------------
  Population standard deviation is    z procedure
  known and normal-based conditions   
  hold                                

  Population standard deviation is    t procedure
  unknown and estimated with sample   
  standard deviation                  

  Small sample with unknown           t procedure, assuming the
  population standard deviation       population is approximately normal
                                      and there are no severe outliers

  Large sample with unknown           t procedures are commonly used;
  population standard deviation       their accuracy depends on the
                                      sampling conditions
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 10. Point Estimation and Estimator Properties

### Point estimate

A **point estimate** is a single value calculated from sample data to
estimate an unknown population parameter.

  Population parameter        Common point estimate
  --------------------------- -----------------------------
  Population mean `μ`         Sample mean `x̄`
  Population variance `σ²`    Sample variance `s²`
  Population proportion `p`   Sample proportion `p̂ = x/n`

A point estimate is convenient, but by itself it does not show how
uncertain the estimate is. A confidence interval provides a range of
plausible values.

### Properties of a good estimator

#### 1. Unbiasedness

An estimator is unbiased if its expected value equals the true
parameter:

`E[θ̂] = θ`

Here, `θ` represents a population parameter and `θ̂` represents its
estimator.

#### 2. Efficiency

Among comparable estimators of the same parameter, an estimator with
smaller variance is generally considered more efficient.

#### 3. Consistency

An estimator is consistent if it approaches the true population
parameter as sample size increases, under the relevant conditions.

#### 4. Sufficiency

A sufficient statistic captures all the sample information relevant to
estimating a parameter within a specified statistical model.

------------------------------------------------------------------------

## 11. Maximum Likelihood Estimation

**Maximum Likelihood Estimation (MLE)** is a method for estimating
unknown model parameters. It selects the parameter value or values that
make the observed data most likely under the chosen probability model.

### General idea

1.  Choose a probability model for the data.
2.  Write the likelihood function for the observed data.
3.  Identify the parameter value that maximizes the likelihood.
4.  Use the maximizing value as the estimate.

For observations `x₁, x₂, ..., xₙ` with parameter `θ`, the likelihood is
often written:

`L(θ) = P(data | θ)`

For independent observations with probability density or mass function
`f`, it is:

`L(θ) = ∏ f(xᵢ | θ)`

It is often easier to maximize the log-likelihood:

`ℓ(θ) = log L(θ)`

The logarithm converts a product into a sum and generally gives the same
maximizing parameter because the logarithm is increasing.

### Why MLE is useful

MLE is widely used in statistical modelling and machine learning to fit
model parameters to observed data. The exact likelihood and estimator
depend on the probability model being used.

------------------------------------------------------------------------

## 12. Confidence Intervals for a Population Mean

### Definition

A **confidence interval (CI)** is a range of values calculated from
sample data to estimate an unknown population parameter, together with a
stated confidence level.

General form:

`Estimate ± (Critical value × Standard error)`

The margin of error is:

`Margin of error = Critical value × Standard error`

### Z-confidence interval

When the population standard deviation `σ` is known and the required
conditions hold:

`x̄ ± z* (σ / √n)`

Where `z*` is the critical z-value for the chosen confidence level.

### t-confidence interval

When the population standard deviation is unknown and is estimated by
`s`:

`x̄ ± t* (s / √n)`

Here, `t*` is the critical value from Student's t-distribution with
`df = n − 1`.

### How to construct a confidence interval

1.  Calculate the sample mean `x̄`.
2.  Find the standard error.
3.  Choose a confidence level, such as 90%, 95%, or 99%.
4.  Select the appropriate critical z- or t-value.
5.  Calculate the margin of error.
6.  Subtract the margin from the estimate for the lower bound.
7.  Add the margin to the estimate for the upper bound.
8.  Interpret the interval in the context of the problem.

### Interpretation of 95% confidence

A 95% confidence level refers to the long-run performance of the method:
if we repeatedly took suitable samples and built intervals in the same
way, about 95% of those intervals would contain the true population
parameter. It does not mean there is a 95% probability that a fixed
parameter lies inside one already-calculated frequentist interval.

### Effect of confidence level

A higher confidence level generally requires a larger critical value, so
the interval becomes wider when the sample and other conditions stay the
same.

### Sample-size formula for estimating a mean

For a target margin of error `E`, when an appropriate value for
population standard deviation `σ` is available:

`n = (z* σ / E)²`

Round the calculated sample size **up** to the next whole number. The
formula is an approximation based on the chosen confidence level and
assumptions.

------------------------------------------------------------------------

## 13. Estimating a Population Proportion

A population proportion is the fraction of a population that has a
particular characteristic, such as the proportion of products that are
defective or the proportion of customers who select a service.

### Sample proportion

`p̂ = x / n`

Where: - `x` is the number of observations with the characteristic,
often called successes. - `n` is the total sample size.

### Estimated standard error

The usual approximate standard error is:

`SE(p̂) = √[p̂(1 − p̂) / n]`

### Approximate confidence interval

`p̂ ± z* √[p̂(1 − p̂) / n]`

This is a normal-approximation interval. Its reliability depends on the
sampling process and whether the approximation is appropriate. For small
samples or proportions near zero or one, another interval method may be
preferable.

### Normal-approximation check

A common guideline is to check that both expected counts are
sufficiently large:

-   `n p̂ ≥ 10`
-   `n (1 − p̂) ≥ 10`

This is a guideline, not a guarantee of accuracy.

### Sample-size formula for a proportion

For a target margin of error `E`:

`n = z*² p̂(1 − p̂) / E²`

If there is no reasonable prior estimate of the population proportion,
use `p̂ = 0.5` for planning. This maximizes `p̂(1 − p̂)` and therefore
gives a conservative sample-size estimate for the normal-approximation
formula. Round the result **up**.

------------------------------------------------------------------------

## 14. Worked Example: Battery Lifetimes

The following example uses the sample shown in the notes.

### Given data

Battery lifetimes, in hours:

`18, 20, 22, 19, 21, 23, 20, 22, 21, 24`

### Step 1: Sample size

There are 10 observations:

`n = 10`

### Step 2: Sample mean

Add the observations:

`18 + 20 + 22 + 19 + 21 + 23 + 20 + 22 + 21 + 24 = 210`

Divide by the sample size:

`x̄ = 210 / 10 = 21.0 hours`

### Step 3: Sample standard deviation

Using the sample standard-deviation formula with denominator `n − 1`
gives approximately:

`s ≈ 1.83 hours`

### Step 4: Standard error

`SE = s / √n`

`SE ≈ 1.83 / √10 ≈ 0.579 hours`

### Step 5: Choose the interval method

The sample is small and the population standard deviation is not known,
so use a t-confidence interval, assuming the population distribution is
sufficiently close to normal and the data meet the method's conditions.

Degrees of freedom:

`df = n − 1 = 9`

For a 95% two-sided interval, the critical value is approximately:

`t* = 2.262`

### Step 6: Calculate the margin of error

`Margin of error = t* × SE`

`Margin of error ≈ 2.262 × 0.579 ≈ 1.31 hours`

### Step 7: Calculate the interval

`95% CI = x̄ ± margin of error`

`95% CI ≈ 21.0 ± 1.31`

Lower limit:

`21.0 − 1.31 ≈ 19.69 hours`

Upper limit:

`21.0 + 1.31 ≈ 22.31 hours`

### Final answer

`95% confidence interval ≈ (19.69, 22.31) hours`

**Interpretation:** Using this method, we are 95% confident that the
population mean battery lifetime lies between approximately 19.69 and
22.31 hours, provided the assumptions of the t-interval are appropriate.

------------------------------------------------------------------------

## 15. Python Practical

### Aim

To calculate sample statistics, estimate probabilities using a normal
model, and calculate a 95% t-confidence interval for the mean battery
lifetime.

### Requirements

-   Python 3
-   NumPy
-   SciPy
-   VS Code or another Python editor (optional)

### Step 1: Install Python packages

Open a terminal and run:

``` bash
python -m pip install numpy scipy
```

If your system uses `python3` instead of `python`, run:

``` bash
python3 -m pip install numpy scipy
```

### Step 2: Create the Python file

Create a file named `inference_case_study.py`.

### Step 3: Enter the program

``` python
import numpy as np
from scipy import stats

# Sample battery lifetimes, in hours
data = np.array(
    [18, 20, 22, 19, 21, 23, 20, 22, 21, 24],
    dtype=float
)

# Basic sample statistics
n = len(data)
mean = data.mean()
sample_sd = data.std(ddof=1)
standard_error = sample_sd / np.sqrt(n)

print(f"Sample size: {n}")
print(f"Sample mean: {mean:.2f} hours")
print(f"Sample standard deviation: {sample_sd:.2f} hours")
print(f"Standard error: {standard_error:.3f} hours")

# Normal model using the sample mean and sample standard deviation
prob_over_23 = 1 - stats.norm.cdf(23, loc=mean, scale=sample_sd)
normal_95 = stats.norm.interval(
    0.95, loc=mean, scale=sample_sd
)

print(
    f"P(lifetime > 23 hours), normal model: {prob_over_23:.3f}"
)
print(
    "Central 95% range under normal model: "
    f"({normal_95[0]:.2f}, {normal_95[1]:.2f}) hours"
)

# 95% t-confidence interval for the population mean
confidence = 0.95
degrees_of_freedom = n - 1
t_critical = stats.t.ppf(
    (1 + confidence) / 2,
    df=degrees_of_freedom
)

margin = t_critical * standard_error
t_interval = (mean - margin, mean + margin)

print(f"Degrees of freedom: {degrees_of_freedom}")
print(f"t critical value: {t_critical:.3f}")
print(
    "95% t-confidence interval for the population mean: "
    f"({t_interval[0]:.2f}, {t_interval[1]:.2f}) hours"
)
```

### Step 4: Run the program

In the terminal, move to the folder containing the file and run:

``` bash
python inference_case_study.py
```

If needed, use:

``` bash
python3 inference_case_study.py
```

### What the program calculates

-   `n`: number of observations.
-   `mean`: sample mean.
-   `sample_sd`: sample standard deviation; `ddof=1` tells NumPy to use
    the sample formula with denominator `n − 1`.
-   `standard_error`: estimated standard error of the sample mean.
-   `stats.norm.cdf(...)`: cumulative probability from a normal
    distribution.
-   `stats.norm.interval(...)`: central interval under the specified
    normal model.
-   `stats.t.ppf(...)`: critical t-value for the selected confidence
    level and degrees of freedom.
-   `t_interval`: lower and upper limits for the confidence interval of
    the population mean.

**Important:** The normal-model probability and central range use the
sample mean and sample standard deviation as model inputs. They are not
the same thing as the t-confidence interval for the population mean. The
t-interval estimates uncertainty about the population mean; the normal
interval describes a central range under the fitted normal model.

### Expected key results

Values may vary slightly in the final decimal due to rounding.

-   Sample size: `10`
-   Sample mean: `21.00 hours`
-   Sample standard deviation: approximately `1.83 hours`
-   Standard error: approximately `0.579 hours`
-   Degrees of freedom: `9`
-   95% t-confidence interval: approximately `(19.69, 22.31) hours`

------------------------------------------------------------------------

## 16. Python Code Explained

### Importing libraries

``` python
import numpy as np
from scipy import stats
```

-   **NumPy** helps store data in an array and calculate numerical
    statistics.
-   **SciPy stats** provides statistical distributions and probability
    functions.

### Storing data

``` python
data = np.array([18, 20, 22, 19, 21, 23, 20, 22, 21, 24], dtype=float)
```

This creates an array of battery lifetimes. `dtype=float` stores the
values as floating-point numbers.

### Counting observations

``` python
n = len(data)
```

`len(data)` returns the number of observations. Here, `n = 10`.

### Calculating the mean

``` python
mean = data.mean()
```

The `.mean()` method calculates the arithmetic average.

### Calculating sample standard deviation

``` python
sample_sd = data.std(ddof=1)
```

`ddof=1` uses `n − 1` in the denominator, which is the common
sample-standard-deviation calculation.

### Calculating standard error

``` python
standard_error = sample_sd / np.sqrt(n)
```

This divides the sample standard deviation by the square root of the
sample size.

### Calculating a normal-model probability

``` python
prob_over_23 = 1 - stats.norm.cdf(23, loc=mean, scale=sample_sd)
```

-   `stats.norm.cdf(23, ...)` calculates the fitted normal model's
    probability of a value less than or equal to 23.
-   `1 - CDF` calculates the probability above 23.
-   `loc=mean` sets the model mean.
-   `scale=sample_sd` sets the model standard deviation.

This calculation assumes the fitted normal model is appropriate.

### Calculating the t critical value

``` python
t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
```

For a two-sided 95% interval, `(1 + 0.95) / 2 = 0.975`. The percentile
function returns the t-value with 97.5% of the distribution to its left.

### Constructing the confidence interval

``` python
margin = t_critical * standard_error
t_interval = (mean - margin, mean + margin)
```

The program calculates the margin of error and subtracts/adds it from
the sample mean to obtain the lower and upper bounds.

------------------------------------------------------------------------

## 17. Applications in Data Science and AI

Inferential statistics helps data scientists make decisions from limited
data and express uncertainty instead of relying only on a single number.

### 1. Model evaluation

A machine-learning model's accuracy, precision, recall, or error rate is
estimated from test data. Statistical intervals can help show how much
uncertainty surrounds these measurements.

### 2. A/B testing

A company may compare two versions of a website or application.
Inferential methods can help estimate whether observed differences in
click-through rates or conversion rates may reflect a real difference
rather than random sample variation.

### 3. Surveys and customer analysis

A survey of a sample of customers can be used to estimate a population
proportion, such as the percentage of customers who prefer a particular
service. The sample size and sampling method affect the quality of the
estimate.

### 4. Quality control

Manufacturers can inspect a sample of products to estimate defect rates
or average product measurements. Sampling helps reduce the need to test
every item.

### 5. Feature standardization

Z-scores can express how far a measurement is from its mean in
standard-deviation units. Standardization is used in some
data-processing and modelling workflows, although not every model
requires it.

### 6. Planning experiments

Confidence levels and target margins of error can be used to plan sample
sizes before collecting data. A larger sample usually gives more precise
estimates, but it also requires more time and resources.

### 7. Monitoring model performance

When a model is evaluated on new data, changes in performance metrics
can be affected by sample variation. Statistical reasoning helps
distinguish observed changes from conclusions that are supported by
enough evidence.

------------------------------------------------------------------------

## 18. Common Mistakes to Avoid

1.  **Confusing population and sample:** The population is the whole
    group; the sample is the selected subset.
2.  **Confusing parameter and statistic:** A parameter describes the
    population; a statistic is calculated from sample data.
3.  **Using `σ` and `s` as if they are always known:** `σ` is the
    population standard deviation; `s` is the sample standard deviation.
4.  **Confusing standard deviation with standard error:** Standard
    deviation describes the spread of observations; standard error
    describes the sampling variability of a statistic.
5.  **Forgetting the square root of sample size:** The standard error of
    the mean is `σ/√n`, not `σ/n`.
6.  **Using the wrong critical distribution:** For a mean with unknown
    population standard deviation, a t procedure is commonly used.
7.  **Forgetting degrees of freedom:** In a one-sample t procedure,
    `df = n − 1`.
8.  **Rounding sample size down:** When a formula gives a required
    sample size, round up to ensure the target is met.
9.  **Overstating a confidence interval:** A 95% confidence procedure
    has 95% long-run coverage under its assumptions; it does not
    guarantee that every interval contains the true parameter.
10. **Assuming a large sample fixes every problem:** A large sample does
    not automatically fix selection bias, poor measurement, or dependent
    observations.
11. **Treating a fitted normal range as a confidence interval for the
    mean:** These describe different quantities and answer different
    questions.
12. **Ignoring assumptions:** Always consider how the data were
    collected and whether the chosen statistical method is appropriate.

------------------------------------------------------------------------

## 19. Quick Revision Sheet

  -----------------------------------------------------------------------
  Concept                             Formula / Key point
  ----------------------------------- -----------------------------------
  Sample mean                         `x̄ = Σxᵢ / n`

  Sample variance                     `s² = Σ(xᵢ − x̄)² / (n − 1)`

  Standard deviation                  `s = √s²`

  Expected value                      `E[X]`

  Variance                            `Var(X) = E[(X − μ)²]`

  Z-score                             `z = (x − μ) / σ`

  Standard error of mean              `SE = σ / √n`

  Estimated standard error            `SE ≈ s / √n`

  t degrees of freedom                `df = n − 1`

  Mean z-interval                     `x̄ ± z*(σ/√n)`

  Mean t-interval                     `x̄ ± t*(s/√n)`

  Sample proportion                   `p̂ = x/n`

  Proportion standard error           `√[p̂(1 − p̂)/n]`

  Approximate proportion CI           `p̂ ± z*√[p̂(1 − p̂)/n]`

  Sample size for a mean              `n = (z*σ/E)²`

  Sample size for a proportion        `n = z*² p̂(1 − p̂)/E²`

  Empirical rule                      68% within 1σ; 95% within 2σ; 99.7%
                                      within 3σ

  MLE                                 Chooses parameter value(s)
                                      maximizing the likelihood of
                                      observed data
  -----------------------------------------------------------------------

### Quick method for a mean confidence interval

1.  Find `n`, `x̄`, and `s`.
2.  Calculate `SE = s/√n`.
3.  Find `df = n − 1`.
4.  Find the t critical value for the chosen confidence level.
5.  Calculate `margin = t* × SE`.
6.  Report `x̄ ± margin` and interpret it in context.

------------------------------------------------------------------------

## 20. Practice Questions

Use these questions for revision.

1.  Define inferential statistics and give one example.
2.  Differentiate between a population and a sample.
3.  Explain the difference between a parameter and a statistic.
4.  Differentiate between discrete and continuous random variables.
5.  What are PMF, PDF, and CDF?
6.  Define expected value, variance, and standard deviation.
7.  State any four properties of the normal distribution.
8.  Explain the 68--95--99.7 empirical rule.
9.  Calculate the z-score when `x = 85`, `μ = 75`, and `σ = 5`.
10. Define sampling distribution and standard error.
11. State the Central Limit Theorem and explain why it is useful.
12. When is the t-distribution used? Define degrees of freedom.
13. What is a point estimate? Explain unbiasedness and consistency.
14. What is Maximum Likelihood Estimation?
15. Write the formula for a confidence interval for a mean when `σ` is
    unknown.
16. Explain the meaning of a 95% confidence interval.
17. A sample contains 40 successes out of 100 observations. Calculate
    the sample proportion.
18. Write the approximate sample-size formula for estimating a
    population mean.
19. Explain why a normal-model interval and a confidence interval for
    the population mean are different.
20. Run the Python practical and identify the sample mean, standard
    deviation, standard error, and confidence interval.

------------------------------------------------------------------------

## 21. Source

Prepared as a detailed GitHub study guide from the uploaded PDF titled
*Inferential Statistics for Data Science & Artificial Intelligence*. The
PDF identifies the material as **Unit III** and credits Dr. S.
Satyanarayana, Sanjivani University.

The worked battery-lifetime example and the Python practical are
included to make the statistical procedures easier to understand and
reproduce. The practical uses Python's NumPy and SciPy libraries.
