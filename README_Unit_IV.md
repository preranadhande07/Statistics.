# Unit IV: Hypothesis Testing for Data Science & Artificial Intelligence

**Detailed Notes • Important Formulas • Solved Examples • Python
Practicals • AI Applications**

> **Source:** Prepared from the uploaded PDF *Hypothesis Testing for
> Data Science & Artificial Intelligence --- Unit IV*, authored by
> Dr. S. Satyanarayana, Sanjivani University. This README expands the
> unit into a GitHub-ready study guide. Examples and code are included
> for learning and reproduction.

------------------------------------------------------------------------

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Basic Terms](#2-basic-terms)
3.  [The Five-Step Hypothesis Testing
    Procedure](#3-the-five-step-hypothesis-testing-procedure)
4.  [One-Tailed and Two-Tailed
    Tests](#4-one-tailed-and-two-tailed-tests)
5.  [Type I Error, Type II Error, and
    Power](#5-type-i-error-type-ii-error-and-power)
6.  [Test Statistic, Critical Value, and
    p-Value](#6-test-statistic-critical-value-and-p-value)
7.  [Hypothesis Testing for a Population Mean: Two-Tailed
    t-Test](#7-hypothesis-testing-for-a-population-mean-two-tailed-t-test)
8.  [Hypothesis Testing for a Population Mean: Left-Tailed
    t-Test](#8-hypothesis-testing-for-a-population-mean-left-tailed-t-test)
9.  [Hypothesis Testing for a Population
    Proportion](#9-hypothesis-testing-for-a-population-proportion)
10. [Chi-Square Test](#10-chi-square-test)
11. [Analysis of Variance (ANOVA)](#11-analysis-of-variance-anova)
12. [Effect Size and Statistical
    Power](#12-effect-size-and-statistical-power)
13. [Multiple Comparisons](#13-multiple-comparisons)
14. [Choosing the Correct Statistical
    Test](#14-choosing-the-correct-statistical-test)
15. [Applications in Data Science and
    AI](#15-applications-in-data-science-and-ai)
16. [Complete Python Case Study](#16-complete-python-case-study)
17. [Python Practicals and Code
    Explanation](#17-python-practicals-and-code-explanation)
18. [Mini Projects and Capstone
    Ideas](#18-mini-projects-and-capstone-ideas)
19. [Common Mistakes to Avoid](#19-common-mistakes-to-avoid)
20. [Quick Revision Formula Sheet](#20-quick-revision-formula-sheet)
21. [Practice Questions](#21-practice-questions)
22. [References](#22-references)

------------------------------------------------------------------------

## 1. Introduction

**Hypothesis testing** is a statistical method used to decide whether
sample data provides enough evidence to reject a claim about a
population.

In real situations, we usually cannot measure every person, product, or
event in a population. Instead, we collect a sample and use statistical
methods to assess a claim. Hypothesis testing helps us decide whether an
observed difference is likely to reflect a real effect or could
reasonably be explained by random sampling variation.

### Example

A bottling machine is designed to fill bottles with 500 ml of liquid. A
quality engineer selects 10 bottles and measures their contents.
Hypothesis testing can help determine whether the average fill amount
differs from 500 ml.

### Why do we use hypothesis testing?

-   To test claims about population means and proportions.
-   To compare products, processes, treatments, or machine-learning
    models.
-   To determine whether two categorical variables are associated.
-   To compare the means of three or more groups.
-   To support decisions using data rather than impressions alone.
-   To evaluate A/B experiments and feature-selection methods.

### Descriptive statistics versus hypothesis testing

  -----------------------------------------------------------------------
  Descriptive statistics              Hypothesis testing
  ----------------------------------- -----------------------------------
  Summarizes observed data.           Evaluates a claim about a
                                      population.

  Uses measures such as mean, median, Uses hypotheses, test statistics,
  and standard deviation.             p-values, and decision rules.

  Describes what was observed.        Assesses how compatible the
                                      observed data are with a specified
                                      null hypothesis.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. Basic Terms

### 2.1 Statistical hypothesis

A **statistical hypothesis** is a statement about a population parameter
or probability distribution that can be assessed using data.

### 2.2 Null hypothesis (`H₀`)

The **null hypothesis** is the default claim being tested. It commonly
states that there is no difference, no effect, or that a parameter
equals a specified value.

Example:

`H₀: μ = 500 ml`

This says the population mean bottle fill is 500 ml.

### 2.3 Alternative hypothesis (`H₁` or `Hₐ`)

The **alternative hypothesis** is the claim supported if the sample
provides sufficient evidence against the null hypothesis.

Examples:

-   Two-tailed: `H₁: μ ≠ 500`
-   Left-tailed: `H₁: μ < 500`
-   Right-tailed: `H₁: μ > 500`

### 2.4 Significance level (`α`)

The significance level is the chosen threshold for the probability of a
Type I error. A common choice is:

`α = 0.05`

A 5% significance level does not mean there is a 5% probability that the
null hypothesis is true.

### 2.5 Test statistic

A **test statistic** summarizes the sample evidence in a form that can
be compared with a probability distribution under the null hypothesis.

Examples include `t`, `z`, `χ²`, and `F`.

### 2.6 p-value

The **p-value** is the probability, assuming the null hypothesis and
model assumptions are true, of observing a result at least as extreme as
the result obtained.

-   A small p-value indicates that the observed result would be unusual
    under the null hypothesis.
-   If `p ≤ α`, reject `H₀`.
-   If `p > α`, fail to reject `H₀`.

**Important:** The p-value is not the probability that `H₀` is true.
Failing to reject `H₀` does not prove that it is true.

### 2.7 Critical value and rejection region

The **critical value** is a cutoff from the reference distribution. The
**rejection region** is the range of test-statistic values that leads to
rejecting the null hypothesis at the chosen significance level.

### 2.8 Degrees of freedom (`df`)

Degrees of freedom describe how many independent pieces of information
are available for estimating variability or selecting a reference
distribution. For a one-sample t-test:

`df = n − 1`

------------------------------------------------------------------------

## 3. The Five-Step Hypothesis Testing Procedure

Use this process for most hypothesis tests.

### Step 1: State the hypotheses

Write the null hypothesis `H₀` and alternative hypothesis `H₁`. Choose
the direction based on the research question, not after seeing the
results.

### Step 2: Choose the significance level and test

Select a significance level such as `α = 0.05`. Choose the test that
matches the type of data, study design, and assumptions.

### Step 3: Calculate the test statistic

Use the sample data and the appropriate formula or statistical software
to calculate the test statistic.

### Step 4: Find the p-value or critical value

Compare the p-value with `α`, or compare the statistic with the critical
value.

### Step 5: Make and explain the decision

-   If `p ≤ α`, **reject `H₀`**.
-   If `p > α`, **fail to reject `H₀`**.

Write the conclusion in the context of the original problem. Do not
simply report a number.

### Decision example

Suppose `p = 0.03` and `α = 0.05`.

Since `0.03 < 0.05`, reject `H₀`. The data provide statistically
significant evidence against the null hypothesis, assuming the test
assumptions are reasonable.

If `p = 0.12`, fail to reject `H₀` at the 5% level. This means the
evidence is insufficient to reject the null hypothesis; it does not
prove the null hypothesis.

------------------------------------------------------------------------

## 4. One-Tailed and Two-Tailed Tests

The alternative hypothesis determines the tail or tails where evidence
against the null hypothesis is counted.

### 4.1 Two-tailed test

Used when we want to know whether a parameter is different in either
direction.

`H₀: μ = μ₀`

`H₁: μ ≠ μ₀`

For a two-tailed test, the significance level is split between the two
tails. At `α = 0.05`, each tail contains `0.025`.

For a t-test, reject if:

`|t| > t*` for the two-sided critical value, or equivalently if `p ≤ α`.

### 4.2 Left-tailed test

Used when the research question asks whether a parameter is smaller than
a reference value.

`H₀: μ ≥ μ₀`

`H₁: μ < μ₀`

Reject the null hypothesis if the statistic is sufficiently far into the
left tail, or if the left-tailed p-value is at most `α`.

### 4.3 Right-tailed test

Used when the question asks whether a parameter is larger than a
reference value.

`H₀: μ ≤ μ₀`

`H₁: μ > μ₀`

Reject the null hypothesis if the statistic is sufficiently far into the
right tail, or if the right-tailed p-value is at most `α`.

### Comparison

  Test           Alternative hypothesis   Research question
  -------------- ------------------------ ------------------------
  Two-tailed     `μ ≠ μ₀`                 Is the mean different?
  Left-tailed    `μ < μ₀`                 Is the mean lower?
  Right-tailed   `μ > μ₀`                 Is the mean higher?

**Good practice:** Choose the tail before inspecting the results.
Changing the test direction after seeing the p-value can inflate
false-positive risk and is a form of p-hacking.

------------------------------------------------------------------------

## 5. Type I Error, Type II Error, and Power

### 5.1 Type I error

A **Type I error** occurs when we reject a null hypothesis that is
actually true. Its probability is represented by `α`.

Example: concluding that a machine's average fill differs from the
target when the true average does not differ.

### 5.2 Type II error

A **Type II error** occurs when we fail to reject a null hypothesis that
is false. Its probability is represented by `β`.

Example: failing to detect that a machine's average fill has changed
when it really has changed.

### 5.3 Statistical power

**Power** is the probability that a test correctly rejects a false null
hypothesis.

`Power = 1 − β`

Power generally increases when: - The sample size increases. - The true
effect is larger. - Measurement variability is lower. - The significance
level is higher, although this also raises Type I error risk.

### Error table

  Reality         Reject `H₀`                Fail to reject `H₀`
  --------------- -------------------------- ---------------------
  `H₀` is true    Type I error               Correct decision
  `H₀` is false   Correct decision (power)   Type II error

### Important point

Reducing `α` makes rejecting a true null hypothesis less likely, but,
for a fixed sample size and effect, it can make a real effect harder to
detect. Study design should consider both types of error.

------------------------------------------------------------------------

## 6. Test Statistic, Critical Value, and p-Value

A test statistic measures how far the observed result is from the value
expected under the null hypothesis, relative to the estimated
variability.

### General decision rule using a p-value

-   `p ≤ α`: reject `H₀`.
-   `p > α`: fail to reject `H₀`.

### General decision rule using a critical value

-   If the statistic falls in the rejection region, reject `H₀`.
-   Otherwise, fail to reject `H₀`.

### Example

A test gives `p = 0.009` at `α = 0.05`.

Because `0.009 < 0.05`, reject `H₀`. The result is statistically
significant at the 5% level.

### Statistical significance is not practical importance

A very small effect can produce a small p-value in a very large sample.
Always consider the effect size, confidence interval, study design, and
real-world consequences.

------------------------------------------------------------------------

## 7. Hypothesis Testing for a Population Mean: Two-Tailed t-Test

### Definition

A **one-sample two-tailed t-test** checks whether a population mean
differs from a specified value when the population standard deviation is
unknown.

### Hypotheses

`H₀: μ = μ₀`

`H₁: μ ≠ μ₀`

### Formula

`t = (x̄ − μ₀) / (s / √n)`

Where: - `x̄` = sample mean - `μ₀` = mean specified by the null
hypothesis - `s` = sample standard deviation - `n` = sample size -
`df = n − 1`

### Decision

For a two-tailed test, reject `H₀` if the absolute t-statistic exceeds
the two-sided critical t-value, or if `p ≤ α`.

### Solved example: Bottle-filling machine

A machine is designed to fill bottles with 500 ml. A sample of 10
bottles gives:

`498, 502, 495, 503, 499, 501, 497, 504, 500, 496`

Test whether the population mean differs from 500 ml at `α = 0.05`.

**Step 1: Hypotheses**

`H₀: μ = 500`

`H₁: μ ≠ 500`

**Step 2: Choose the test**

The population standard deviation is unknown, so use a one-sample
t-test.

`n = 10`, `df = 9`

**Step 3: Calculate the sample statistics**

`x̄ = 499.5 ml`

`s ≈ 3.028 ml`

`SE = s / √n ≈ 3.028 / √10 ≈ 0.957 ml`

**Step 4: Calculate t**

`t = (499.5 − 500) / 0.957 ≈ −0.522`

The two-sided p-value is approximately `0.614`.

**Step 5: Decide**

Since `0.614 > 0.05`, fail to reject `H₀`.

**Conclusion:** The sample does not provide sufficient evidence that the
machine's population mean fill differs from 500 ml. This does not prove
that the machine is perfectly calibrated.

### Python function

``` python
from scipy import stats

bottles = [498, 502, 495, 503, 499, 501, 497, 504, 500, 496]
t_stat, p_value = stats.ttest_1samp(bottles, popmean=500)

print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 3))
```

------------------------------------------------------------------------

## 8. Hypothesis Testing for a Population Mean: Left-Tailed t-Test

### Definition

A **left-tailed t-test** checks whether the population mean is below a
specified value.

### Hypotheses

`H₀: μ ≥ μ₀`

`H₁: μ < μ₀`

### Test statistic

`t = (x̄ − μ₀) / (s / √n)`

The degrees of freedom are `n − 1`.

### Solved example: Battery-life claim

A vendor claims that batteries last at least 21 hours. A sample of 12
batteries has the following lifetimes:

`19, 21, 18, 20, 17, 22, 19, 18, 20, 21, 17, 19`

Test whether the population mean lifetime is below 21 hours at
`α = 0.05`.

**Step 1: Hypotheses**

`H₀: μ ≥ 21`

`H₁: μ < 21`

**Step 2: Select the test**

Use a one-sample left-tailed t-test.

`n = 12`, `df = 11`

**Step 3: Sample statistics**

`x̄ = 19.25 hours`

`s ≈ 1.603 hours`

`SE ≈ 1.603 / √12 ≈ 0.463 hours`

**Step 4: Test statistic**

`t = (19.25 − 21) / 0.463 ≈ −3.78`

The left-tailed p-value is approximately `0.0015`.

**Step 5: Decision**

Since `0.0015 < 0.05`, reject `H₀`.

**Conclusion:** The sample provides strong evidence that the population
mean battery life is below 21 hours, assuming the test conditions are
appropriate.

### Python code for a left-tailed test

``` python
from scipy import stats

battery_life = [19, 21, 18, 20, 17, 22, 19, 18, 20, 21, 17, 19]

t_stat, p_two_sided = stats.ttest_1samp(battery_life, popmean=21)

# Convert to a left-tailed p-value only when the observed direction is left.
p_left = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

print("t-statistic:", round(t_stat, 3))
print("left-tailed p-value:", round(p_left, 4))
```

The alternative direction must be chosen before looking at the result.

------------------------------------------------------------------------

## 9. Hypothesis Testing for a Population Proportion

### Definition

A **one-proportion z-test** checks whether a population proportion
differs from a claimed proportion.

Examples include: - Website click-through rate. - Percentage of
defective products. - Customer conversion rate. - Proportion of correct
model predictions.

### Sample proportion

`p̂ = x / n`

Where `x` is the number of successes and `n` is the sample size.

### Hypotheses for a two-tailed test

`H₀: p = p₀`

`H₁: p ≠ p₀`

### Test statistic

Under the null hypothesis, the standard error is calculated using `p₀`:

`SE₀ = √[p₀(1 − p₀) / n]`

`z = (p̂ − p₀) / SE₀`

For a two-tailed test, reject if `|z|` exceeds the two-sided critical
z-value, or if `p-value ≤ α`.

### Normal approximation condition

A common guideline is to check:

-   `n p₀ ≥ 10`
-   `n(1 − p₀) ≥ 10`

These are practical guidelines, not guarantees. For small counts, an
exact binomial test may be more appropriate.

### Solved example: Ad click-through rate

A marketing team expects a 30% click-through rate. In a new campaign, 96
out of 400 users click the ad. Test whether the rate has changed at
`α = 0.05`.

**Step 1: Hypotheses**

`H₀: p = 0.30`

`H₁: p ≠ 0.30`

**Step 2: Sample proportion**

`p̂ = 96 / 400 = 0.24`

**Step 3: Check approximation**

`n p₀ = 400 × 0.30 = 120`

`n(1 − p₀) = 400 × 0.70 = 280`

Both are greater than 10.

**Step 4: Calculate standard error and z**

`SE₀ = √[0.30 × 0.70 / 400] ≈ 0.0229`

`z = (0.24 − 0.30) / 0.0229 ≈ −2.62`

The two-sided p-value is approximately `0.009`.

**Step 5: Decision**

Since `0.009 < 0.05`, reject `H₀`.

**Conclusion:** The campaign click-through rate is statistically
significantly different from 30%; in this sample, it is lower.

### Python code

``` python
from statsmodels.stats.proportion import proportions_ztest

successes = 96
sample_size = 400
benchmark = 0.30

z_stat, p_value = proportions_ztest(
    count=successes,
    nobs=sample_size,
    value=benchmark,
    prop_var=benchmark
)

print("z-statistic:", round(z_stat, 3))
print("p-value:", round(p_value, 4))
```

------------------------------------------------------------------------

## 10. Chi-Square Test

### Definition

A **chi-square (`χ²`) test** is used with categorical data. It compares
observed frequencies with the frequencies expected under the null
hypothesis.

The two common forms are: 1. Chi-square goodness-of-fit test. 2.
Chi-square test of independence.

### 10.1 Chi-square statistic

`χ² = Σ [(O − E)² / E]`

Where: - `O` = observed frequency - `E` = expected frequency

Large differences between observed and expected counts produce a larger
chi-square statistic. The rejection region is in the right tail.

### 10.2 Goodness-of-fit test

This test asks whether the observed frequencies for one categorical
variable fit a specified distribution.

For `k` categories, a common degrees-of-freedom formula is:

`df = k − 1`

This formula may need adjustment if parameters are estimated from the
data.

### 10.3 Test of independence

This test asks whether two categorical variables are associated.

For an `r × c` contingency table:

`Expected count Eᵢⱼ = (row total × column total) / grand total`

`df = (r − 1)(c − 1)`

### Solved example: Gender and product preference

A store records the following product preferences:

  Group         Product A   Product B   Product C     Total
  ----------- ----------- ----------- ----------- ---------
  Male                 30          40          30       100
  Female               20          30          50       100
  **Total**        **50**      **70**      **80**   **200**

Question: Are gender and product preference independent?

**Step 1: Hypotheses**

-   `H₀`: Gender and product preference are independent.
-   `H₁`: Gender and product preference are associated.

**Step 2: Calculate expected counts**

For the male/Product A cell:

`E = (100 × 50) / 200 = 25`

The expected counts for each row are `25, 35, 40`.

  -----------------------------------------------------------------------
  Group                   Product A          Product B          Product C
                         (expected)         (expected)         (expected)
  -------------- ------------------ ------------------ ------------------
  Male                           25                 35                 40

  Female                         25                 35                 40
  -----------------------------------------------------------------------

**Step 3: Calculate the statistic**

`χ² = (30−25)²/25 + (40−35)²/35 + (30−40)²/40`

`+ (20−25)²/25 + (30−35)²/35 + (50−40)²/40`

`χ² ≈ 8.43`

Degrees of freedom:

`df = (2 − 1)(3 − 1) = 2`

**Step 4: Decision**

At `α = 0.05`, the critical value is approximately `5.991`. The p-value
is approximately `0.0148`.

Since `8.43 > 5.991` and `0.0148 < 0.05`, reject `H₀`.

**Conclusion:** The data provide evidence of an association between
gender and product preference. This association does not establish that
gender causes a preference.

### Python code

``` python
import numpy as np
from scipy import stats

observed = np.array([
    [30, 40, 30],  # Male
    [20, 30, 50]   # Female
])

chi2_stat, p_value, dof, expected = stats.chi2_contingency(
    observed,
    correction=False
)

print("Chi-square statistic:", round(chi2_stat, 2))
print("Degrees of freedom:", dof)
print("p-value:", round(p_value, 4))
print("Expected counts:")
print(expected)

if p_value < 0.05:
    print("Decision: Reject H0; evidence of an association.")
else:
    print("Decision: Fail to reject H0.")
```

### Assumptions and cautions

-   Observations should be independent.
-   Data should be counts in mutually exclusive categories.
-   Expected counts should generally be sufficiently large; a common
    guideline is at least 5 in each cell.
-   For sparse 2×2 tables, Fisher's exact test may be more appropriate.
-   Association does not prove causation.

------------------------------------------------------------------------

## 11. Analysis of Variance (ANOVA)

### Definition

**Analysis of Variance (ANOVA)** is a statistical method used to test
whether the means of three or more groups differ.

Running many separate t-tests increases the chance of making at least
one Type I error. One-way ANOVA tests all group means together.

### Hypotheses

For `k` groups:

`H₀: μ₁ = μ₂ = ... = μₖ`

`H₁: At least one group mean differs`

### Main idea

ANOVA compares: - **Between-group variation:** how far group means are
from the grand mean. - **Within-group variation:** how much observations
vary inside each group.

### Formulas

`F = MS_between / MS_within`

`MS_between = SSB / (k − 1)`

`MS_within = SSW / (N − k)`

Where: - `SSB` = sum of squares between groups - `SSW` = sum of squares
within groups - `k` = number of groups - `N` = total number of
observations

Degrees of freedom:

-   Between groups: `df₁ = k − 1`
-   Within groups: `df₂ = N − k`

A large F-statistic indicates that between-group variation is large
relative to within-group variation.

### Solved example: Three teaching methods

Test scores from three teaching methods are:

-   Method A: `85, 90, 88, 87` --- mean `87.5`
-   Method B: `80, 82, 78, 80` --- mean `80.0`
-   Method C: `92, 95, 90, 91` --- mean `92.0`

There are `k = 3` groups and `N = 12` scores. The grand mean is `86.5`.

**Step 1: Hypotheses**

-   `H₀`: All three population means are equal.
-   `H₁`: At least one population mean differs.

**Step 2: Sum of squares**

Between-group sum of squares:

`SSB = Σ nᵢ(x̄ᵢ − x̄)² = 294`

Within-group sum of squares:

`SSW = 35`

**Step 3: Degrees of freedom**

`df_between = k − 1 = 2`

`df_within = N − k = 9`

**Step 4: Mean squares**

`MS_between = 294 / 2 = 147`

`MS_within = 35 / 9 ≈ 3.889`

**Step 5: F-statistic**

`F = 147 / 3.889 ≈ 37.8`

The p-value is approximately `0.00004`.

**Step 6: Decision**

Since `p < 0.05`, reject `H₀`.

**Conclusion:** The data provide strong evidence that the mean scores
are not all equal across the three teaching methods.

### ANOVA table

  Source             Sum of squares (SS)   df   Mean square (MS)      F
  ---------------- --------------------- ---- ------------------ ------
  Between groups                     294    2            147.000   37.8
  Within groups                       35    9              3.889    ---
  Total                              329   11                ---    ---

### What happens after significant ANOVA?

ANOVA tells us that at least one mean differs, but it does not identify
which groups differ. Use an appropriate post-hoc method, such as
**Tukey's HSD**, to compare group pairs while controlling the
family-wise error rate.

### Python code

``` python
from scipy import stats

method_a = [85, 90, 88, 87]
method_b = [80, 82, 78, 80]
method_c = [92, 95, 90, 91]

f_stat, p_value = stats.f_oneway(
    method_a,
    method_b,
    method_c
)

print("F-statistic:", round(f_stat, 2))
print("p-value:", round(p_value, 5))

if p_value < 0.05:
    print("Decision: Reject H0; at least one group mean differs.")
else:
    print("Decision: Fail to reject H0.")
```

### ANOVA assumptions

-   Independent observations.
-   Approximately normal residuals within each group.
-   Similar population variances across groups for standard one-way
    ANOVA.
-   A suitable quantitative outcome.

If variances are very unequal, consider Welch's ANOVA. If distributional
assumptions are badly violated, a method such as Kruskal--Wallis may be
suitable, depending on the research question.

------------------------------------------------------------------------

## 12. Effect Size and Statistical Power

### 12.1 Why a p-value is not enough

A p-value helps assess how incompatible the data are with the null
hypothesis. It does not tell us how large or practically important the
effect is.

With a very large sample, a tiny difference can be statistically
significant but practically unimportant.

Report, where appropriate: - Test statistic. - p-value. - Effect size. -
Confidence interval. - Sample size and study design.

### 12.2 Effect size

An **effect size** measures the magnitude of a difference or
association.

For two means, one common standardized effect size is Cohen's `d`:

`d = (x̄₁ − x̄₂) / s_pooled`

The pooled standard deviation for two independent groups is:

`s_pooled = √{[(n₁−1)s₁² + (n₂−1)s₂²] / (n₁+n₂−2)}`

Very rough guidelines often describe `|d| ≈ 0.2` as small, `0.5` as
medium, and `0.8` as large. These are only general conventions; context
matters.

Example: if two group means are 80 and 86 and the pooled standard
deviation is 8:

`d = (86 − 80) / 8 = 0.75`

The standardized difference is `0.75`.

### 12.3 Power

`Power = 1 − β`

Power is the probability of detecting a specified real effect when that
effect exists. A common planning target is 80% power, though the right
target depends on the consequences of errors and available resources.

### Factors affecting power

-   Larger sample size generally increases power.
-   Larger true effect size generally increases power.
-   Lower variability generally increases power.
-   A larger significance level increases power but also increases Type
    I error risk.

### Practical interpretation

Statistical significance asks whether the data provide evidence against
the null hypothesis. Effect size helps describe the magnitude of the
observed difference. Practical importance asks whether that magnitude
matters in the real situation.

------------------------------------------------------------------------

## 13. Multiple Comparisons

### Definition

The **multiple-comparisons problem** occurs when many hypotheses are
tested. The more tests we perform, the more likely we are to obtain at
least one false positive by chance.

For example, with 20 independent tests each at `α = 0.05`, the expected
number of false positives when all null hypotheses are true is
`20 × 0.05 = 1`.

### 13.1 Bonferroni correction

For `m` tests, use:

`α_adjusted = α / m`

Example: if `α = 0.05` and there are 10 tests:

`α_adjusted = 0.05 / 10 = 0.005`

This method is simple and conservative.

### 13.2 Holm correction

Holm's step-down method adjusts p-values or thresholds sequentially. It
controls the family-wise error rate and is generally less conservative
than applying Bonferroni to every test.

### 13.3 Benjamini--Hochberg procedure

The Benjamini--Hochberg procedure controls the **false discovery rate**,
the expected proportion of false discoveries among rejected hypotheses,
under its applicable conditions. It is often useful when screening many
features.

### Why this matters in AI

Testing thousands of features, repeatedly checking A/B results, or
trying many model settings can produce apparently significant results by
chance. Plan analyses in advance and use appropriate corrections.

------------------------------------------------------------------------

## 14. Choosing the Correct Statistical Test

  -----------------------------------------------------------------------
  Research question /     Common test             Key result
  data                                            
  ----------------------- ----------------------- -----------------------
  Does one population     One-sample t-test       t-statistic and p-value
  mean equal a target,                            
  with unknown population                         
  SD?                                             

  Is one population mean  Left-tailed one-sample  Left-tailed p-value
  lower than a target?    t-test                  

  Does one population     One-proportion z-test   z-statistic and p-value
  proportion differ from                          
  a benchmark?                                    

  Are two categorical     Chi-square test of      χ² and p-value
  variables associated?   independence            

  Does one categorical    Chi-square              χ² and p-value
  variable fit a          goodness-of-fit         
  specified distribution?                         

  Do three or more        One-way ANOVA           F-statistic and p-value
  independent group means                         
  differ?                                         

  Which pairs differ      Post-hoc test such as   Pairwise comparisons
  after a significant     Tukey HSD               
  ANOVA?                                          

  Do two conversion rates Two-proportion z-test   z-statistic and p-value
  differ?                                         

  Are two models          Paired t-test on        t-statistic and p-value
  different on the same   fold-wise differences   
  cross-validation folds?                         

  Is a 2×2 categorical    Fisher's exact test may Exact p-value
  table sparse?           be suitable             

  Are group variances     Welch's t-test /        Robust comparison of
  unequal?                Welch's ANOVA           means
  -----------------------------------------------------------------------

Always check the data type, sampling design, independence,
distributional assumptions, and sample size before selecting a test.

------------------------------------------------------------------------

## 15. Applications in Data Science and AI

### 15.1 A/B testing

Compare two versions of a website, app, or advertisement. A
two-proportion test can compare conversion rates; a t-test can compare a
continuous metric such as average session duration.

### 15.2 Machine-learning model comparison

When two models are evaluated on the same cross-validation folds, a
paired test on fold-wise differences can account for the paired design.
Care is still needed because cross-validation scores may not be fully
independent.

### 15.3 Feature selection

-   Chi-square tests can evaluate associations between categorical
    features and a categorical target.
-   ANOVA F-tests can rank numeric features for classification tasks.
-   When testing many features, apply a suitable multiple-testing
    strategy.

### 15.4 Quality control

A one-sample t-test can assess whether a process mean differs from its
target, such as bottle volume, component dimensions, or manufacturing
time.

### 15.5 Product and customer analytics

Chi-square tests can help assess whether plan type is associated with
churn category or whether user group is associated with product
preference.

### 15.6 Medical and scientific studies

Tests can evaluate differences between treatment groups or proportions
of outcomes, provided the study design and assumptions support the
method.

### 15.7 Model monitoring and drift

Statistical tests can help identify changes in data distributions or
performance metrics. A significant result should be interpreted
alongside effect size, sample size, and practical consequences.

------------------------------------------------------------------------

## 16. Complete Python Case Study

### Aim

To use Python to perform: 1. One-way ANOVA for three teaching methods.
2. A chi-square test of independence for gender and product preference.

### Requirements

-   Python 3
-   NumPy
-   SciPy
-   VS Code, Jupyter Notebook, or Google Colab

### Step 1: Install the required packages

``` bash
python -m pip install numpy scipy statsmodels
```

If your system uses `python3`, use:

``` bash
python3 -m pip install numpy scipy statsmodels
```

### Step 2: Create a file

Create a Python file named `anova_chisquare.py`.

### Step 3: Enter the program

``` python
import numpy as np
from scipy import stats

alpha = 0.05

# -------------------------------------------------
# 1. One-way ANOVA: compare three teaching methods
# -------------------------------------------------
method_a = [85, 90, 88, 87]
method_b = [80, 82, 78, 80]
method_c = [92, 95, 90, 91]

f_stat, p_anova = stats.f_oneway(
    method_a,
    method_b,
    method_c
)

print("ONE-WAY ANOVA")
print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_anova:.5f}")

if p_anova <= alpha:
    print("Decision: Reject H0.")
    print("Conclusion: At least one teaching-method mean differs.")
else:
    print("Decision: Fail to reject H0.")
    print("Conclusion: Insufficient evidence of different group means.")

# -------------------------------------------------
# 2. Chi-square test: gender vs product preference
# Rows: Male, Female
# Columns: Product A, Product B, Product C
# -------------------------------------------------
observed = np.array([
    [30, 40, 30],
    [20, 30, 50]
])

chi2_stat, p_chi, dof, expected = stats.chi2_contingency(
    observed,
    correction=False
)

print("\nCHI-SQUARE TEST OF INDEPENDENCE")
print(f"Chi-square statistic: {chi2_stat:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_chi:.4f}")
print("Expected counts:")
print(expected)

if p_chi <= alpha:
    print("Decision: Reject H0.")
    print("Conclusion: Evidence of an association between gender and preference.")
else:
    print("Decision: Fail to reject H0.")
    print("Conclusion: Insufficient evidence of an association.")
```

### Step 4: Run the program

Open a terminal in the folder containing the file and run:

``` bash
python anova_chisquare.py
```

### Expected key results

**ANOVA** - F-statistic: approximately `37.80` - p-value: approximately
`0.00004` - Decision at `α = 0.05`: reject `H₀` - Conclusion: at least
one teaching-method mean differs.

**Chi-square** - Chi-square statistic: approximately `8.43` - Degrees of
freedom: `2` - p-value: approximately `0.0148` - Expected counts:
`[[25, 35, 40], [25, 35, 40]]` - Decision at `α = 0.05`: reject `H₀` -
Conclusion: gender and product preference are associated in this sample.

Small differences in the last decimal may occur due to software versions
and rounding.

------------------------------------------------------------------------

## 17. Python Practicals and Code Explanation

### Practical 1: Run a one-sample t-test

``` python
from scipy import stats

data = [498, 502, 495, 503, 499, 501, 497, 504, 500, 496]
t_stat, p_value = stats.ttest_1samp(data, popmean=500)

print(t_stat, p_value)
```

`ttest_1samp` tests whether the sample mean differs from the specified
population mean. By default, it performs a two-sided test.

### Practical 2: Run a one-proportion z-test

``` python
from statsmodels.stats.proportion import proportions_ztest

z_stat, p_value = proportions_ztest(
    count=96,
    nobs=400,
    value=0.30,
    prop_var=0.30
)

print(z_stat, p_value)
```

This compares the observed proportion `96/400` with the benchmark
`0.30`.

### Practical 3: Run a chi-square test

``` python
import numpy as np
from scipy import stats

table = np.array([[30, 40, 30], [20, 30, 50]])
chi2, p, dof, expected = stats.chi2_contingency(
    table,
    correction=False
)

print("chi-square:", chi2)
print("p-value:", p)
print("degrees of freedom:", dof)
print("expected counts:\n", expected)
```

The function returns the chi-square statistic, p-value, degrees of
freedom, and expected counts.

### Practical 4: Run ANOVA

``` python
from scipy import stats

A = [85, 90, 88, 87]
B = [80, 82, 78, 80]
C = [92, 95, 90, 91]

f_stat, p_value = stats.f_oneway(A, B, C)
print("F-statistic:", f_stat)
print("p-value:", p_value)
```

`f_oneway` runs a standard one-way ANOVA. Check the test assumptions
before interpreting the result.

### Practical 5: Tukey post-hoc test

``` python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

scores = [85, 90, 88, 87, 80, 82, 78, 80, 92, 95, 90, 91]
groups = ["A"] * 4 + ["B"] * 4 + ["C"] * 4

result = pairwise_tukeyhsd(scores, groups, alpha=0.05)
print(result)
```

Use this after a significant ANOVA to examine which group pairs differ,
while controlling the family-wise error rate.

### Understanding common Python functions

  ------------------------------------------------------------------------
  Function                             Purpose
  ------------------------------------ -----------------------------------
  `stats.ttest_1samp(data, popmean)`   One-sample t-test

  `stats.f_oneway(A, B, C)`            One-way ANOVA

  `stats.chi2_contingency(table)`      Chi-square test of independence

  `proportions_ztest(...)`             Test one or more proportions

  `pairwise_tukeyhsd(...)`             Pairwise post-hoc comparisons

  `stats.t.cdf(...)`                   Cumulative probability from a
                                       t-distribution

  `stats.norm.cdf(...)`                Cumulative probability from a
                                       normal distribution
  ------------------------------------------------------------------------

### Reusable decision helper

``` python
def decide(test_name, statistic, p_value, alpha=0.05):
    decision = "REJECT H0" if p_value <= alpha else "FAIL TO REJECT H0"
    print(f"{test_name}: statistic={statistic:.3f}, "
          f"p={p_value:.4f} -> {decision}")
```

Example:

``` python
decide("Bottle t-test", -0.522, 0.614)
```

This helper formats a result. The user must still select the correct
test, interpret its assumptions, and explain the decision in context.

------------------------------------------------------------------------

## 18. Mini Projects and Capstone Ideas

### Project 1: Hypothesis-Test Toolkit

**Goal:** Build a small tool that takes data and a question, runs an
appropriate test, and reports the statistic, p-value, decision, and
plain-language conclusion.

**Concepts:** t-test, proportion test, chi-square, ANOVA, five-step
procedure.

### Project 2: A/B Test Analyzer

**Goal:** Compare two website or app versions using conversion counts.

**Features:** - Calculate conversion rates and absolute lift. - Run a
two-proportion test. - Report p-value and confidence interval. - Explain
statistical versus practical significance. - Warn against repeatedly
checking p-values without a planned procedure.

### Project 3: Categorical Association Explorer

**Goal:** Examine which categorical variables are associated with an
outcome such as customer churn.

**Features:** - Build contingency tables. - Run chi-square tests. -
Check expected cell counts. - Apply a multiple-testing correction. -
Rank and visualize associations.

### Project 4: Multi-Group Comparison

**Goal:** Compare average outcomes across several groups.

**Features:** - Run ANOVA. - Check assumptions. - Use a post-hoc test
when appropriate. - Consider Welch's ANOVA or Kruskal--Wallis when
assumptions are not suitable. - Display group summaries and box plots.

### Project 5: Machine-Learning Model Comparison

**Goal:** Compare performance scores from several models.

**Features:** - Collect cross-validation scores. - Use paired
comparisons when the design is paired. - Use ANOVA for an appropriate
comparison across multiple groups. - Report effect sizes and
uncertainty. - Discuss limitations of tests on cross-validation folds.

### Project 6: Automated Experimentation Engine

**Goal:** Create a reproducible tool that selects a statistical test
based on the question and data design.

**Features:** - Identify numerical and categorical data. - Select among
mean tests, proportion tests, chi-square, and ANOVA. - Check basic
assumptions. - Apply multiple-testing correction when needed. - Produce
a report with hypotheses, statistic, p-value, effect size, and
conclusion.

------------------------------------------------------------------------

## 19. Common Mistakes to Avoid

1.  **Writing the wrong alternative hypothesis:** Make sure the test
    direction matches the research question.
2.  **Choosing the tail after seeing the result:** This can inflate
    false-positive risk.
3.  **Misinterpreting the p-value:** It is not the probability that the
    null hypothesis is true.
4.  **Saying "accept `H₀`":** Prefer "fail to reject `H₀`."
5.  **Ignoring assumptions:** A test can produce a number even when its
    assumptions are unsuitable.
6.  **Treating statistical significance as practical importance:**
    Report effect size and uncertainty.
7.  **Using many t-tests instead of ANOVA:** Multiple unadjusted tests
    inflate Type I error.
8.  **Stopping after a significant ANOVA:** Use a suitable post-hoc
    method to identify group differences.
9.  **Using chi-square with very small expected counts without
    checking:** Consider an exact method when appropriate.
10. **Assuming association means causation:** A chi-square association
    does not prove a causal relationship.
11. **Ignoring multiple testing:** Feature screening and repeated
    experiments need suitable corrections.
12. **Overstating non-significant results:** A non-significant result
    does not prove there is no effect.
13. **Reporting only a p-value:** Include sample size, effect size,
    confidence interval, and context where possible.
14. **Ignoring practical effect size:** A tiny effect may not justify a
    business or engineering change even if significant.

------------------------------------------------------------------------

## 20. Quick Revision Formula Sheet

  ------------------------------------------------------------------------------------
  Topic                               Formula / Rule
  ----------------------------------- ------------------------------------------------
  Significance level                  Commonly `α = 0.05`

  p-value decision                    Reject `H₀` if `p ≤ α`

  One-sample t-statistic              `t = (x̄ − μ₀) / (s/√n)`

  One-sample t degrees of freedom     `df = n − 1`

  Two-tailed mean hypotheses          `H₀: μ = μ₀`; `H₁: μ ≠ μ₀`

  Left-tailed mean hypotheses         `H₀: μ ≥ μ₀`; `H₁: μ < μ₀`

  Sample proportion                   `p̂ = x/n`

  One-proportion null standard error  `SE₀ = √[p₀(1 − p₀)/n]`

  One-proportion z-statistic          `z = (p̂ − p₀)/SE₀`

  Chi-square statistic                `χ² = Σ[(O − E)²/E]`

  Expected contingency count          `Eᵢⱼ = (row total × column total)/grand total`

  Chi-square df for independence      `df = (r − 1)(c − 1)`

  Chi-square goodness-of-fit df       Usually `k − 1`, with adjustments if parameters
                                      are estimated

  ANOVA F-statistic                   `F = MS_between / MS_within`

  Between-groups mean square          `MS_between = SSB/(k − 1)`

  Within-groups mean square           `MS_within = SSW/(N − k)`

  ANOVA degrees of freedom            Between: `k − 1`; within: `N − k`

  Statistical power                   `1 − β`

  Bonferroni threshold                `α_adjusted = α/m`

  Cohen's d                           `(x̄₁ − x̄₂)/s_pooled`
  ------------------------------------------------------------------------------------

### Five-step method to memorize

1.  Write `H₀` and `H₁`.
2.  Choose `α` and the correct test.
3.  Calculate the test statistic.
4.  Find the p-value or compare with a critical value.
5.  Reject or fail to reject `H₀`, then explain the conclusion in
    context.

------------------------------------------------------------------------

## 21. Practice Questions

1.  Define hypothesis testing.
2.  Differentiate between the null and alternative hypotheses.
3.  Define significance level and p-value.
4.  What is a Type I error? What is a Type II error?
5.  Define statistical power.
6.  Differentiate between one-tailed and two-tailed tests.
7.  State the five steps of hypothesis testing.
8.  Write the formula for a one-sample t-test and its degrees of
    freedom.
9.  A test gives `p = 0.03` at `α = 0.05`. State the decision.
10. Explain why failing to reject `H₀` does not prove that it is true.
11. A machine is intended to fill 500 ml. Explain how to test whether
    its mean differs from 500 ml.
12. What are the hypotheses for testing whether mean battery life is
    below 21 hours?
13. Write the formula for a one-proportion z-test.
14. Why does the one-proportion z-test use `p₀` in its null standard
    error?
15. Define the chi-square test and write its statistic.
16. Differentiate between chi-square goodness-of-fit and chi-square
    independence tests.
17. Write the expected-count formula for a contingency table.
18. What is ANOVA? State its null hypothesis.
19. Write the ANOVA F-statistic and degrees of freedom.
20. Why may a post-hoc test be needed after significant ANOVA?
21. Define effect size and Cohen's d.
22. Explain the multiple-comparisons problem.
23. Calculate the Bonferroni threshold for 10 tests at `α = 0.05`.
24. Name a test for comparing two conversion rates.
25. Which tests can be used for categorical and numerical feature
    selection?
26. Run the Python case study and report the ANOVA and chi-square
    results.
27. Explain the difference between statistical significance and
    practical importance.
28. List three assumptions or conditions that should be checked before
    running a statistical test.

------------------------------------------------------------------------

## 22. References

The uploaded Unit IV PDF lists the following references and further
reading:

1.  Montgomery, D. C. & Runger, G. C., *Applied Statistics and
    Probability for Engineers*, Wiley.
2.  Gupta, S. C. & Kapoor, V. K., *Fundamentals of Mathematical
    Statistics*, Sultan Chand.
3.  Casella, G. & Berger, R. L., *Statistical Inference*, Cengage.
4.  Wasserman, L., *All of Statistics*, Springer.
5.  Bruce, P. & Bruce, A., *Practical Statistics for Data Scientists*,
    O'Reilly.
6.  Documentation for SciPy `stats`, `statsmodels`, and scikit-learn
    `feature_selection`.

**Prepared from:** *Hypothesis Testing for Data Science & Artificial
Intelligence --- Unit IV*, Dr. S. Satyanarayana, Professor, Artificial
Intelligence & Data Science, Sanjivani University.
