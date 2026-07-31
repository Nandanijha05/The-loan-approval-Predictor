# 💳 Loan Approval Predictor: Logistic Regression & Classification Breakdown

> **Personal ML Interview Handbook Series | Topic 1: Binary Classification & Logistic Regression**
> 
> A comprehensive, end-to-end Machine Learning project implementing **Logistic Regression from scratch**, complete with formal mathematical derivations, classification metrics, and critical interview takeaways.

---

## 📌 1. Project Overview

The **Loan Approval Predictor** is a binary classification project designed to determine whether an applicant's loan request should be **Approved (`1`)** or **Rejected (`0`)** based on demographic, financial, and employment parameters (e.g., Applicant Income, Loan Amount, Credit History, Education, and Marital Status).

* **Goal:** Predict loan approval accurately while controlling financial risk using custom probability thresholds.
* **Core Highlight:** Algorithm built **from scratch** using raw mathematical derivations and Gradient Descent, without relying on `scikit-learn`'s high-level abstractions.

---

## 📊 2. Dataset Overview

* **Target Variable:** `Loan_Status`
  * `Y` / `1` $\rightarrow$ **Approved**
  * `N` / `0` $\rightarrow$ **Rejected**
* **Key Numerical Features:** `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
* **Key Categorical Features:** `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Credit_History`

---

## 🧹 3. Data Preprocessing & Cleaning

1. **Missing Value Imputation:**
   * **Numerical Columns (e.g., `LoanAmount`):** Imputed using **Median** to prevent outlier skewness.
   * **Categorical Columns (e.g., `Gender`, `Married`):** Imputed using **Mode** (most frequent class).
2. **Outlier Treatment:** Handled extreme skewness in income and loan figures using logarithmic transformations and capping.

---

## ⚙️ 4. Feature Engineering & Encoding

* **Created Features:**
  * `Income_Category`: Binned `ApplicantIncome` into discrete tiers (Low, Medium, High).
  * `Large_Loan`: Binary indicator flagging loans above critical threshold ($\text{LoanAmount} > 170$).
* **Categorical Encoding:**
  * **One-Hot Encoding:** Applied to nominal variables with no intrinsic order (e.g., `Property_Area`, `Gender`).
  * **Ordinal Encoding:** Applied to ordered features like `Education` and `Income_Category`.

---

## 📏 5. Standardization (Feature Scaling)

To optimize Gradient Descent, features are scaled using **Z-score Standardization**:

$$x' = \frac{x - \mu}{\sigma}$$

### Why Standardization Matters
* Features like `ApplicantIncome` ($\approx 10,000$) and `Credit_History` ($0$ or $1$) operate on vastly different scales.
* Unscaled features create elongated, elliptical cost surface contours, causing Gradient Descent to oscillate wildly and converge slowly.
* Scaling ensures spherical contours, allowing faster, direct convergence to the global minimum.

---

## 🧮 6. Complete Mathematical Derivations

### 6.1 Why Not Linear Regression for Classification?
1. **Unbounded Outputs:** Linear regression outputs values in $(-\infty, +\infty)$, making it impossible to interpret predictions directly as bounded probabilities $P \in [0, 1]$.
2. **Outlier Sensitivity:** A single extreme outlier shifts the linear decision boundary dramatically, severely ruining classification accuracy.

---

### 6.2 Derivation 1: Logit Function to Sigmoid Curve

#### Step A: Define Odds Ratio
$$\text{Odds} = \frac{P}{1 - P}$$

#### Step B: Define Log-Odds (Logit Link Function)
To map probability $P \in (0, 1)$ to continuous space $(-\infty, +\infty)$, we take the natural log of the odds and equate it to a linear equation $z = XW + b$:

$$\ln\left(\frac{P}{1 - P}\right) = z$$

#### Step C: Solve for $P$ (Mathematical Proof)
Exponentiate both sides:
$$\frac{P}{1 - P} = e^z$$

Multiply both sides by $(1 - P)$:
$$P = e^z (1 - P)$$
$$P = e^z - P e^z$$

Rearrange terms with $P$ on the left side:
$$P + P e^z = e^z$$
$$P(1 + e^z) = e^z$$

Solve for $P$:
$$P = \frac{e^z}{1 + e^z}$$

Divide numerator and denominator by $e^z$:
$$P = \frac{1}{1 + e^{-z}}$$

Substitute $z = XW + b$:
$$\sigma(z) = \frac{1}{1 + e^{-(XW + b)}}$$

* **Sigmoid Domain:** $(-\infty, +\infty)$
* **Sigmoid Range:** $(0, 1)$

---

### 6.3 Derivation 2: Binary Cross-Entropy (Log Loss)

We use Maximum Likelihood Estimation (MLE) to derive Binary Cross-Entropy.

For a single sample with true target $y \in \{0, 1\}$ and predicted probability $\hat{y} = \sigma(z)$:
* If $y = 1 \implies P(y|x) = \hat{y}$
* If $y = 0 \implies P(y|x) = 1 - \hat{y}$

Combine both cases into a single Bernoulli distribution equation:
$$P(y|x) = \hat{y}^y (1 - \hat{y})^{(1 - y)}$$

#### Take the Log-Likelihood ($\log L$):
$$\log P(y|x) = \log \left( \hat{y}^y (1 - \hat{y})^{(1 - y)} \right)$$
$$\log P(y|x) = y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})$$

To turn this into a loss function to **minimize** (rather than maximize likelihood), multiply by $-1$:

$$\text{Loss}(\hat{y}, y) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

#### Average Cost Function over $m$ training samples:
$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(\hat{y}^{(i)}\right) + \left(1 - y^{(i)}\right) \log\left(1 - \hat{y}^{(i)}\right) \right]$$

---

### 6.4 Derivation 3: Gradient Descent Partial Derivatives

We apply the **Chain Rule** to compute $\frac{\partial L}{\partial w_j}$.

Let:
1. $z = \sum_{j} w_j x_j + b$
2. $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$
3. $L = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$

#### Step 1: Derivative of Loss $L$ with respect to Prediction $\hat{y}$
$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} = \frac{-\hat{y}y + y\hat{y} - \hat{y}(1 - y)}{\hat{y}(1 - \hat{y})} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

#### Step 2: Derivative of Sigmoid $\hat{y}$ with respect to $z$
$$\hat{y} = (1 + e^{-z})^{-1}$$
$$\frac{\partial \hat{y}}{\partial z} = -1(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \hat{y}(1 - \hat{y})$$

#### Step 3: Combine Steps 1 & 2 ($\frac{\partial L}{\partial z}$)
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \left( \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right) \cdot \left( \hat{y}(1 - \hat{y}) \right) = \hat{y} - y$$

#### Step 4: Derivative of $z$ with respect to Weight $w_j$ and Bias $b$
$$\frac{\partial z}{\partial w_j} = x_j, \quad \frac{\partial z}{\partial b} = 1$$

#### Step 5: Final Partial Derivatives (Chain Rule)
$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_j} = (\hat{y} - y) x_j$$

For $m$ training samples in matrix form:
$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T (\hat{Y} - Y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

#### Parameter Update Rule:
$$W \leftarrow W - \alpha \cdot \frac{\partial J}{\partial W}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$

---

## 📈 7. Evaluation Metrics & Confusion Matrix

### The Confusion Matrix

| | **Predicted Approved (1)** | **Predicted Rejected (0)** |
|---|---|---|
| **Actual Approved (1)** | **True Positive (TP)** | **False Negative (FN)** |
| **Actual Rejected (0)** | **False Positive (FP)** | **True Negative (TN)** |

---

### Metric Definitions & Formulas

| Metric | Formula | What It Measures | Ideal Use Case |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correct predictions | Balanced datasets **only** |
| **Precision** | $\frac{TP}{TP + FP}$ | Quality of positive predictions | When **False Positives** are costly (e.g., Loan Approval, Spam) |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Ability to find all actual positives | When **False Negatives** are critical (e.g., Cancer, Fraud) |
| **Specificity** | $\frac{TN}{TN + FP}$ | Ability to detect actual negatives | Medical testing / ruling out disease |
| **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision & Recall | Imbalanced datasets |

---

## 🎯 8. ROC Curve, AUC, & Threshold Tuning

### ROC Curve (Receiver Operating Characteristic)
* **Y-Axis:** True Positive Rate (Recall) $= \frac{TP}{TP + FN}$
* **X-Axis:** False Positive Rate (FPR) $= \frac{FP}{FP + TN} = 1 - \text{Specificity}$
* **Start Point:** $(0,0)$ when $\text{Threshold} = 1.0$
* **End Point:** $(1,1)$ when $\text{Threshold} = 0.0$

### Area Under Curve (AUC) Interpretation
* **$\text{AUC} = 1.0$**: Perfect model differentiation.
* **$\text{AUC} = 0.8 - 0.9$**: Excellent performance.
* **$\text{AUC} = 0.5$**: Random guessing baseline (no predictive power).

---

### Threshold Tuning Trade-Offs

$$\text{Probability } \hat{y} \ge \text{Threshold} \implies \text{Class } 1 \quad \text{else} \quad \text{Class } 0$$

* **Increasing Threshold (e.g., $0.5 \rightarrow 0.8$):** Model becomes strict.
  * $\text{Precision} \uparrow$, $\text{Recall} \downarrow$, $\text{False Positives} \downarrow$
* **Decreasing Threshold (e.g., $0.5 \rightarrow 0.2$):** Model becomes lenient.
  * $\text{Recall} \uparrow$, $\text{Precision} \downarrow$, $\text{False Negatives} \downarrow$

---

## 🔥 9. Gold-Standard Interview Tricks & Cheat Sheet

1. **`predict()` vs `predict_proba()`:** `predict_proba()` returns raw class probabilities $[P(y=0), P(y=1)]$; `predict()` applies a threshold (default 0.5) to output discrete class labels.
2. **Mathematical Complement Rules:**
   * $\text{Specificity} + \text{FPR} = 1$
   * $\text{Recall} + \text{FNR} = 1$
3. **F1 Score Special Condition:** When $\text{Precision} = \text{Recall}$, $\text{F1 Score} = \text{Precision} = \text{Recall}$.
4. **Logistic Regression Decision Boundary:** A linear hyperplane $XW + b = 0$ in the feature space.
5. **Output Property:** Logistic Regression outputs continuous probabilities, **not** discrete classes directly.
6. **Domain Strategy Matrix:**
   * **Loan Approval / Credit Risk:** Prioritize **Precision** (avoid approving loans destined for default).
   * **Cancer / Medical Diagnosis:** Prioritize **Recall** (avoid letting sick patients go untreated).
   * **Fraud Detection:** Prioritize **Recall** (catch as many fraudulent transactions as possible).
   * **Spam Detection:** Prioritize **Precision** (avoid sending critical real emails to the spam folder).

---

## ❓ 10. Frequently Asked Interview Questions

<details>
<summary><b>1. Why can't we use Mean Squared Error (MSE) for Logistic Regression?</b></summary>
When MSE is combined with the Sigmoid function, the resulting cost function is non-convex. It contains multiple local minima, which means Gradient Descent cannot guarantee reaching the global minimum. Binary Cross-Entropy yields a convex cost surface.
</details>

<details>
<summary><b>2. How does Logistic Regression handle multi-class classification?</b></summary>
Via two approaches:
1. <b>One-vs-Rest (OvR):</b> Trains $K$ separate binary classifiers (one per class against all others).
2. <b>Multinomial (Softmax):</b> Replaces the Sigmoid function with the Softmax function to produce a multi-class probability distribution directly.
</details>

<details>
<summary><b>3. Why is Logistic Regression considered a linear model despite the Sigmoid activation?</b></summary>
Because the decision boundary $XW + b = 0$ is linear with respect to the input features $X$. The Sigmoid transformation acts purely as a non-linear mapping function from log-odds to probability space.
</details>

---


---

