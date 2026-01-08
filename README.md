# 🏛️ Strategic Asset Allocation & Mean-Variance Optimization

### Endowment Portfolio Construction (IPS-Driven)

## 📌 Project Overview

This project implements an **institutional-grade strategic asset allocation framework** for a **$1B perpetual endowment**, integrating:

* **Investment Policy Statement (IPS) constraints**
* **Mean-Variance Optimization (MVO)**
* **Risk contribution & diversification analysis**
* **Efficient frontier construction**
* **Multiple policy-compliant model portfolios**

The analysis is framed around a realistic endowment case study (**Evergreen Foundation Endowment**) and produces **implementation-ready portfolios** aligned with governance, liquidity, and risk constraints.

---

## 🎯 Investment Objective

As defined in the IPS:

* **Primary Objective**:
  Preserve and grow **real (inflation-adjusted) purchasing power**
* **Return Target**:
  CPI + 4.0% (net of fees)
* **Spending Policy**:
  4.0% annual distribution (3-year trailing average)
* **Investment Horizon**:
  Perpetual / intergenerational

---

## 🧠 Core Portfolio Construction Concepts

* Strategic Asset Allocation (SAA)
* Mean-Variance Optimization (Markowitz)
* Risk-adjusted utility maximization
* Asset-class-level constraints
* Illiquidity management
* Risk contribution vs capital allocation
* Efficient frontier under policy constraints

---

## 🛠️ Technologies & Tools

* **Python**
* **Pandas & NumPy** – data handling & matrix algebra
* **SciPy (SLSQP)** – constrained optimization
* **Matplotlib & Seaborn** – professional visualization
* **Excel** – capital market assumptions input

---

## 📂 Project Structure

*All files are located in a single folder.*

```
Endowment-Asset-Allocation/
│
├── portfolio_optimization.py      # Full MVO & IPS-constrained engine
├── Strategic_Asset_Allocation.pdf # Investment committee presentation
├── sample_data.xlsx               # Expected returns & covariance inputs
├── *.png                          # Generated charts & analytics
└── README.md
```

---

## 📊 Capital Market Assumptions (Inputs)

Inputs are sourced from long-term, policy-consistent assumptions:

* Expected returns by asset class
* Annualized covariance matrix
* Conservative, governance-aligned estimates

These inputs are intentionally **long-horizon focused**, emphasizing **diversification benefits** rather than short-term forecasts.

---

## ⚙️ Portfolio Construction Methodology

### Optimization Framework

* **Mean-Variance Optimization**
### Global Constraints (IPS-Driven)

* Fully invested (weights sum to 100%)
* No short selling
* Cash ≥ 3%
* Illiquid assets ≤ 25%
* No leverage

---

## 🧩 Asset Class Structure

Portfolios are built across the following categories:

* Global Equities
* Private Equity
* Real Assets
* Hedge Funds
* Credit
* Core Fixed Income
* Cash

Each asset is mapped programmatically to its category to ensure **constraint enforcement at the policy level**, not just at the security level.

---

## 📐 Model Portfolios Constructed

Three policy-compliant portfolios are optimized:

| Portfolio        | Risk Profile      | Objective                  |
| ---------------- | ----------------- | -------------------------- |
| **Conservative** | Lower volatility  | Capital preservation       |
| **Moderate**     | Balanced          | Best IPS alignment         |
| **Growth**       | Higher volatility | Long-term purchasing power |

Each portfolio:

* Respects category min/max ranges
* Meets volatility targets
* Satisfies liquidity & illiquidity constraints

---

## 📈 Efficient Frontier

* Frontier is constructed **only from IPS-feasible portfolios**
* Demonstrates the **true opportunity set** available to the endowment
* Model portfolios lie **on or near the efficient frontier**

Saved output:

* `efficient_frontier.png`

---

## 🔍 Risk Contribution Analysis

Beyond capital allocation, the project decomposes:

* **Marginal risk contribution**
* **Percentage risk contribution by asset class**

Key insight:

> Equity-oriented assets dominate total portfolio risk, while defensive assets meaningfully dampen volatility.

Saved output:

* `risk_contribution_analysis.png`

---

## 📊 Visualizations Generated

The engine automatically produces:

1. Efficient frontier with model portfolios
2. Top-10 asset weights per portfolio
3. Category allocation comparison
4. Risk contribution by category
5. Correlation heatmap (diversification analysis)
6. Risk-return summary (Sharpe-adjusted)
7. Portfolio pie charts

All plots are **exported as high-resolution PNGs** for presentation use.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn openpyxl
```

### 2️⃣ Run Optimization

```bash
python portfolio_optimization.py
```

This will:

* Optimize all three portfolios
* Compute the efficient frontier
* Generate all charts
* Save portfolio weights to CSV

---

## 📚 What I Learned

* Translating an **Investment Policy Statement into code**
* Building **constraint-aware optimization engines**
* Managing illiquidity and liquidity explicitly
* Understanding the difference between **capital allocation and risk contribution**
* Communicating quantitative results to an investment committee

---

## 🔧 Potential Extensions

* Monte Carlo simulation of real purchasing power
* Stress testing & scenario analysis
* Regime-based covariance estimation
* Downside risk optimization (CVaR)
* Dynamic rebalancing framework
