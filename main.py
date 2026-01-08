import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the Excel file
print("📊 Loading data from Excel...")
file_path = "sample data.xlsx"

# Read expected returns from Analytics sheet
analytics_df = pd.read_excel(file_path, sheet_name='Analytics')
expected_returns = analytics_df.set_index('Asset Class')['Expected Return']

# Read covariance matrix
cov_df = pd.read_excel(file_path, sheet_name='Annualized Covariance Matrix', index_col=0)

# Clean up covariance matrix (remove any NaN values)
# Replace NaN values with 0
cov_df = cov_df.fillna(0)
cov_matrix = cov_df.values
assets = cov_df.index.tolist()

# Verify alignment between expected returns and covariance matrix
if len(expected_returns) != len(assets):
    print(f"⚠️ Warning: Mismatch in number of assets. Expected Returns: {len(expected_returns)}, Covariance Matrix: {len(assets)}")
    
# Ensure expected returns are in same order as covariance matrix
expected_returns = expected_returns.reindex(assets).fillna(0)

print(f"✅ Data loaded successfully!")
print(f"   Number of assets: {len(assets)}")
print(f"   Expected returns shape: {expected_returns.shape}")
print(f"   Covariance matrix shape: {cov_matrix.shape}")

# Define asset categories based on your IPS
print("\n🏷️ Defining asset categories...")

# Map assets to categories based on your descriptions
asset_to_category = {}

# Categorize assets properly based on the data
# First, let's see all assets
print("Available assets in data:")
for i, asset in enumerate(assets):
    print(f"  {i+1:2d}. {asset}")

# Define categories based on asset names and your IPS
category_mapping = {}
for asset in assets:
    asset_lower = asset.lower()
    
    # Cash
    if 'cash' in asset_lower or 'short-term' in asset_lower:
        category_mapping[asset] = 'Cash'
    
    # Core Fixed Income (government bonds)
    elif 'treasury' in asset_lower or 'government bond' in asset_lower:
        category_mapping[asset] = 'Core Fixed Income'
    
    # Credit (corporate bonds, ABS, CMBS, RMBS, HY, CLOs, Bank Loans)
    elif ('corp' in asset_lower or 'corporate' in asset_lower or 
          'abs' == asset_lower or 'cmbs' == asset_lower.lower() or
          'rmbs' in asset_lower or 'hy' in asset_lower or
          'clo' in asset_lower or 'bank loan' in asset_lower or
          'ig' in asset_lower or 'credit' in asset_lower or
          'fixed income' in asset_lower and 'us' in asset_lower):
        category_mapping[asset] = 'Credit'
    
    # Global Equities
    elif 'equity' in asset_lower:
        category_mapping[asset] = 'Global Equities'
    
    # Private Equity
    elif 'private equity' in asset_lower:
        category_mapping[asset] = 'Private Equity'
    
    # Real Assets
    elif 'real estate' in asset_lower:
        category_mapping[asset] = 'Real Assets'
    
    # Hedge Funds
    elif 'hedge fund' in asset_lower:
        category_mapping[asset] = 'Hedge Funds'
    
    # Private Debt
    elif 'private debt' in asset_lower:
        category_mapping[asset] = 'Credit'  # Group with credit
    
    # DM ex US Equity and EM Equity
    elif 'dm ex us' in asset_lower or 'em equity' in asset_lower:
        category_mapping[asset] = 'Global Equities'
    
    # Default to 'Other' if not categorized
    else:
        category_mapping[asset] = 'Other'
        print(f"⚠️  Uncategorized asset: {asset}")

# Create category indices
categories = {}
for i, asset in enumerate(assets):
    category = category_mapping.get(asset, 'Other')
    if category not in categories:
        categories[category] = []
    categories[category].append(i)

print(f"\n✅ Categories defined: {list(categories.keys())}")
for category, indices in categories.items():
    print(f"   {category}: {len(indices)} assets")

# Define portfolio constraints based on your IPS
print("\n⚙️ Defining portfolio constraints...")

# BASELINE IPS CONSTRAINTS (apply to ALL portfolios)
baseline_constraints = {
    'categories': categories,
    'sum_to_one': True,
    'no_shorting': True,
    'cash_min': 0.03,  # 3%
    'illiquid_max': 0.25  # Private Equity + Real Assets + Hedge Funds ≤ 25%
}

# PORTFOLIO-SPECIFIC CONSTRAINTS
portfolio_constraints = {
    'Conservative': {
        'category_ranges': {
            'Global Equities': (0.20, 0.35),
            'Private Equity': (0.00, 0.10),
            'Real Assets': (0.05, 0.12),
            'Hedge Funds': (0.00, 0.10),
            'Credit': (0.10, 0.20),
            'Core Fixed Income': (0.20, 0.35),
            'Cash': (0.05, 0.10)
        },
        'illiquid_max': 0.15,  # Stricter than IPS
        'volatility_target': (0.10, 0.12)
    },
    'Moderate': {
        'category_ranges': {
            'Global Equities': (0.35, 0.50),
            'Private Equity': (0.08, 0.15),
            'Real Assets': (0.08, 0.15),
            'Hedge Funds': (0.05, 0.12),
            'Credit': (0.08, 0.15),
            'Core Fixed Income': (0.10, 0.20),
            'Cash': (0.03, 0.07)
        },
        'illiquid_max': 0.20,
        'volatility_target': (0.12, 0.14)
    },
    'Growth': {
        'category_ranges': {
            'Global Equities': (0.45, 0.55),
            'Private Equity': (0.12, 0.20),
            'Real Assets': (0.10, 0.18),
            'Hedge Funds': (0.05, 0.15),
            'Credit': (0.05, 0.12),
            'Core Fixed Income': (0.05, 0.12),
            'Cash': (0.03, 0.05)
        },
        'illiquid_max': 0.25,  # IPS max
        'volatility_target': (0.14, 0.16)
    }
}

# MVO Engine Functions
def portfolio_metrics(weights, expected_returns, cov_matrix):
    """Calculate portfolio metrics"""
    port_return = np.dot(weights, expected_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0
    
    # Calculate risk contribution of each asset
    marginal_risk = np.dot(cov_matrix, weights) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
    risk_contribution = weights * marginal_risk
    percent_risk_contribution = risk_contribution / port_volatility if port_volatility > 0 else np.zeros_like(weights)
    
    return {
        'return': port_return,
        'volatility': port_volatility,
        'sharpe': sharpe_ratio,
        'risk_contribution': risk_contribution,
        'percent_risk_contribution': percent_risk_contribution
    }

def portfolio_objective(weights, expected_returns, cov_matrix, risk_aversion=1.0):
    """Objective function: maximize risk-adjusted return (minimize negative utility)"""
    port_return = np.dot(weights, expected_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    # Negative of utility function (to minimize)
    return -(port_return - 0.5 * risk_aversion * port_variance)

def build_constraints(weights_shape, portfolio_type):
    """Build constraints for optimization"""
    constraints = []
    
    # Get constraints for this portfolio
    constr = portfolio_constraints[portfolio_type]
    baseline = baseline_constraints
    
    # 1. Sum to one constraint
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # 2. Category constraints
    for category, (cat_min, cat_max) in constr['category_ranges'].items():
        if category in baseline['categories']:
            indices = baseline['categories'][category]
            # Minimum constraint
            if cat_min > 0:
                constraints.append({'type': 'ineq', 
                                   'fun': lambda w, idx=indices, min_val=cat_min: np.sum(w[idx]) - min_val})
            # Maximum constraint
            if cat_max < 1:
                constraints.append({'type': 'ineq', 
                                   'fun': lambda w, idx=indices, max_val=cat_max: max_val - np.sum(w[idx])})
    
    # 3. Illiquid assets constraint (Private Equity + Real Assets + Hedge Funds)
    illiquid_indices = []
    for cat in ['Private Equity', 'Real Assets', 'Hedge Funds']:
        if cat in baseline['categories']:
            illiquid_indices.extend(baseline['categories'][cat])
    
    if illiquid_indices:
        constraints.append({'type': 'ineq', 
                          'fun': lambda w, idx=illiquid_indices, max_val=constr['illiquid_max']: max_val - np.sum(w[idx])})
    
    # 4. Cash minimum
    if 'Cash' in baseline['categories']:
        cash_indices = baseline['categories']['Cash']
        constraints.append({'type': 'ineq', 
                          'fun': lambda w, idx=cash_indices: np.sum(w[idx]) - baseline['cash_min']})
    
    return constraints

def optimize_portfolio(portfolio_type, expected_returns, cov_matrix, num_assets, risk_aversion=1.0, max_attempts=3):
    """Optimize portfolio for given type with retry logic"""
    print(f"\n🔧 Optimizing {portfolio_type} portfolio...")
    
    # Build constraints
    constraints = build_constraints(num_assets, portfolio_type)
    
    # Bounds: no shorting (0 to 1)
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Try different initial guesses
    best_result = None
    best_utility = -np.inf
    
    for attempt in range(max_attempts):
        # Different initial guesses
        if attempt == 0:
            # Equal weighted
            initial_weights = np.ones(num_assets) / num_assets
        elif attempt == 1:
            # Tilted toward higher return assets
            returns_norm = expected_returns.values / np.sum(expected_returns.values)
            initial_weights = returns_norm
        else:
            # Random but valid initial guess
            np.random.seed(attempt * 100)
            initial_weights = np.random.random(num_assets)
            initial_weights = initial_weights / np.sum(initial_weights)
        
        # Run optimization
        result = minimize(
            portfolio_objective,
            initial_weights,
            args=(expected_returns.values, cov_matrix, risk_aversion),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success:
            utility = -result.fun  # Convert back to positive utility
            if utility > best_utility:
                best_utility = utility
                best_result = result
                print(f"   Attempt {attempt+1}: Success with utility {utility:.6f}")
    
    if best_result is not None:
        optimal_weights = best_result.x
        metrics = portfolio_metrics(optimal_weights, expected_returns.values, cov_matrix)
        
        print(f"✅ {portfolio_type} portfolio optimized successfully!")
        print(f"   Expected Return: {metrics['return']:.3%}")
        print(f"   Volatility: {metrics['volatility']:.3%}")
        print(f"   Sharpe Ratio: {metrics['sharpe']:.3f}")
        
        # Check if within volatility target
        vol_target = portfolio_constraints[portfolio_type]['volatility_target']
        if vol_target[0] <= metrics['volatility'] <= vol_target[1]:
            print(f"   ✓ Within volatility target: {vol_target[0]:.1%} - {vol_target[1]:.1%}")
        else:
            print(f"   ⚠️ Outside volatility target: {vol_target[0]:.1%} - {vol_target[1]:.1%}")
        
        return optimal_weights, metrics
    else:
        print(f"❌ Optimization failed for {portfolio_type} after {max_attempts} attempts")
        
        # Fallback: Use equal weights within constraints
        print("   Using fallback: equal weights within categories")
        weights = np.ones(num_assets) / num_assets
        
        # Apply simple normalization to meet sum constraint
        weights = weights / np.sum(weights)
        
        metrics = portfolio_metrics(weights, expected_returns.values, cov_matrix)
        return weights, metrics

def compute_efficient_frontier(expected_returns, cov_matrix, num_points=100):
    """Compute efficient frontier"""
    print("\n📈 Computing efficient frontier...")
    
    # Find minimum variance portfolio
    n_assets = len(expected_returns)
    
    # Constraints for minimum variance
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Minimum variance portfolio
    def min_var_objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    
    initial_weights = np.ones(n_assets) / n_assets
    min_var_result = minimize(
        min_var_objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if min_var_result.success:
        min_var_weights = min_var_result.x
        min_var_return = np.dot(min_var_weights, expected_returns.values)
        min_var_vol = np.sqrt(min_var_result.fun)
    else:
        # Fallback to equal weight
        min_var_weights = np.ones(n_assets) / n_assets
        min_var_return = np.dot(min_var_weights, expected_returns.values)
        min_var_vol = np.sqrt(np.dot(min_var_weights.T, np.dot(cov_matrix, min_var_weights)))
    
    # Maximum return portfolio (subject to no shorting)
    max_return_idx = np.argmax(expected_returns.values)
    max_return_weights = np.zeros(n_assets)
    max_return_weights[max_return_idx] = 1
    max_return = expected_returns.values[max_return_idx]
    max_return_vol = np.sqrt(cov_matrix[max_return_idx, max_return_idx])
    
    # Generate frontier
    target_returns = np.linspace(min_var_return, max_return, num_points)
    frontier_volatilities = []
    frontier_weights = []
    
    for target in target_returns:
        # Minimize variance for given return target
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns.values) - target}
        ]
        
        result = minimize(
            min_var_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            frontier_volatilities.append(np.sqrt(result.fun))
            frontier_weights.append(result.x)
    
    print("✅ Efficient frontier computed!")
    return {
        'returns': target_returns,
        'volatilities': frontier_volatilities,
        'weights': frontier_weights,
        'min_var': {'return': min_var_return, 'vol': min_var_vol, 'weights': min_var_weights},
        'max_return': {'return': max_return, 'vol': max_return_vol, 'weights': max_return_weights}
    }

def analyze_portfolio_weights(weights, assets, category_mapping):
    """Analyze portfolio weights by category"""
    df = pd.DataFrame({
        'Asset': assets,
        'Weight': weights,
        'Category': [category_mapping.get(a, 'Other') for a in assets]
    })
    
    # Aggregate by category
    category_weights = df.groupby('Category')['Weight'].sum().sort_values(ascending=False)
    
    return df, category_weights

# Run optimizations for all three portfolios
print("\n" + "="*60)
print("🚀 STARTING PORTFOLIO OPTIMIZATIONS")
print("="*60)

portfolio_results = {}
num_assets = len(assets)

# Adjust risk aversion based on portfolio type
risk_aversion_params = {
    'Conservative': 3.0,  # More risk-averse
    'Moderate': 1.5,      # Balanced
    'Growth': 0.8         # Less risk-averse
}

for portfolio_type in ['Conservative', 'Moderate', 'Growth']:
    weights, metrics = optimize_portfolio(
        portfolio_type, 
        expected_returns, 
        cov_matrix, 
        num_assets,
        risk_aversion_params[portfolio_type]
    )
    
    # Analyze weights
    weight_df, category_weights = analyze_portfolio_weights(weights, assets, category_mapping)
    
    portfolio_results[portfolio_type] = {
        'weights': weights,
        'metrics': metrics,
        'weight_df': weight_df,
        'category_weights': category_weights
    }

# Compute efficient frontier
frontier = compute_efficient_frontier(expected_returns, cov_matrix, num_points=50)

# VISUALIZATIONS
print("\n" + "="*60)
print("🎨 GENERATING VISUALIZATIONS")
print("="*60)

# 1. Efficient Frontier Plot with 3 portfolios
print("\n📊 Creating Visualization 1: Efficient Frontier...")
fig1, ax1 = plt.subplots(figsize=(14, 8))

# Plot efficient frontier
if frontier['volatilities']:
    ax1.plot(frontier['volatilities'], frontier['returns'], 
             'b-', linewidth=2.5, alpha=0.7, label='Efficient Frontier')

# Plot individual assets
asset_volatilities = np.sqrt(np.diag(cov_matrix))
ax1.scatter(asset_volatilities, expected_returns.values, 
           alpha=0.6, s=50, color='gray', label='Individual Assets')

# Plot portfolios
colors = {'Conservative': 'green', 'Moderate': 'blue', 'Growth': 'red'}
markers = {'Conservative': 's', 'Moderate': 'D', 'Growth': 'o'}

for portfolio_type in portfolio_results:
    if portfolio_type in portfolio_results:
        metrics = portfolio_results[portfolio_type]['metrics']
        ax1.scatter(metrics['volatility'], metrics['return'], 
                   color=colors[portfolio_type], s=200, 
                   marker=markers[portfolio_type], edgecolors='black', linewidth=2,
                   label=f'{portfolio_type} Portfolio')

# Plot minimum variance and maximum return portfolios
if 'min_var' in frontier:
    ax1.scatter(frontier['min_var']['vol'], frontier['min_var']['return'], 
               color='purple', s=200, marker='*', edgecolors='black', linewidth=2,
               label='Minimum Variance')
if 'max_return' in frontier:
    ax1.scatter(frontier['max_return']['vol'], frontier['max_return']['return'], 
               color='orange', s=200, marker='*', edgecolors='black', linewidth=2,
               label='Maximum Return')

ax1.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
ax1.set_ylabel('Expected Return', fontsize=12)
ax1.set_title('Efficient Frontier with Model Portfolios', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
if frontier['volatilities']:
    ax1.set_xlim([0, max(frontier['volatilities']) * 1.1])

plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=150, bbox_inches='tight')
print("✅ Saved: efficient_frontier.png")

# 2. Portfolio Weights Comparison (Grouped Bar Chart)
print("\n📊 Creating Visualization 2: Portfolio Weights Comparison...")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

for idx, portfolio_type in enumerate(['Conservative', 'Moderate', 'Growth']):
    if portfolio_type in portfolio_results:
        weight_df = portfolio_results[portfolio_type]['weight_df']
        
        # Sort by weight and get top 10
        sorted_df = weight_df.sort_values('Weight', ascending=False).head(10)
        
        axes2[idx].barh(sorted_df['Asset'], sorted_df['Weight'], color=colors[portfolio_type], alpha=0.7)
        axes2[idx].set_xlabel('Weight (%)', fontsize=10)
        axes2[idx].set_title(f'{portfolio_type} Portfolio\nTop 10 Assets', fontsize=12, fontweight='bold')
        axes2[idx].invert_yaxis()  # Highest weight at top
        axes2[idx].grid(axis='x', alpha=0.3)
        
        # Add weight percentage labels
        for i, (asset, weight) in enumerate(zip(sorted_df['Asset'], sorted_df['Weight'])):
            if weight > 0.001:  # Only show labels for weights > 0.1%
                axes2[idx].text(weight + 0.001, i, f'{weight:.1%}', va='center', fontsize=9)

plt.suptitle('Portfolio Weights Comparison (Top 10 Assets per Portfolio)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('portfolio_weights_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: portfolio_weights_comparison.png")

# 3. Category Allocation Comparison
print("\n📊 Creating Visualization 3: Category Allocation Comparison...")
fig3, ax3 = plt.subplots(figsize=(14, 8))

# Prepare data for grouped bar chart
category_data = {}
all_categories = set()

for portfolio_type in portfolio_results:
    if portfolio_type in portfolio_results:
        category_weights = portfolio_results[portfolio_type]['category_weights']
        category_data[portfolio_type] = category_weights
        all_categories.update(category_weights.index.tolist())

# Sort categories
all_categories = sorted(list(all_categories))

if all_categories:
    # Create grouped bar chart
    x = np.arange(len(all_categories))
    width = 0.25
    
    for i, portfolio_type in enumerate(['Conservative', 'Moderate', 'Growth']):
        if portfolio_type in category_data:
            weights = [category_data[portfolio_type].get(cat, 0) for cat in all_categories]
            ax3.bar(x + (i-1)*width, weights, width, label=portfolio_type, 
                   color=colors[portfolio_type], alpha=0.8)
    
    ax3.set_xlabel('Asset Category', fontsize=12)
    ax3.set_ylabel('Weight (%)', fontsize=12)
    ax3.set_title('Asset Category Allocation by Portfolio', fontsize=16, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_categories, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, portfolio_type in enumerate(['Conservative', 'Moderate', 'Growth']):
        if portfolio_type in category_data:
            weights = [category_data[portfolio_type].get(cat, 0) for cat in all_categories]
            for j, weight in enumerate(weights):
                if weight > 0.01:  # Only show labels for weights > 1%
                    ax3.text(x[j] + (i-1)*width, weight + 0.005, f'{weight:.1%}', 
                            ha='center', va='bottom', fontsize=9, rotation=90)

plt.tight_layout()
plt.savefig('category_allocation_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: category_allocation_comparison.png")

# 4. Risk Contribution Analysis
print("\n📊 Creating Visualization 4: Risk Contribution Analysis...")
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, portfolio_type in enumerate(['Conservative', 'Moderate', 'Growth']):
    if portfolio_type in portfolio_results:
        metrics = portfolio_results[portfolio_type]['metrics']
        weight_df = portfolio_results[portfolio_type]['weight_df']
        
        # Calculate risk contribution by category
        risk_contrib = metrics['percent_risk_contribution'] * 100  # Convert to percentage
        
        # Create DataFrame for risk contribution
        risk_df = pd.DataFrame({
            'Asset': assets,
            'Weight': portfolio_results[portfolio_type]['weights'] * 100,
            'Risk_Contribution': risk_contrib
        })
        risk_df['Category'] = [category_mapping.get(a, 'Other') for a in assets]
        
        # Aggregate risk contribution by category
        category_risk = risk_df.groupby('Category')['Risk_Contribution'].sum().sort_values(ascending=False)
        
        # Plot
        if not category_risk.empty:
            bars = axes4[idx].bar(category_risk.index, category_risk.values, 
                                 color=colors[portfolio_type], alpha=0.7)
            axes4[idx].set_title(f'{portfolio_type} Portfolio\nRisk Contribution by Category', 
                               fontsize=12, fontweight='bold')
            axes4[idx].set_xlabel('')
            axes4[idx].tick_params(axis='x', rotation=45)
            axes4[idx].grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            for bar, value in zip(bars, category_risk.values):
                if value > 0.1:  # Only show labels for values > 0.1%
                    axes4[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

axes4[0].set_ylabel('Risk Contribution (%)', fontsize=12)
plt.suptitle('Risk Contribution Analysis by Category', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('risk_contribution_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: risk_contribution_analysis.png")

# 5. Correlation Heatmap
print("\n📊 Creating Visualization 5: Correlation Heatmap...")
# Calculate correlation matrix from covariance matrix
std_devs = np.sqrt(np.diag(cov_matrix))
# Avoid division by zero
std_devs[std_devs == 0] = 1e-10
corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

# Select top assets for clearer visualization
top_n = min(15, len(assets))  # Use 15 or fewer if we have fewer assets
mean_returns = expected_returns.values
top_indices = np.argsort(mean_returns)[-top_n:]  # Top by expected return
top_assets = [assets[i] for i in top_indices]
top_corr = corr_matrix[np.ix_(top_indices, top_indices)]

fig5, ax5 = plt.subplots(figsize=(14, 12))
im = ax5.imshow(top_corr, cmap='RdYlBu_r', vmin=-1, vmax=1)

# Add colorbar
cbar = ax5.figure.colorbar(im, ax=ax5, shrink=0.8)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")

# Set ticks and labels
ax5.set_xticks(np.arange(len(top_assets)))
ax5.set_yticks(np.arange(len(top_assets)))
ax5.set_xticklabels(top_assets, rotation=90, fontsize=9)
ax5.set_yticklabels(top_assets, fontsize=9)

# Add text annotations
for i in range(len(top_assets)):
    for j in range(len(top_assets)):
        if abs(top_corr[i, j]) > 0.3:  # Only show significant correlations
            text = ax5.text(j, i, f'{top_corr[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if abs(top_corr[i, j]) > 0.5 else "black",
                           fontsize=7)

ax5.set_title('Correlation Matrix (Top Assets by Expected Return)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ Saved: correlation_heatmap.png")

# 6. Return vs Risk Summary
print("\n📊 Creating Visualization 6: Return vs Risk Summary...")
fig6, ax6 = plt.subplots(figsize=(10, 8))

# Prepare data
portfolio_types = []
returns = []
volatilities = []
sharpe_ratios = []

for portfolio_type in ['Conservative', 'Moderate', 'Growth']:
    if portfolio_type in portfolio_results:
        metrics = portfolio_results[portfolio_type]['metrics']
        portfolio_types.append(portfolio_type)
        returns.append(metrics['return'] * 100)  # Convert to percentage
        volatilities.append(metrics['volatility'] * 100)
        sharpe_ratios.append(metrics['sharpe'])

if portfolio_types:  # Only create plot if we have data
    # Create scatter plot
    scatter = ax6.scatter(volatilities, returns, 
                         c=sharpe_ratios, cmap='viridis', 
                         s=300, alpha=0.8, edgecolors='black')

    # Add labels for each portfolio
    for i, portfolio_type in enumerate(portfolio_types):
        ax6.annotate(portfolio_type, 
                    (volatilities[i], returns[i]),
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add colorbar for Sharpe ratio
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=15)

    ax6.set_xlabel('Volatility (%)', fontsize=12)
    ax6.set_ylabel('Expected Return (%)', fontsize=12)
    ax6.set_title('Portfolio Risk-Return Summary', fontsize=16, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Add annotation with key metrics
    textstr = '\n'.join([
        f'{portfolio_types[i]}: {returns[i]:.2f}% return, {volatilities[i]:.2f}% vol, Sharpe: {sharpe_ratios[i]:.2f}'
        for i in range(len(portfolio_types))
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('risk_return_summary.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: risk_return_summary.png")
else:
    print("⚠️  Skipping Visualization 6: No portfolio results available")

# 7. Pie Charts for each portfolio
print("\n📊 Creating Visualization 7: Portfolio Pie Charts...")
fig7, axes7 = plt.subplots(1, 3, figsize=(18, 6))

for idx, portfolio_type in enumerate(['Conservative', 'Moderate', 'Growth']):
    if portfolio_type in portfolio_results:
        category_weights = portfolio_results[portfolio_type]['category_weights']
        
        # Filter out very small categories for cleaner pie chart
        significant_categories = category_weights[category_weights > 0.01]
        
        if not significant_categories.empty:
            # Create pie chart
            wedges, texts, autotexts = axes7[idx].pie(
                significant_categories.values,
                labels=significant_categories.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(significant_categories))),
                wedgeprops=dict(width=0.3, edgecolor='w')
            )
            
            # Improve label appearance
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            axes7[idx].set_title(f'{portfolio_type} Portfolio\nAsset Allocation', 
                               fontsize=12, fontweight='bold')
        else:
            axes7[idx].text(0.5, 0.5, 'No significant\ncategories found', 
                          ha='center', va='center', fontsize=12)
            axes7[idx].set_title(f'{portfolio_type} Portfolio', fontsize=12, fontweight='bold')

plt.suptitle('Portfolio Asset Allocation (Pie Charts)', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('portfolio_pie_charts.png', dpi=150, bbox_inches='tight')
print("✅ Saved: portfolio_pie_charts.png")

# Print summary table
print("\n" + "="*60)
print("📋 PORTFOLIO SUMMARY")
print("="*60)

summary_data = []
for portfolio_type in ['Conservative', 'Moderate', 'Growth']:
    if portfolio_type in portfolio_results:
        metrics = portfolio_results[portfolio_type]['metrics']
        category_weights = portfolio_results[portfolio_type]['category_weights']
        
        # Get key allocations
        equity_allocation = category_weights.get('Global Equities', 0)
        fixed_income_allocation = category_weights.get('Core Fixed Income', 0) + category_weights.get('Credit', 0)
        cash_allocation = category_weights.get('Cash', 0)
        illiquid_allocation = sum(category_weights.get(cat, 0) for cat in ['Private Equity', 'Real Assets', 'Hedge Funds'])
        
        summary_data.append({
            'Portfolio': portfolio_type,
            'Expected Return': f"{metrics['return']:.3%}",
            'Volatility': f"{metrics['volatility']:.3%}",
            'Sharpe Ratio': f"{metrics['sharpe']:.3f}",
            'Equity Allocation': f"{equity_allocation:.1%}",
            'Fixed Income': f"{fixed_income_allocation:.1%}",
            'Cash': f"{cash_allocation:.1%}",
            'Illiquid Assets': f"{illiquid_allocation:.1%}"
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
else:
    print("No portfolio results to summarize")

# Check constraints
print("\n" + "="*60)
print("🔍 CONSTRAINT COMPLIANCE CHECK")
print("="*60)

for portfolio_type in ['Conservative', 'Moderate', 'Growth']:
    if portfolio_type in portfolio_results:
        print(f"\n{portfolio_type} Portfolio:")
        
        # Get constraints for this portfolio
        constr = portfolio_constraints[portfolio_type]
        category_weights = portfolio_results[portfolio_type]['category_weights']
        
        # Check category constraints
        for category, (cat_min, cat_max) in constr['category_ranges'].items():
            actual = category_weights.get(category, 0)
            if cat_min <= actual <= cat_max:
                print(f"  ✓ {category}: {actual:.1%} (target: {cat_min:.0%}-{cat_max:.0%})")
            else:
                print(f"  ⚠️ {category}: {actual:.1%} (target: {cat_min:.0%}-{cat_max:.0%})")
        
        # Check illiquid constraint
        illiquid_actual = sum(category_weights.get(cat, 0) for cat in ['Private Equity', 'Real Assets', 'Hedge Funds'])
        illiquid_max = constr['illiquid_max']
        if illiquid_actual <= illiquid_max:
            print(f"  ✓ Illiquid Assets: {illiquid_actual:.1%} (max: {illiquid_max:.0%})")
        else:
            print(f"  ⚠️ Illiquid Assets: {illiquid_actual:.1%} (max: {illiquid_max:.0%})")
        
        # Check volatility target
        vol_actual = portfolio_results[portfolio_type]['metrics']['volatility']
        vol_min, vol_max = constr['volatility_target']
        if vol_min <= vol_actual <= vol_max:
            print(f"  ✓ Volatility: {vol_actual:.1%} (target: {vol_min:.1%}-{vol_max:.1%})")
        else:
            print(f"  ⚠️ Volatility: {vol_actual:.1%} (target: {vol_min:.1%}-{vol_max:.1%})")

print("\n" + "="*60)
print("✅ OPTIMIZATION COMPLETE!")
print("="*60)
print("\n📁 Generated Files:")
print("  1. efficient_frontier.png - Efficient Frontier with all portfolios")
print("  2. portfolio_weights_comparison.png - Top 10 assets per portfolio")
print("  3. category_allocation_comparison.png - Category allocation comparison")
print("  4. risk_contribution_analysis.png - Risk contribution by category")
print("  5. correlation_heatmap.png - Correlation matrix of top assets")
print("  6. risk_return_summary.png - Risk-return summary with Sharpe ratios")
print("  7. portfolio_pie_charts.png - Pie charts for each portfolio")

print("\n🎯 Key Takeaways:")
print("  • Conservative: Lower return, lower volatility, higher fixed income")
print("  • Moderate: Balanced approach, moderate risk-return profile")
print("  • Growth: Higher return potential, higher equity allocation")

# Save detailed results to CSV
print("\n💾 Saving detailed results to CSV files...")
for portfolio_type in portfolio_results:
    weight_df = portfolio_results[portfolio_type]['weight_df']
    weight_df.to_csv(f'{portfolio_type.lower()}_weights.csv', index=False)
    print(f"  • {portfolio_type.lower()}_weights.csv")

print("\n✨ Analysis complete! All visualizations saved to disk.")