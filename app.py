from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions
from pypfopt.hierarchical_portfolio import HRPOpt

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Portfolio Optimizer API',
        'version': '2.0',
        'endpoints': ['/optimize', '/rebalance', '/risk-metrics', '/compare'],
        'objectives': ['max_sharpe', 'min_volatility', 'max_return', 'efficient_risk', 'efficient_return', 'risk_parity', 'hrp']
    })


@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Optimize portfolio using various objectives.
    
    Expected input:
    {
        "portfolio": [
            {
                "ticker": "7203 JP EQUITY",
                "name": "Toyota",
                "weight": 0.08,
                "price": 2500,
                "returns": 0.12,
                "volatility": 0.25,
                "sector": "Consumer Discretionary"
            }
        ],
        "objective": "max_sharpe",  // Options: max_sharpe, min_volatility, max_return, efficient_risk, efficient_return, risk_parity, hrp
        "risk_free_rate": 0.01,
        "target_volatility": 0.15,  // For efficient_risk
        "target_return": 0.12,      // For efficient_return
        "min_weight": 0.02,         // Minimum position size (2%)
        "max_weight": 0.15,         // Maximum position size (15%)
        "sector_max": 0.30          // Maximum sector allocation (30%)
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        objective = data.get('objective', 'max_sharpe').lower()
        risk_free_rate = data.get('risk_free_rate', 0.01)
        target_volatility = data.get('target_volatility', 0.15)
        target_return = data.get('target_return', 0.12)
        min_weight = data.get('min_weight', 0.0)
        max_weight = data.get('max_weight', 1.0)
        sector_max = data.get('sector_max', None)
        
        if len(portfolio) < 2:
            return jsonify({'error': 'Need at least 2 assets for optimization'}), 400
        
        # Extract data
        tickers = [p['ticker'] for p in portfolio]
        names = {p['ticker']: p.get('name', p['ticker']) for p in portfolio}
        current_weights = {p['ticker']: p['weight'] for p in portfolio}
        expected_rets = {p['ticker']: p.get('returns', 0.10) for p in portfolio}
        volatilities = {p['ticker']: p.get('volatility', 0.20) for p in portfolio}
        sectors = {p['ticker']: p.get('sector', 'Unknown') for p in portfolio}
        
        # Create expected returns Series
        mu = pd.Series(expected_rets)
        
        # Create covariance matrix from volatilities
        # Using correlation assumptions based on sectors
        n = len(tickers)
        corr_matrix = np.full((n, n), 0.3)  # Base correlation
        
        # Higher correlation within same sector
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i][j] = 1.0
                elif sectors[tickers[i]] == sectors[tickers[j]]:
                    corr_matrix[i][j] = 0.5  # Higher intra-sector correlation
        
        vol_array = np.array([volatilities[t] for t in tickers])
        cov_matrix = np.outer(vol_array, vol_array) * corr_matrix
        S = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
        
        # Build sector mapper for constraints
        sector_mapper = sectors
        unique_sectors = list(set(sectors.values()))
        
        # Initialize result
        cleaned_weights = {}
        performance = (0, 0, 0)
        method_used = objective
        
        # Hierarchical Risk Parity (doesn't use EfficientFrontier)
        if objective == 'hrp':
            # HRP needs returns DataFrame, we'll simulate from volatilities
            np.random.seed(42)
            n_samples = 252
            returns_data = pd.DataFrame(
                np.random.multivariate_normal(
                    [expected_rets[t] / 252 for t in tickers],
                    cov_matrix / 252,
                    n_samples
                ),
                columns=tickers
            )
            
            hrp = HRPOpt(returns_data)
            cleaned_weights = hrp.optimize()
            
            # Calculate performance manually
            w = np.array([cleaned_weights[t] for t in tickers])
            ret = np.sum(w * np.array([expected_rets[t] for t in tickers]))
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            performance = (ret, vol, sharpe)
            
        else:
            # Standard EfficientFrontier optimization
            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
            
            # Add sector constraints if specified
            if sector_max and sector_max < 1.0:
                for sector in unique_sectors:
                    sector_tickers = [t for t in tickers if sectors[t] == sector]
                    if len(sector_tickers) > 1:
                        ef.add_constraint(lambda w, st=sector_tickers: 
                            sum(w[tickers.index(t)] for t in st) <= sector_max)
            
            # Add regularization to avoid extreme weights
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            try:
                if objective == 'max_sharpe':
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                    
                elif objective == 'min_volatility':
                    weights = ef.min_volatility()
                    
                elif objective == 'max_return':
                    # Maximize return (will hit upper bounds)
                    ef_temp = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
                    weights = ef_temp.max_sharpe(risk_free_rate=-1)  # Hack: negative rf maximizes return
                    
                elif objective == 'efficient_risk':
                    weights = ef.efficient_risk(target_volatility=target_volatility)
                    
                elif objective == 'efficient_return':
                    weights = ef.efficient_return(target_return=target_return)
                    
                elif objective == 'risk_parity':
                    # Risk parity: equal risk contribution
                    ef_rp = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
                    weights = ef_rp.min_volatility()  # Start with min vol
                    # Then adjust for risk parity
                    ef_rp2 = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
                    ef_rp2.add_objective(objective_functions.L2_reg, gamma=2)  # Stronger regularization pushes toward equal weight
                    weights = ef_rp2.min_volatility()
                    method_used = 'risk_parity (approx)'
                    
                else:
                    # Default to max sharpe
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                    method_used = 'max_sharpe (default)'
                    
                cleaned_weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            except Exception as opt_error:
                # Fallback to min volatility if optimization fails
                ef_fallback = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
                weights = ef_fallback.min_volatility()
                cleaned_weights = ef_fallback.clean_weights()
                performance = ef_fallback.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                method_used = f'min_volatility (fallback from {objective}: {str(opt_error)[:50]})'
        
        # Calculate current portfolio performance
        current_w = np.array([current_weights[t] for t in tickers])
        current_return = np.sum(current_w * np.array([expected_rets[t] for t in tickers]))
        current_vol = np.sqrt(np.dot(current_w.T, np.dot(cov_matrix, current_w)))
        current_sharpe = (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0
        
        # Calculate sector allocations
        current_sector_alloc = {}
        optimal_sector_alloc = {}
        for sector in unique_sectors:
            current_sector_alloc[sector] = sum(current_weights[t] for t in tickers if sectors[t] == sector)
            optimal_sector_alloc[sector] = sum(cleaned_weights.get(t, 0) for t in tickers if sectors[t] == sector)
        
        # Build result
        result = {
            'status': 'success',
            'method': method_used,
            'parameters': {
                'objective': objective,
                'risk_free_rate': risk_free_rate,
                'min_weight': min_weight,
                'max_weight': max_weight,
                'sector_max': sector_max,
                'target_volatility': target_volatility if objective == 'efficient_risk' else None,
                'target_return': target_return if objective == 'efficient_return' else None
            },
            'current_portfolio': {
                'weights': {t: round(w * 100, 2) for t, w in current_weights.items()},
                'expected_return': round(current_return * 100, 2),
                'volatility': round(current_vol * 100, 2),
                'sharpe_ratio': round(current_sharpe, 3),
                'sector_allocation': {k: round(v * 100, 2) for k, v in current_sector_alloc.items()}
            },
            'optimal_portfolio': {
                'weights': {t: round(w * 100, 2) for t, w in cleaned_weights.items()},
                'expected_return': round(performance[0] * 100, 2),
                'volatility': round(performance[1] * 100, 2),
                'sharpe_ratio': round(performance[2], 3),
                'sector_allocation': {k: round(v * 100, 2) for k, v in optimal_sector_alloc.items()}
            },
            'improvement': {
                'return_change': round((performance[0] - current_return) * 100, 2),
                'volatility_change': round((performance[1] - current_vol) * 100, 2),
                'sharpe_change': round(performance[2] - current_sharpe, 3)
            },
            'rebalancing_trades': []
        }
        
        # Calculate rebalancing trades
        for ticker in tickers:
            current = current_weights[ticker]
            optimal = cleaned_weights.get(ticker, 0)
            diff = optimal - current
            
            if abs(diff) > 0.005:  # Only show if >0.5% change
                result['rebalancing_trades'].append({
                    'ticker': ticker,
                    'name': names[ticker],
                    'sector': sectors[ticker],
                    'current_weight': round(current * 100, 2),
                    'optimal_weight': round(optimal * 100, 2),
                    'change': round(diff * 100, 2),
                    'action': 'BUY' if diff > 0 else 'SELL'
                })
        
        # Sort by absolute change
        result['rebalancing_trades'].sort(key=lambda x: abs(x['change']), reverse=True)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/rebalance', methods=['POST'])
def rebalance_portfolio():
    """
    Calculate trades needed to reach target weights.
    
    Expected input:
    {
        "portfolio": [
            {"ticker": "7203 JP EQUITY", "name": "Toyota", "weight": 0.08, "target_weight": 0.10, "price": 2500}
        ],
        "portfolio_value": 1000000
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        portfolio_value = data.get('portfolio_value', 1000000)
        
        trades = []
        total_buy = 0
        total_sell = 0
        
        for p in portfolio:
            current = p.get('weight', 0)
            target = p.get('target_weight', current)
            diff = target - current
            
            if abs(diff) > 0.005:
                trade_value = diff * portfolio_value
                price = p.get('price', 1)
                shares = int(abs(trade_value) / price) if price > 0 else 0
                
                if diff > 0:
                    total_buy += abs(trade_value)
                else:
                    total_sell += abs(trade_value)
                
                trades.append({
                    'ticker': p['ticker'],
                    'name': p.get('name', p['ticker']),
                    'sector': p.get('sector', 'Unknown'),
                    'current_weight': round(current * 100, 2),
                    'target_weight': round(target * 100, 2),
                    'change': round(diff * 100, 2),
                    'action': 'BUY' if diff > 0 else 'SELL',
                    'trade_value': round(abs(trade_value), 0),
                    'approx_shares': shares,
                    'price': price
                })
        
        trades.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return jsonify({
            'status': 'success',
            'portfolio_value': portfolio_value,
            'trades': trades,
            'summary': {
                'total_buy_value': round(total_buy, 0),
                'total_sell_value': round(total_sell, 0),
                'net_cash_flow': round(total_sell - total_buy, 0),
                'total_turnover_pct': round((total_buy + total_sell) / 2 / portfolio_value * 100, 2),
                'num_trades': len(trades)
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/risk-metrics', methods=['POST'])
def risk_metrics():
    """
    Calculate comprehensive portfolio risk metrics.
    
    Expected input:
    {
        "portfolio": [
            {"ticker": "7203 JP EQUITY", "name": "Toyota", "weight": 0.08, "beta": 1.1, "volatility": 0.25, "sector": "Auto"}
        ]
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        
        if not portfolio:
            return jsonify({'error': 'No portfolio data provided'}), 400
        
        # Extract data
        tickers = [p['ticker'] for p in portfolio]
        weights = np.array([p['weight'] for p in portfolio])
        betas = np.array([p.get('beta', 1.0) for p in portfolio])
        vols = np.array([p.get('volatility', 0.20) for p in portfolio])
        sectors = {p['ticker']: p.get('sector', 'Unknown') for p in portfolio}
        
        # Portfolio beta
        portfolio_beta = np.sum(weights * betas)
        
        # Portfolio volatility (with correlation assumptions)
        n = len(portfolio)
        corr = 0.3  # Base correlation
        corr_matrix = np.full((n, n), corr)
        
        # Higher correlation within same sector
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i][j] = 1.0
                elif sectors[tickers[i]] == sectors[tickers[j]]:
                    corr_matrix[i][j] = 0.5
        
        cov_matrix = np.outer(vols, vols) * corr_matrix
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_var)
        
        # Concentration metrics
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else 0
        max_weight = np.max(weights)
        top_5_weight = np.sum(np.sort(weights)[-5:])
        
        # Sector concentration
        sector_weights = {}
        for p in portfolio:
            sector = p.get('sector', 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + p['weight']
        
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        sector_herfindahl = sum(w ** 2 for w in sector_weights.values())
        effective_sectors = 1 / sector_herfindahl if sector_herfindahl > 0 else 0
        
        # Risk contribution per position
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var if portfolio_var > 0 else weights
        
        position_risk = []
        for i, p in enumerate(portfolio):
            position_risk.append({
                'ticker': p['ticker'],
                'name': p.get('name', p['ticker']),
                'weight': round(weights[i] * 100, 2),
                'risk_contribution': round(risk_contrib[i] * 100, 2),
                'marginal_risk': round(marginal_contrib[i] * 100, 4)
            })
        
        position_risk.sort(key=lambda x: x['risk_contribution'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'portfolio_metrics': {
                'portfolio_beta': round(portfolio_beta, 3),
                'portfolio_volatility': round(portfolio_volatility * 100, 2),
                'annualized_volatility': round(portfolio_volatility * 100, 2),
                'var_95_daily': round(portfolio_volatility * 1.645 / np.sqrt(252) * 100, 2),
                'var_99_daily': round(portfolio_volatility * 2.326 / np.sqrt(252) * 100, 2)
            },
            'concentration_metrics': {
                'herfindahl_index': round(herfindahl, 4),
                'effective_positions': round(effective_n, 1),
                'max_position_weight': round(max_weight * 100, 2),
                'top_5_weight': round(top_5_weight * 100, 2),
                'effective_sectors': round(effective_sectors, 1),
                'max_sector_weight': round(max_sector_weight * 100, 2)
            },
            'sector_allocation': {k: round(v * 100, 2) for k, v in sorted(sector_weights.items(), key=lambda x: -x[1])},
            'position_risk_contribution': position_risk[:10],  # Top 10 risk contributors
            'risk_assessment': {
                'concentration_risk': 'HIGH' if max_weight > 0.15 or max_sector_weight > 0.35 else 'MEDIUM' if max_weight > 0.10 else 'LOW',
                'diversification': 'POOR' if effective_n < 5 else 'MODERATE' if effective_n < 10 else 'GOOD',
                'market_sensitivity': 'HIGH' if portfolio_beta > 1.2 else 'MODERATE' if portfolio_beta > 0.8 else 'LOW',
                'sector_concentration': 'HIGH' if max_sector_weight > 0.35 else 'MODERATE' if max_sector_weight > 0.25 else 'LOW'
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare_strategies():
    """
    Compare multiple optimization strategies side by side.
    
    Expected input:
    {
        "portfolio": [...],
        "strategies": ["max_sharpe", "min_volatility", "risk_parity"],
        "risk_free_rate": 0.01
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        strategies = data.get('strategies', ['max_sharpe', 'min_volatility'])
        
        results = {}
        
        for strategy in strategies:
            # Call optimize for each strategy
            strategy_data = {**data, 'objective': strategy}
            
            # Reuse optimize logic
            with app.test_request_context(json=strategy_data):
                response = optimize_portfolio()
                if isinstance(response, tuple):
                    results[strategy] = {'error': 'Failed'}
                else:
                    results[strategy] = response.get_json()
        
        return jsonify({
            'status': 'success',
            'comparison': results
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
