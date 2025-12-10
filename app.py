from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions

app = Flask(__name__)
CORS(app)  

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Portfolio Optimizer API',
        'endpoints': ['/optimize', '/rebalance', '/risk-metrics']
    })

@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Optimize portfolio for maximum Sharpe ratio.
    
    Expected input:
    {
        "portfolio": [
            {"ticker": "7203 JP EQUITY", "name": "Toyota", "weight": 0.08, "price": 2500, "returns": 0.12, "volatility": 0.25},
            ...
        ],
        "risk_free_rate": 0.01  # optional, defaults to 0.01 (1%)
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        risk_free_rate = data.get('risk_free_rate', 0.01)
        
        if len(portfolio) < 2:
            return jsonify({'error': 'Need at least 2 assets for optimization'}), 400
        
        tickers = [p['ticker'] for p in portfolio]
        current_weights = {p['ticker']: p['weight'] for p in portfolio}
        expected_rets = {p['ticker']: p.get('returns', 0.10) for p in portfolio}
        volatilities = {p['ticker']: p.get('volatility', 0.20) for p in portfolio}
        
        mu = pd.Series(expected_rets)
        
        # Covariance matrix from volatilities, crr = .3 def
        n = len(tickers)
        corr_matrix = np.full((n, n), 0.3)
        np.fill_diagonal(corr_matrix, 1.0)
        
        vol_array = np.array([volatilities[t] for t in tickers])
        cov_matrix = np.outer(vol_array, vol_array) * corr_matrix
        S = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
        
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        
        try:
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        except Exception as e:
            # excessive max_sharpe exit
            ef = EfficientFrontier(mu, S)
            weights = ef.min_volatility()
        
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
        # est curr port perf
        current_return = sum(current_weights[t] * expected_rets[t] for t in tickers)
        current_vol = np.sqrt(
            sum(
                current_weights[ti] * current_weights[tj] * S.loc[ti, tj]
                for ti in tickers for tj in tickers
            )
        )
        current_sharpe = (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0
        
        result = {
            'status': 'success',
            'current_portfolio': {
                'weights': {t: round(w * 100, 2) for t, w in current_weights.items()},
                'expected_return': round(current_return * 100, 2),
                'volatility': round(current_vol * 100, 2),
                'sharpe_ratio': round(current_sharpe, 3)
            },
            'optimal_portfolio': {
                'weights': {t: round(w * 100, 2) for t, w in cleaned_weights.items()},
                'expected_return': round(performance[0] * 100, 2),
                'volatility': round(performance[1] * 100, 2),
                'sharpe_ratio': round(performance[2], 3)
            },
            'improvement': {
                'return_change': round((performance[0] - current_return) * 100, 2),
                'volatility_change': round((performance[1] - current_vol) * 100, 2),
                'sharpe_change': round(performance[2] - current_sharpe, 3)
            },
            'rebalancing_trades': []
        }
        
        # Config trades
        for ticker in tickers:
            current = current_weights[ticker]
            optimal = cleaned_weights.get(ticker, 0)
            diff = optimal - current

            # Show if >0.5% change
            if abs(diff) > 0.005:  
                name = next((p['name'] for p in portfolio if p['ticker'] == ticker), ticker)
                result['rebalancing_trades'].append({
                    'ticker': ticker,
                    'name': name,
                    'current_weight': round(current * 100, 2),
                    'optimal_weight': round(optimal * 100, 2),
                    'change': round(diff * 100, 2),
                    'action': 'BUY' if diff > 0 else 'SELL'
                })
        
        result['rebalancing_trades'].sort(key=lambda x: abs(x['change']), reverse=True)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/rebalance', methods=['POST'])
def rebalance_portfolio():
    """
    Simple rebalancing to target weights.
    
    Expected input:
    {
        "portfolio": [
            {"ticker": "7203 JP EQUITY", "name": "Toyota", "weight": 0.08, "target_weight": 0.10, "price": 2500},
            ...
        ],
        "portfolio_value": 1000000  # Total portfolio value in JPY
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        portfolio_value = data.get('portfolio_value', 1000000)
        
        trades = []
        
        for p in portfolio:
            current = p.get('weight', 0)
            target = p.get('target_weight', current)
            diff = target - current
            
            if abs(diff) > 0.005:
                trade_value = diff * portfolio_value
                price = p.get('price', 1)
                shares = int(trade_value / price) if price > 0 else 0
                
                trades.append({
                    'ticker': p['ticker'],
                    'name': p.get('name', p['ticker']),
                    'current_weight': round(current * 100, 2),
                    'target_weight': round(target * 100, 2),
                    'change': round(diff * 100, 2),
                    'action': 'BUY' if diff > 0 else 'SELL',
                    'trade_value': round(abs(trade_value), 0),
                    'approx_shares': abs(shares)
                })
        
        trades.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return jsonify({
            'status': 'success',
            'portfolio_value': portfolio_value,
            'trades': trades,
            'total_turnover': round(sum(abs(t['change']) for t in trades) / 2, 2)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/risk-metrics', methods=['POST'])
def risk_metrics():
    """
    Calculate portfolio risk metrics.
    
    Expected input:
    {
        "portfolio": [
            {"ticker": "7203 JP EQUITY", "weight": 0.08, "beta": 1.1, "volatility": 0.25},
            ...
        ]
    }
    """
    try:
        data = request.json
        portfolio = data.get('portfolio', [])
        
        if not portfolio:
            return jsonify({'error': 'No portfolio data provided'}), 400
        
        # Calculate weighted metrics
        portfolio_beta = sum(p['weight'] * p.get('beta', 1.0) for p in portfolio)
        
        # Simplified portfolio volatility (assumes some correlation)
        weights = np.array([p['weight'] for p in portfolio])
        vols = np.array([p.get('volatility', 0.20) for p in portfolio])
        
        # Approximate portfolio vol with 0.3 average correlation
        n = len(portfolio)
        corr = 0.3
        portfolio_var = sum(
            weights[i] * weights[j] * vols[i] * vols[j] * (1.0 if i == j else corr)
            for i in range(n) for j in range(n)
        )
        portfolio_volatility = np.sqrt(portfolio_var)
        
        # Concentration metrics
        herfindahl = sum(w**2 for w in weights)
        effective_n = 1 / herfindahl if herfindahl > 0 else 0
        max_weight = max(weights)
        
        # Sector concentration (if available)
        sectors = {}
        for p in portfolio:
            sector = p.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + p['weight']
        
        return jsonify({
            'status': 'success',
            'risk_metrics': {
                'portfolio_beta': round(portfolio_beta, 3),
                'portfolio_volatility': round(portfolio_volatility * 100, 2),
                'annualized_volatility': round(portfolio_volatility * 100, 2),
                'herfindahl_index': round(herfindahl, 4),
                'effective_positions': round(effective_n, 1),
                'max_position_weight': round(max_weight * 100, 2),
                'sector_allocation': {k: round(v * 100, 2) for k, v in sectors.items()}
            },
            'risk_assessment': {
                'concentration_risk': 'HIGH' if max_weight > 0.15 else 'MEDIUM' if max_weight > 0.10 else 'LOW',
                'diversification': 'POOR' if effective_n < 5 else 'MODERATE' if effective_n < 10 else 'GOOD',
                'market_sensitivity': 'HIGH' if portfolio_beta > 1.2 else 'MODERATE' if portfolio_beta > 0.8 else 'LOW'
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
