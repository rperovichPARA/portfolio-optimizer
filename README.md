# Portfolio Optimizer API

A Flask API for portfolio optimization using PyPortfolioOpt.

## Endpoints

### GET /
Health check endpoint.

### POST /optimize
Optimize portfolio for maximum Sharpe ratio.

**Request:**
```json
{
  "portfolio": [
    {
      "ticker": "7203 JP EQUITY",
      "name": "Toyota",
      "weight": 0.08,
      "price": 2500,
      "returns": 0.12,
      "volatility": 0.25
    }
  ],
  "risk_free_rate": 0.01
}
```

**Response:**
```json
{
  "status": "success",
  "current_portfolio": {
    "weights": {"7203 JP EQUITY": 8.0},
    "expected_return": 12.0,
    "volatility": 25.0,
    "sharpe_ratio": 0.44
  },
  "optimal_portfolio": {
    "weights": {"7203 JP EQUITY": 10.0},
    "expected_return": 14.0,
    "volatility": 22.0,
    "sharpe_ratio": 0.59
  },
  "rebalancing_trades": [
    {
      "ticker": "7203 JP EQUITY",
      "name": "Toyota",
      "current_weight": 8.0,
      "optimal_weight": 10.0,
      "change": 2.0,
      "action": "BUY"
    }
  ]
}
```

### POST /rebalance
Calculate trades needed to reach target weights.

### POST /risk-metrics
Calculate portfolio risk metrics (beta, volatility, concentration).

## Deployment

Deployed on Render. See render.yaml for configuration.
