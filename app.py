"""
Flask backend for the Mutual Fund Portfolio Recommendation Chatbot.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agentic_chatbot import AgenticPortfolioAgent as PortfolioChatbot
from riskfolio_optimizer import risk_folio
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize chatbot with session management
import uuid
chatbot = PortfolioChatbot(session_id=str(uuid.uuid4()))

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get bot response
        bot_response = chatbot.chat(user_message)
        
        # Handle agentic response format
        if isinstance(bot_response, dict):
            response_data = {
                'response': bot_response.get('response', ''),
                'state': bot_response.get('state', chatbot.state),
                'optimization_result': bot_response.get('optimization_result'),
                'should_reset': bot_response.get('should_reset', False)  # Include reset flag
            }
            # Include suggested funds if available
            if 'suggested_funds' in chatbot.state:
                response_data['suggested_funds'] = chatbot.state.get('suggested_funds', {})
            return jsonify(response_data)
        else:
            return jsonify({
                'response': bot_response,
                'state': chatbot.state,
                'suggested_funds': chatbot.state.get('suggested_funds', {})
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run portfolio optimization."""
    try:
        payload = request.json
        
        # Call risk_folio function
        result = risk_folio(
            currency=payload.get('currency'),
            primary_risk_bucket=payload.get('primary_risk_bucket'),
            sub_risk_bucket=payload.get('sub_risk_bucket'),
            volatility_target_pct=payload.get('volatility_target_pct'),
            drawdown_target_pct=payload.get('drawdown_target_pct'),
            fund_counts=payload.get('fund_counts', {}),
            asset_split_targets=payload.get('asset_split_targets', {}),
            geography_constraints=payload.get('geography_constraints', {}),
            tax_saver_target_pct=payload.get('tax_saver_target_pct')
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset conversation state."""
    chatbot.reset()
    return jsonify({'status': 'reset'})

@app.route('/api/get-payload', methods=['GET'])
def get_payload():
    """Get current optimization payload."""
    payload = chatbot._build_optimization_payload()
    return jsonify(payload)

@app.route('/api/funds', methods=['GET'])
def get_funds():
    """Get all available funds, optionally filtered by currency."""
    from dummy_data import generate_dummy_funds
    currency = request.args.get('currency', None)
    
    df = generate_dummy_funds()
    
    if currency:
        df = df[df["currency"] == currency]
    
    # Convert to JSON-serializable format
    funds_list = []
    for _, row in df.iterrows():
        fund = {
            "name": row["name"],
            "type": row["type"],
            "currency": row["currency"],
            "geography": row["geography"],
            "market_cap": row["market_cap"],
            "volatility": row["volatility"],
            "returns": row["returns"],
            "category": row["category"],
            "max_drawdown": row["max_drawdown"],
            "sharpe_ratio": row["sharpe_ratio"]
        }
        funds_list.append(fund)
    
    # Group by category
    funds_by_category = {}
    for fund in funds_list:
        category = fund["category"]
        if category not in funds_by_category:
            funds_by_category[category] = []
        funds_by_category[category].append(fund)
    
    # Group by currency -> geography -> category for better segregation
    funds_by_currency_geography = {}
    for fund in funds_list:
        currency = fund["currency"]
        geography = fund.get("geography", "Unknown")
        category = fund["category"]
        
        if currency not in funds_by_currency_geography:
            funds_by_currency_geography[currency] = {}
        
        if geography not in funds_by_currency_geography[currency]:
            funds_by_currency_geography[currency][geography] = {}
        
        if category not in funds_by_currency_geography[currency][geography]:
            funds_by_currency_geography[currency][geography][category] = []
        
        funds_by_currency_geography[currency][geography][category].append(fund)
    
    return jsonify({
        "funds": funds_list,
        "by_category": funds_by_category,
        "by_currency_geography": funds_by_currency_geography,
        "total": len(funds_list)
    })

@app.route('/api/funds/annual-returns', methods=['GET'])
def get_annual_returns():
    """Get 5-year annual returns for funds."""
    from fund_returns_utils import get_fund_annual_returns
    fund_name = request.args.get('fund_name', None)
    currency = request.args.get('currency', None)
    category = request.args.get('category', None)
    
    results = get_fund_annual_returns(fund_name=fund_name, currency=currency, category=category)
    return jsonify({
        "annual_returns": results,
        "total_funds": len(results)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)


