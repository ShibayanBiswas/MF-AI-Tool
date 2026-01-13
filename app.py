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

# Initialize chatbot
chatbot = PortfolioChatbot()

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
            return jsonify({
                'response': bot_response.get('response', ''),
                'state': bot_response.get('state', chatbot.state),
                'optimization_result': bot_response.get('optimization_result')
            })
        else:
            return jsonify({
                'response': bot_response,
                'state': chatbot.state
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

