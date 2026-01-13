# Mutual Fund Weighted Portfolio Recommendation Chatbot

An AI-powered chatbot that helps users build diversified mutual fund portfolios by inferring risk tolerance and optimizing fund allocations using the Riskfolio library.

## Features

- 🤖 **Agentic AI System**: Uses OpenAI GPT-4o-mini with autonomous tool usage and decision-making
- 📊 **Risk-Inferred Portfolio Building**: Infers risk tolerance from user responses without direct questions
- 🎯 **Multi-Currency Support**: Supports both INR (India-only) and USD (USA/Japan/India)
- 📈 **Advanced Optimization**: Uses Riskfolio library with different optimization models for different risk profiles
- 💼 **Comprehensive Fund Universe**: Includes equity, debt, balanced, and tax-saver (ELSS) funds
- 🎨 **Beautiful UI**: Modern, responsive chat interface

## Project Structure

```
MF AI Agent/
├── app.py                 # Flask backend server
├── agentic_chatbot.py    # Agentic AI chatbot with tool usage
├── riskfolio_optimizer.py # Portfolio optimization using Riskfolio
├── dummy_data.py          # Dummy mutual fund data
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── README.md             # This file
├── AGENTIC_SYSTEM.md     # Agentic system documentation
├── setup_instructions.md # Setup guide
└── static/
    └── index.html        # Frontend UI
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Conversation Flow

1. **Opening**: The chatbot asks about your investment goals
2. **Currency Selection**: Choose INR or USD
3. **Risk Assessment**: Answer scenario-based questions to infer risk tolerance
4. **Portfolio Structure**: Default fund counts are assigned based on risk profile
5. **Sub-Risk Refinement**: Fine-tune volatility/drawdown tolerance
6. **Asset Allocation**: Specify or use default asset splits
7. **Optimization**: Get optimized portfolio weights

### Risk Buckets

- **LOW**: Conservative portfolios with focus on debt and large-cap equity
- **MEDIUM**: Balanced portfolios with mix of equity and debt
- **HIGH**: Aggressive portfolios with emphasis on mid/small-cap equity

### Optimization Models

Different optimization models are used based on risk profile:

- **HIGH Risk**: `max_return`, `max_alpha`, `max_sharpe`
- **MEDIUM Risk**: `max_sharpe`, `risk_parity`
- **LOW Risk**: `min_volatility`, `risk_parity`

## API Endpoints

- `POST /api/chat` - Send chat message and get response
- `POST /api/optimize` - Run portfolio optimization
- `POST /api/reset` - Reset conversation state
- `GET /api/get-payload` - Get current optimization payload

## Dependencies

- Flask - Web framework
- OpenAI - AI language model
- Riskfolio-Lib - Portfolio optimization
- Pandas/NumPy - Data processing
- Flask-CORS - Cross-origin resource sharing

## Notes

- The application uses dummy mutual fund data for demonstration
- In production, replace dummy data with real mutual fund data
- Ensure OpenAI API key has sufficient credits
- Riskfolio optimization may take a few seconds for complex portfolios

## License

This project is for educational/demonstration purposes.

