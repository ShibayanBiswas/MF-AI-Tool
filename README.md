# Mutual Fund Weighted Portfolio Recommendation Chatbot

An AI-powered chatbot that helps users build diversified mutual fund portfolios by inferring risk tolerance and optimizing fund allocations using advanced portfolio optimization techniques.

## Features

- 🤖 **Hierarchical Agentic AI System**: Uses OpenAI GPT-4o-mini with specialized agents for each task
- 📊 **Risk-Inferred Portfolio Building**: Infers risk tolerance from user responses without direct questions
- 🎯 **Multi-Currency Support**: Supports both INR (India-only) and USD (USA/Japan/India/Europe/UK/China)
- 📈 **Advanced Optimization**: Uses scipy.optimize with different optimization models for different risk profiles
- 💼 **Comprehensive Fund Universe**: Includes equity, debt, balanced, and tax-saver (ELSS) funds
- 🎨 **Beautiful UI**: Modern, responsive chat interface with animations and touch controls
- 🌍 **Geography-Based Allocation**: Distributes funds across geographies based on user preferences

## Project Structure

```
MF AI Agent/
├── app.py                 # Flask backend server
├── agentic_chatbot.py    # Agentic AI chatbot wrapper (uses coordinator)
├── agents/               # Individual agent modules
│   ├── base.py           # BaseAgent class
│   ├── currency_agent.py # Currency selection agent
│   ├── geography_agent.py # Geography constraints agent
│   ├── risk_agent.py     # Risk assessment agent
│   ├── fund_selection_agent.py # Fund selection agent
│   ├── sub_risk_agent.py # Sub-risk refinement agent
│   └── optimization_agent.py # Portfolio optimization agent
├── teams/                # Team/coordinator modules
│   └── coordinator.py    # Main orchestrator
├── riskfolio_optimizer.py # Portfolio optimization using scipy.optimize
├── dummy_data.py          # Dummy mutual fund data
├── fund_returns_utils.py  # Fund returns calculation utilities
├── database.py           # SQLite database operations
├── run_database.py        # Database initialization script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── README.md             # This file
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
- scipy.optimize - Portfolio optimization
- Pandas/NumPy - Data processing
- Flask-CORS - Cross-origin resource sharing

## Architecture

### Hierarchical Agentic System

The chatbot uses a hierarchical agentic architecture with specialized agents for each task:

#### Agent Responsibilities

1. **CurrencyAgent** - Handles currency selection (INR or USD)
2. **GeographyAgent** - Handles geography constraints for USD investments (USA, India, Japan, Europe, UK, China)
3. **RiskAssessmentAgent** - Infers risk profile from user answers and sets default fund counts
4. **FundSelectionAgent** - Selects actual funds based on criteria, distributing across geographies
5. **SubRiskRefinementAgent** - Refines sub-risk bucket and sets volatility/drawdown targets
6. **OptimizationAgent** - Validates parameters and runs portfolio optimization

#### Coordinator

The `CoordinatorAgent` orchestrates the workflow:
- Routes messages to appropriate agents based on context
- Manages shared state across all agents
- Prevents loops by checking context before asking questions
- Saves conversation history to database
- Loads context from database on initialization

#### Workflow

```
User Message
    ↓
CoordinatorAgent
    ↓
Determine Current Agent (based on context)
    ↓
Route to Appropriate Agent
    ↓
Agent Executes (uses tools, updates context)
    ↓
Return Result with Next Agent
    ↓
Coordinator Updates Context & Saves to DB
    ↓
Return Response to User
```

### Key Features

1. **Prevents Loops**: Each agent checks context before asking questions
2. **Proper Tool Calling**: Each agent has specific tools for its task
3. **Context Awareness**: Shared context across all agents
4. **History Management**: Conversation history maintained and loaded from database
5. **Anti-Hallucination**: Agents use only data from context, not invented data
6. **Geography Distribution**: Funds are distributed across geographies based on user preferences

## Notes

- The application uses dummy mutual fund data for demonstration
- In production, replace dummy data with real mutual fund data
- Ensure OpenAI API key has sufficient credits
- Portfolio optimization uses scipy.optimize and may take a few seconds for complex portfolios
- The system uses a hierarchical agentic architecture with specialized agents for each task

## License

This project is for educational/demonstration purposes.

