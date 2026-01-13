# Setup Instructions

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create .env File**
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   Navigate to: `http://localhost:5000`

## Environment Setup

### Windows (PowerShell)
```powershell
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Linux/Mac
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Troubleshooting

### Import Errors
If you get import errors for `riskfolio`, make sure you've installed all dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### OpenAI API Errors
- Verify your API key is correct in the `.env` file
- Check that you have sufficient API credits
- Ensure the API key has proper permissions

### Port Already in Use
If port 5000 is already in use, you can change it:
```python
# In app.py, change:
app.run(debug=True, host='0.0.0.0', port=5000)
# To:
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Testing the Chatbot

1. Start with: "I want to invest for growth"
2. Choose currency: "INR" or "USD"
3. Answer risk questions naturally
4. Review the optimized portfolio

## Notes

- The chatbot uses dummy data for demonstration
- Replace dummy data with real mutual fund data for production use
- Optimization may take a few seconds for complex portfolios

