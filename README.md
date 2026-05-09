# FinSight — Multi-Agent Financial News Intelligence Platform

<div align="center">

**Evidence-grounded signal generation with RAG-powered multi-agent analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#features) • [Quick Start](#quick-start) • [Testing](#testing-rag-quality) • [Documentation](FinSight_RAG/README.md)

</div>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) OpenAI API key for LLM-powered agents
- (Optional) Bloomberg Terminal for B-PIPE integration

### Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/Yikai-Li/FinSight.git
cd FinSight
```

2️⃣ **Navigate to the main application directory**
```bash
cd FinSight_RAG
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Configure environment variables (Optional)**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...          # Optional - enables GPT-4o-mini agents
# BLOOMBERG_HOST=localhost        # B-PIPE host (default: localhost)
# BLOOMBERG_PORT=8194             # B-PIPE port (default: 8194)
```

### Starting the Application

#### Option 1: Streamlit Web UI (Recommended)

Launch the interactive web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

**Windows users can also use:**
```bash
start.bat
```

#### Option 2: CLI Mode

Run analysis from the command line without the UI:

```bash
python run_analysis.py --ticker AAPL --verbose
```

**CLI Options:**
```bash
python run_analysis.py --help

Options:
  --ticker TICKER       Stock ticker to analyze (e.g., AAPL, NVDA)
  --top-k K            Number of news chunks to retrieve (default: 10)
  --horizon {5,20}     Investment horizon in days (default: 5)
  --verbose            Enable detailed output
  --save-signal        Save signal to demo_data/signals.json
```

---

## 📋 Features

### 🔴 Live Analysis
- Real-time stock signal generation for any ticker
- Multi-source news aggregation (Yahoo, Bloomberg, Reuters, CNBC, MarketWatch)
- Evidence-based reasoning with full source citation
- Technical factor integration (RSI, MACD, Bollinger Bands, etc.)
- Macro event context (geopolitical news, Fed decisions, sanctions, etc.)

### 📊 Market Scan
- Top 100 stocks by volume/price/market cap
- News-driven ticker discovery
- Quick signal queue generation

### 📈 Market Monitor
- Real-time index tracking (SPY, QQQ, DIA, IWM)
- Sector heatmap visualization
- Signal queue management
- Market pulse scatter plots
- Disagreement flag detection

### 🔍 Evidence Audit
- Thematic market sector analysis
- 8 pre-configured market themes (Semiconductors, Tech, Gold, Energy, etc.)
- One-click AI analysis per theme
- Full agent trace visibility

### 🔬 Evaluation & Backtesting
- Real forward return computation (5d and 20d horizons)
- Hit rate vs random/sentiment baselines
- Signal-level outcome tracking
- Performance metrics dashboard

---

## 🧪 Testing RAG Quality

FinSight includes a comprehensive RAG quality testing suite to evaluate retrieval performance, source diversity, and credibility-weighted ranking.

### Running RAG Tests

**Test a single ticker:**
```bash
cd FinSight_RAG
python test_rag_quality.py --ticker AAPL --verbose
```

**Run comprehensive multi-ticker tests:**
```bash
python test_rag_quality.py --run-all-tests
```

**Benchmark retrieval methods:**
```bash
python test_rag_quality.py --benchmark NVDA
```

**Test source credibility ranking:**
```bash
python test_rag_quality.py --test-credibility TSLA
```

**Save results to JSON:**
```bash
python test_rag_quality.py --run-all-tests --save-results test_results.json
```

### What Gets Tested

The RAG quality test suite evaluates:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Relevance Score** | How relevant retrieved chunks are to the ticker | ≥ 0.30 |
| **Source Diversity** | Number of unique news sources in top-K | ≥ 3 sources |
| **Avg Credibility** | Weighted credibility of retrieved sources | ≥ 0.70 |
| **Bloomberg Priority** | High-authority chunks included in results | Present |
| **Retrieval Time** | Speed of TF-IDF indexing and retrieval | < 3000ms |
| **Coverage Ratio** | Retrieved chunks vs total indexed chunks | Reported |

### Sample Output

```
============================================================
Testing RAG quality for AAPL
============================================================
Ingesting news for AAPL...
Retrieving top-10 chunks...

📊 Retrieval Metrics:
  Total chunks indexed:  47
  Chunks retrieved:      10
  Avg relevance score:   0.782
  Source diversity:      5 sources
  Unique sources:        Yahoo Finance, Reuters, CNBC, Bloomberg, MarketWatch
  Avg credibility:       0.754
  Bloomberg chunks:      2
  Retrieval time:        1247.32ms
  Coverage ratio:        0.213

✅ All quality checks PASSED

📄 Sample Retrieved Chunks (top 3):
  [1] Bloomberg | Apple Inc. Reports Record Q3 Earnings Beat...
      Credibility: 0.95 | Bloomberg: True
      Text: Apple Inc. (NASDAQ: AAPL) reported third-quarter earnings...

  [2] Reuters | Apple shares surge on AI product roadmap announcement
      Credibility: 0.85 | Bloomberg: False
      Text: Shares of Apple rose 3.2% in after-hours trading following...

  [3] CNBC | iPhone 16 pre-orders exceed analyst expectations
      Credibility: 0.78 | Bloomberg: False
      Text: Pre-order data for Apple's latest iPhone 16 lineup shows...
```

### Advanced Testing Options

```bash
# Custom top-K retrieval
python test_rag_quality.py --ticker MSFT --top-k 15 --verbose

# Test specific tickers with result saving
python test_rag_quality.py --ticker NVDA --save-results nvda_test.json

# Full benchmark suite (all default tickers)
python test_rag_quality.py --run-all-tests --verbose
```

### Understanding Test Results

**✅ PASS Criteria:**
- All quality thresholds met
- Multiple diverse sources retrieved
- High-credibility sources prioritized
- Fast retrieval performance

**❌ FAIL Indicators:**
- Low relevance scores (< 0.30) → Query tuning needed
- Low source diversity (< 3) → Ingestion pipeline issue
- Low avg credibility (< 0.70) → Source weighting problem
- No chunks retrieved → Connection or ticker issue

---

## 🏗️ Project Structure

```
FinS/
│
├── README.md                          # This file — main documentation
│
├── FinSight_RAG/                      # Main application directory
│   ├── app.py                         # Streamlit UI (5 tabs)
│   ├── run_analysis.py                # CLI runner
│   ├── test_rag_quality.py            # RAG quality test suite ⭐
│   ├── requirements.txt               # Python dependencies
│   ├── .env.example                   # Environment template
│   ├── start.bat                      # Windows launcher
│   │
│   ├── .streamlit/
│   │   └── config.toml                # Streamlit configuration
│   │
│   ├── demo_data/
│   │   ├── signals.json               # Generated signals
│   │   └── prices.csv                 # Historical price data
│   │
│   ├── docs/                          # Additional documentation
│   │
│   └── src/finance_news_analyzer/     # Core modules
│       ├── agent_runner.py            # Pipeline orchestrator
│       ├── rag_pipeline.py            # News ingestion & retrieval
│       ├── news_ingester.py           # RSS/Bloomberg fetcher
│       ├── technical_factors.py       # Quant indicators
│       ├── macro_events.py            # Geopolitical scanner
│       ├── stock_screener.py          # Market scanner
│       ├── evaluation.py              # Backtesting engine
│       ├── bloomberg_api.py           # B-PIPE integration
│       └── schemas.py                 # Data models
│
├── person2_agent_system_handoff/      # LLM agent system (GPT-4o-mini)
│   └── person2_agent_system/
│       ├── src/orchestration/         # LangGraph workflow
│       └── prompts/                   # Agent prompts
│
└── old_scripts/                       # Archived utility scripts
```

---

## 🎯 Usage Examples

### Example 1: Quick Stock Analysis

```bash
cd FinSight_RAG
streamlit run app.py
```

1. Navigate to **🔴 Live Analysis** tab
2. Enter ticker (e.g., `NVDA`)
3. Click **Run Analysis**
4. Review signal card, evidence sources, and agent reasoning

### Example 2: Market Scan

1. Go to **📊 Market Scan** tab
2. Click **Scan Top 100 by Volume**
3. Review discovered tickers
4. Click **Save to Tickers** for interesting signals

### Example 3: Evaluate Performance

1. Generate several signals over time
2. Go to **🔬 Evaluation** tab
3. Click **Evaluate Signals**
4. Compare hit rate vs baselines

### Example 4: CLI Batch Analysis

```bash
# Analyze multiple tickers
for ticker in AAPL NVDA TSLA MSFT; do
  python run_analysis.py --ticker $ticker --save-signal --verbose
done

# View saved signals
cat demo_data/signals.json | python -m json.tool
```

---

## ⚙️ Configuration

### Pipeline Modes

**Heuristic Mode (Default - No API Key)**
- Offline operation, < 2s per ticker
- Keyword-based sentiment scoring
- Rule-based technical bias
- Fast batch processing

**LLM Mode (OpenAI API Key Required)**
- GPT-4o-mini powered agents
- Full reasoning & thesis formation
- Evidence citation with credibility weighting
- LangGraph multi-agent workflow

### Streamlit Settings (⚙️ Sidebar)

| Setting | Options | Description |
|---------|---------|-------------|
| OpenAI API Key | Text input | Enables LLM mode when provided |
| Bloomberg B-PIPE | Toggle | Enable Bloomberg Terminal integration |
| Ticker List | Dropdown | Session-persistent ticker selection |
| Analysis Horizon | 5d / 20d | Forward return evaluation period |

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                            │
│  Yahoo • Bloomberg • Reuters • CNBC • MarketWatch • Google News  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE (TF-IDF)                       │
│  Article Fetch → Chunking → Indexing → Top-K Retrieval         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENRICHMENT LAYER                             │
│  Technical Factors (RSI, MACD, Bollinger)                      │
│  Macro Events (Fed, Wars, Sanctions, Summits)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  THREE-AGENT PIPELINE                           │
│  Analyst → Evidence extraction, sentiment scoring               │
│  Strategist → Thesis formation, direction setting               │
│  Decision → Confidence calibration, final signal                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SIGNAL PACKET                              │
│  Direction • Confidence • Reasoning • Citations • Metrics       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue: "No module named 'src.finance_news_analyzer'"**
```bash
# Make sure you're in the FinSight_RAG directory
cd FinSight_RAG
python app.py  # or streamlit run app.py
```

**Issue: "No chunks retrieved"**
- Check internet connection
- Verify ticker symbol is valid (use yfinance convention)
- Try a more popular ticker (AAPL, MSFT, NVDA)

**Issue: Bloomberg connection failed**
- Ensure Bloomberg Terminal is running
- Check BLOOMBERG_HOST and BLOOMBERG_PORT in .env
- Bloomberg B-PIPE requires paid subscription

**Issue: Slow retrieval (> 5s)**
- Reduce `--top-k` value
- Check network latency to news sources
- Consider caching articles locally

---

## 📈 Performance Benchmarks

| Operation | Time (avg) | Notes |
|-----------|------------|-------|
| News ingestion (10 sources) | 1.2s | Network dependent |
| TF-IDF indexing (50 chunks) | 0.15s | Local computation |
| Top-K retrieval (K=10) | 0.05s | Sub-100ms typical |
| Heuristic pipeline (full) | 1.8s | Offline mode |
| LLM pipeline (GPT-4o-mini) | 4.5s | API latency + reasoning |
| RAG quality test (1 ticker) | 1.5s | Includes ingestion |

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional news sources (Wall Street Journal, Financial Times, Barron's)
- Enhanced evaluation metrics (Sharpe ratio, max drawdown)
- Alternative LLM backends (Claude, Llama, Gemini)
- Real-time WebSocket news feeds
- Portfolio-level signal aggregation

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 📚 Additional Resources

- **[Full Technical Documentation](FinSight_RAG/README.md)** — Detailed architecture, API reference, and evaluation methodology
- **[Project Proposal](FinSight_RAG/Project_Proposal.pdf)** — Original design document
- **[Project Plan](FinSight_RAG/final_project_plan_0423.docx)** — Development roadmap

---

## ⚠️ Disclaimer

FinSight is an **educational research platform** for institutional-quality financial signal generation. It is **not financial advice**. All investment decisions should be made with proper due diligence and consultation with qualified financial advisors.

Market data and news sources may have delays, inaccuracies, or biases. Past performance does not guarantee future results.

---

<div align="center">

**Built with ❤️ for evidence-grounded investment research**

[⬆ Back to Top](#finsight--multi-agent-financial-news-intelligence-platform)

</div>
