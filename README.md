
# Ceded Reinsurance Governance Demo

An end-to-end demonstration platform for ceded reinsurance governance, exposure management, cash reconciliation, and executive reporting.

This project showcases how insurers and reinsurers can build a single governed analytics layer above existing systems to improve transparency, control, and decision-making.

---

## 🚀 Overview

This demo includes three core modules:

### 1. Exposure & Credit Control (CEC)
- Portfolio exposure rollups
- Counterparty concentration analysis
- Credit limit monitoring
- Stress testing and scenario analysis
- Full audit and lineage tracking

### 2. Cash Pairing & Reconciliation
- Automated cash-to-statement matching
- Confidence scoring
- Exception prioritization
- Operational worklists
- Exportable reconciliation outputs

### 3. Consultant Copilot
- Executive-ready narratives
- Guided prompts
- Evidence-backed explanations
- Exportable memos and reports
- Optional LLM integration

All modules operate on a shared, governed data foundation.

---

## 📁 Project Structure

```
ceded-re-governance-demo/
│
├── app.py
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── pages/
├── services/
├── components/
├── models/
├── utils/
├── scripts/
├── README.md
├── LICENSE
└── requirements.txt
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- pip
- Git
- Virtual environment (recommended)

### Setup

```bash
git clone https://github.com/stelladong-RA/ceded-re-governance-demo.git
cd ceded-re-governance-demo

python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open in browser:

http://localhost:8501

---

## 📊 Demo Data

Synthetic demo data only.

Location:
data/raw/

No client data included.

---

## 🛡️ Governance

- Lineage tracking
- Join validation
- Completeness checks
- Assumptions disclosure
- Audit evidence

---

## 🤖 LLM Support

Fallback mode by default.

Enable via:

export OPENAI_API_KEY="key"

services/llm_service.py

---

## ☁️ Deployment

Supports Streamlit Cloud.

---

## 📄 License

MIT License.

---

## ⚠️ Disclaimer

Demo only. Not production software.

---

## 📬 Contact

Maintained by Stella Dong.
