# 📊 Ceded Reinsurance Analytics Platform (Demo)

A professional **Streamlit-based demonstration platform** for **ceded reinsurance governance, controls, and analytics**, covering:

- Exposure & Credit Control (CEC)
- Cash Pairing & Reconciliation
- Consultant Copilot (Narratives & Q&A)

This repository demonstrates how fragmented insurance data can be transformed into a **governed, auditable, and decision-ready analytics layer**.

> ⚠️ All data in this repository is synthetic and for illustration only.

---

## 🎯 Purpose

This demo is designed for:

- Client presentations  
- Internal capability demonstrations  
- Proof-of-value / pilot discussions  
- Consulting delivery accelerators  

It shows how insurers and reinsurers can:

- Improve governance and transparency  
- Reduce manual reconciliation  
- Strengthen credit and exposure controls  
- Generate executive-ready narratives  

---

## 🧩 Key Modules

### 1️⃣ Exposure & Credit Control (CEC)
- Unified portfolio exposure view  
- Counterparty concentration analysis  
- Credit utilization monitoring  
- Stress testing  
- Full audit trail  

### 2️⃣ Cash Pairing & Reconciliation
- Automated cash-to-statement matching  
- Confidence scoring  
- Exception queue  
- Variance analysis  
- Operational worklists  

### 3️⃣ Consultant Copilot
- Executive narrative generation  
- Evidence-backed summaries  
- Controlled Q&A  
- Deterministic fallback mode  

---

## 📁 Repository Structure

```
synpulse_demo/
│
├── app.py
├── config/
├── data/
├── pages/
├── services/
├── components/
├── models/
├── utils/
```

---

## ⚙️ Requirements

- Python 3.9+
- pip or conda
- Virtual environment (recommended)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ceded-re-analytics-demo.git
cd ceded-re-analytics-demo
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is missing:

```bash
pip install streamlit pandas numpy plotly pyyaml
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open:

http://localhost:8501

---

## 📊 Demo Data

Located in:

```
data/raw/
```

Synthetic demo files:

- pas.csv
- claims.csv
- placements.csv
- cash.csv
- statements.csv
- counterparties.csv

Regenerate:

```bash
python -m scripts.regenerate_raw_demo_data
```

---

## 🔐 Copilot Configuration (Optional)

Without key: fallback mode.

With OpenAI:

```bash
export OPENAI_API_KEY="your_key"
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4o-mini"
```

---

## 🏛 Governance

Includes:

- Validation
- Join checks
- Lineage
- Assumptions
- Evidence packs

---

## 📤 Exports

- CSV
- JSON
- TXT

For audit and reporting.

---

## 📄 License

MIT License

Demo and consulting purposes only.

---

## 🤝 Contact
stella.dong@reinsuranceanalytics.io 
contact@reinsuranceanalytics.io 

Reinsurance Analytics
Ceded Re Governance Platform
https://www.reinsuranceanalytics.io

---

## ⭐ Value

From spreadsheets → governed analytics.
