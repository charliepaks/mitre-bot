# MITRE ATT&CK Threat Classification App  

*Automated threat alert classification with reasoning powered by AI*

---

## ✨ Overview

This Streamlit application lets you input a security alert’s title and/or description and instantly returns:  

- **MITRE ATT&CK Tactic** (e.g., Persistence, Privilege Escalation)  
- **MITRE ATT&CK Technique** (e.g., T1078 – Valid Accounts)  
- **Detailed reasoning** explaining how the alert maps to the tactic & technique  

It uses a local MITRE ATT&CK PDF database, indexed via FAISS, and leverages AI for high‑quality, context‑aware classification.

---

## 🚀 Features

- **Fast RAG Retrieval**  
  - Splits PDF into 500‑token chunks with 100‑token overlap  
  - Stores embeddings in a FAISS vector store for sub‑second searches  

- **GPT‑4o–Backed Reasoning**  
  - Uses `langchain.chat_models.ChatOpenAI(model_name="gpt-4o")`  
  - Returns human‑readable explanations along with classifications  

- **Intuitive Streamlit UI**  
  - Sidebar configuration for API key & PDF path  
  - One‑click “Load & Index” step (cached per session)  
  - Clean input form and expandable source excerpts  

- **Easy Deployment**  
  - Pure Python & `requirements.txt` setup  
  - Can deploy on any hosting (Azure, DigitalOcean, Heroku, etc.)  

---

## 💾 Installation

 **Clone the repo**  

```bash
   git clone https://github.com/charliepaks/mitre-bot.git
   cd mitre-bot
```

 **Create virtual environment**

```bash

python3 -m venv venv
source venv/bin/activate
```

 **Install dependencies**

```bash

pip install -r requirements.txt
```

 Place the MITRE ATT&CK PDF

- Name it mitre_attack.pdf (or update the path in the sidebar)

- Ensure it resides alongside app.py

## ⚙️ Configuration

1. OpenAI API Key

- In the Streamlit sidebar

2. PDF Path

- Default: mitre_attack.pdf

- Change if your file has a different name or location.

## ▶️ Usage

Run the app:

```bash

streamlit run app.py
```

Load & Index

- Click Load and Index MITRE Database

- Wait for the FAISS index to build (cached for the session)

Analyze Alerts

- Enter a Title and/or Description

- Click Analyze Threat

- View the predicted Tactic, Technique, and reasoning

- Expand Show source documents to inspect retrieved passages

## 🛠️ Customization & Tuning

- Chunking Strategy:
Adjust chunk_size and chunk_overlap in RecursiveCharacterTextSplitter for more or less context per retrieval.

- Retrieval Depth:
Change the k in vector_store.as_retriever(search_kwargs={"k": 3}) to retrieve more/less candidate passages.

- LLM Parameters:
Modify temperature, model_name (e.g., "gpt-4o" when available), or add max_tokens in ChatOpenAI() for varied response styles.

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add awesome feature"`)  
4. Push to your branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

---

## ❓ Troubleshooting

- **“File not found” error**:  
  Confirm `mitre_attack.pdf` exists in the same directory as `app.py`, or update the path in the sidebar.  

- **Slow indexing**:  
  Try increasing overlap or reducing chunk size, or run on a more powerful machine.  

- **API errors**:  
  Ensure your OpenAI key is valid and you have quota for GPT‑4o calls.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  
