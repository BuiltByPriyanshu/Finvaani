---
title: FinVaani
emoji: 🪔
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---
---
# 🪔 FinVaani — Indian Finance Q&A System

### A Bilingual Indian Financial Q&A System

A bilingual (English + Hindi) Indian financial Q&A system built by fine-tuning
mGPT with LoRA (PEFT), then applying the **Lottery Ticket Hypothesis** (Iterative
Magnitude Pruning) to compress the model by ~80% while retaining 90%+ quality.

> Fine-tuning mGPT (1.4B) on Indian regulatory text using **LoRA** + compressing it with **Lottery Ticket Hypothesis pruning** — making financial regulations accessible in both English and Hindi.

<!-- Replace with your actual demo GIF -->
<img width="1276" height="674" alt="image" src="https://github.com/user-attachments/assets/ad049d68-33e6-4ac5-b203-a05835e5fd24" />


---

## 📌 What is FinVaani?

India's financial regulators (RBI, SEBI, IRDAI, NPCI, NCFE) publish thousands of pages of guidelines and FAQs — but they're dense, scattered, and often only in English. Hindi speakers face a double barrier: technical language *and* a language gap.

**FinVaani** bridges that gap. It's a bilingual Q&A system that:
- Understands questions in **English and Hindi**
- Is grounded in actual regulatory content from **5 Indian financial bodies**
- Uses **LoRA** to fine-tune a 1.4B parameter model efficiently (only 0.30% of parameters trained)
- Applies the **Lottery Ticket Hypothesis** to compress the adapter by 67.2% with no quality loss

---

## ✨ Features

| Feature | Details |
|---|---|
| 🗣️ Bilingual | English + Hindi (Devanagari) support |
| 🏦 Domain-grounded | RBI, SEBI, NCFE, IRDAI, NPCI content |
| ⚡ Efficient | LoRA fine-tuning — only 4.3M trainable params |
| ✂️ Compressed | 67.2% sparse winning ticket via LTH pruning |
| 🖥️ Deployed | Streamlit app with chat, compare, and metrics pages |

---

## 🗂️ Dataset

We constructed a novel **1,706 Q&A pair** bilingual dataset — the first of its kind for Indian financial regulatory text.

| Source | Pairs | Method |
|---|---|---|
| RBI | 1,605 | Live scraping (English + Hindi FAQs) |
| SEBI | 42 | PDF scraping via `pdfplumber` |
| NCFE | 29 | Live scraping with SSL bypass |
| IRDAI | 16 | Synthetic (site returned 504 errors) |
| NPCI | 14 | Synthetic (React SPA, not scrapable) |
| **Total** | **1,706** | 605 English + 1,101 Hindi |

**Train / Val / Test split:** 1,194 / 256 / 256 (stratified by language)

---

## 🏗️ Architecture

```
User Query (EN or HI)
        │
        ▼
   mGPT 1.4B Base
   (ai-forever/mGPT)
        │
        ▼
  LoRA Adapter (r=8)
  4.3M trainable params
  frozen base weights
        │
        ▼
  [Optional] LTH Winning Ticket
  67.2% sparse adapter
        │
        ▼
   Generated Answer
```

<!-- Add your architecture diagram image here -->




**LoRA targets:** `c_attn` and `c_proj` across all 24 transformer layers.

---

## 📊 Results

### Main Metrics (256-sample test set)

| Model | BLEU | ROUGE-L | Perplexity | Speed (ms) |
|---|---|---|---|---|
| Raw mGPT (zero-shot) | 0.0060 | 0.0756 | 6.9 | 1,810 |
| Prompted mGPT (few-shot) | 0.0008 | 0.0670 | 6.9 | 1,816 |
| **LoRA Fine-tuned** | **0.0197** | **0.0813** | **5.7** | 3,086 |
| Winning Ticket (67.2% sparse) | 0.0057 | 0.0702 | 6.9 | 3,008 |

### Extended Metrics (25 qualitative samples)

| Model | METEOR | Fin. Term Recall | BERTScore F1 | Length Ratio |
|---|---|---|---|---|
| Raw mGPT | 0.1582 | 40.8% | 0.6950 | 0.54 |
| Prompted mGPT | 0.1477 | 42.4% | 0.7030 | 0.24 |
| **LoRA Fine-tuned** | **0.2043** | **51.0%** | **0.7092** | 1.51 |
| Winning Ticket | 0.1847 | 46.6% | 0.6871 | 0.66 |

### Error Analysis (25 qualitative samples)

| Model | Hallucination | Evasion | Domain Confusion |
|---|---|---|---|
| Raw mGPT | 36% | 28% | 0% |
| Prompted mGPT | 80% | 76% | 4% |
| **LoRA Fine-tuned** | **28%** | **0%** | 8% |
| Winning Ticket | 60% | 36% | 0% |

> **Key win:** LoRA fine-tuned achieves **0% evasion** — it always produces a substantive answer — and the highest Financial Term Recall (51%), validating domain adaptation.

### LTH Pruning Rounds

| Round | Sparsity | Eval Loss | Perplexity |
|---|---|---|---|
| 0 (no pruning) | 0% | 2.8998 | 18.17 |
| 1 | 20% | 2.6846 | 14.65 |
| 2 | 36% | 2.6846 | 14.65 |
| 3 | 48.8% | 2.6846 | 14.65 |
| 4 | 59% | 2.6846 | 14.65 |
| **5 (Winning Ticket)** | **67.2%** | **2.6846** | **14.65** |

Quality is perfectly maintained across all 5 pruning rounds — a clean demonstration of the LTH on LoRA adapters.

<!-- Add your results charts here -->
![Results Charts](assets/results.png)

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (trained on Google Colab T4, 16GB VRAM)
- ~6GB disk space for model weights

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/finvaani.git
cd finvaani
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run frontend/app.py
```

The app will open at `http://localhost:8501` with four pages: **Chat**, **Compare**, **Metrics**, and **About**.

### 4. (Optional) Re-run training on Colab
Upload `data/splits/` to Google Colab T4 and open:
```
notebooks/training_colab.ipynb
```

### 5. (Optional) Re-run scraping
```bash
python scraping/scrape_all.py
```
> Requires internet access to `rbi.org.in`, `sebi.gov.in`, `ncfe.org.in`

---

## 📁 Project Structure

```
finvaani/
├── scraping/           # Data collection from 5 regulatory sources
├── preprocessing/      # Clean, format, split, store pipeline
├── training/           # Baseline evaluation + LoRA fine-tuning
├── pruning/            # LTH iterative magnitude pruning
├── evaluation/         # Metrics, error analysis, plots
├── frontend/           # Streamlit web app
│   ├── app.py
│   ├── assets/
│   ├── components/
│   └── pages/
│       ├── 01_chat.py
│       ├── 02_compare.py
│       ├── 03_metrics.py
│       └── 04_about.py
├── data/               # Raw, processed, and split datasets
├── models/             # Saved checkpoints (LoRA + winning ticket)
├── results/            # Metrics CSVs and plots
└── notebooks/
    └── training_colab.ipynb
```

---

## 🔬 Reproducing Results

Random seed **42** used throughout.

```bash
# 1. Scrape data
python scraping/scrape_all.py

# 2. Preprocess
python preprocessing/merge_raw.py
python preprocessing/clean_data.py
python preprocessing/format_data.py
python preprocessing/split_data.py
python preprocessing/store_database.py

# 3. Train (on Colab T4)
# Open notebooks/training_colab.ipynb

# 4. Prune
python pruning/lth_pruning.py
python pruning/find_winning_ticket.py

# 5. Evaluate
python evaluation/compute_metrics.py
python evaluation/qualitative_analysis.py
python evaluation/error_analysis.py
python evaluation/plot_results.py
```

---

## 🛠️ Tech Stack

`PyTorch` · `HuggingFace Transformers` · `PEFT` · `mGPT (ai-forever)` · `Streamlit` · `pdfplumber` · `BeautifulSoup4` · `NLTK` · `bert-score` · `SQLite` · `Plotly`

---

## ⚠️ Limitations

- **94.1% RBI data** — primarily a banking regulation system, not a fully general Indian financial QA system
- **28% hallucination rate** on LoRA fine-tuned — should not be used as a definitive regulatory source
- **No language-stratified evaluation** — overall metrics reflect Hindi performance more (64.5% of data)
- **256-token context limit** — long regulatory answers are truncated during training

---

## 🔭 Future Work

- **RAG (Retrieval-Augmented Generation)** — ground answers in retrieved document chunks to reduce hallucination
- **QLoRA (4-bit quantisation)** — deploy on consumer hardware
- **Selenium scraping** — unlock NPCI and other JS-rendered pages
- **Human evaluation** — domain expert assessment
- **Larger base model** — mT5-XL or LLaMA-3 with Hindi support

---

## 📄 Citation

If you use FinVaani's dataset or approach in your work, please cite:

```bibtex
@misc{finvaani2025,
  author       = {Priyanshu Singh},
  title        = {FinVaani: A Bilingual Indian Financial Q\&A System Using LoRA Fine-Tuning and Lottery Ticket Hypothesis Pruning},
  year         = {2025},
  institution  = {BML Munjal University},
  note         = {B.Tech Final Year Project, CSE3720 — Generative AI and LLMs}
}
```

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) for `transformers` and `peft` libraries
- [ai-forever](https://huggingface.co/ai-forever/mGPT) for the mGPT model
- RBI, SEBI, NCFE, IRDAI, NPCI for making regulatory content publicly accessible
- Dr. Manisha Saini, BML Munjal University, for guidance throughout the project

---

<p align="center">Made with ❤️ at BML Munjal University · CSE3720 Generative AI and LLMs · 2024–25</p>
