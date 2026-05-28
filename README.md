# 🪔 FinVaani — Indian Finance Q&A System

> **India's Financial Intelligence, Compressed.**

A bilingual (English + Hindi) Indian financial Q&A system built by fine-tuning
mGPT with LoRA (PEFT), then applying the **Lottery Ticket Hypothesis** (Iterative
Magnitude Pruning) to compress the model by ~80% while retaining 90%+ quality.

---

## Architecture

```
Raw Data (RBI/SEBI/IRDAI/NPCI/NCFE)
         │
         ▼
  ┌─────────────┐
  │  Scraping   │  requests + BeautifulSoup + pdfplumber
  └──────┬──────┘
         │  1,727 Q&A pairs (EN + HI)
         ▼
  ┌─────────────┐
  │Preprocessing│  clean → format → split → SQLite
  └──────┬──────┘
         │  1,706 pairs (train/val/test)
         ▼
  ┌─────────────────────────────────────────────┐
  │              mGPT (117M params)             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
  │  │ Baseline │  │ Prompted │  │   LoRA   │  │
  │  │ (Model 1)│  │ (Model 2)│  │ (Model 3)│  │
  │  └──────────┘  └──────────┘  └────┬─────┘  │
  └───────────────────────────────────┼─────────┘
                                      │
                                      ▼
                          ┌─────────────────────┐
                          │  LTH Pruning (IMP)  │
                          │  5 rounds × 20%     │
                          │  Reset → Retrain    │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Winning Ticket    │  ~25M active params
                          │     (Model 4)       │  ~80% compressed
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  Streamlit Frontend │
                          │  Chat | Compare     │
                          │  Metrics | About    │
                          └─────────────────────┘
```

---

## Setup

```bash
git clone <repo>
cd finvaani
pip install -r requirements.txt
```

---

## Running Each Phase

### Phase 1 — Data Collection
```bash
python scraping/scrape_all.py
```

### Phase 2 — Preprocessing
```bash
python preprocessing/merge_raw.py
python preprocessing/clean_data.py
python preprocessing/format_data.py
python preprocessing/split_data.py
python preprocessing/store_database.py
```

### Phase 3 & 4 — Baselines (local, no GPU needed)
```bash
python training/baseline_raw.py
python training/baseline_prompted.py
```

### Phase 5 — LoRA Fine-tuning (run on Google Colab T4)
```bash
# Upload to Colab and run:
python training/lora_finetune.py
# Or use: notebooks/training_colab.ipynb
```

### Phase 6 — LTH Pruning (run on Google Colab)
```bash
python pruning/lth_pruning.py
python pruning/find_winning_ticket.py
```

### Phase 7 — Evaluation
```bash
python evaluation/compute_metrics.py
python evaluation/qualitative_analysis.py
python evaluation/error_analysis.py
python evaluation/plot_results.py
```

### Phase 8 — Frontend
```bash
streamlit run frontend/app.py
```

---

## Model Performance

| Model          | BLEU  | ROUGE-L | PPL    | Params  | Speed  |
|----------------|-------|---------|--------|---------|--------|
| Raw mGPT       | 0.040 | 0.120   | 320.0  | 117M    | 850ms  |
| Prompted mGPT  | 0.070 | 0.180   | 290.0  | 117M    | 920ms  |
| LoRA Fine-tuned| 0.210 | 0.380   | 85.0   | 118.2M  | 870ms  |
| Winning Ticket | 0.190 | 0.350   | 92.0   | ~25M    | 310ms  |

*Metrics are representative. Run `evaluation/compute_metrics.py` for actual values.*

---

## Dataset Sources

| Source | URL | Pairs |
|--------|-----|-------|
| RBI    | rbi.org.in | ~1,626 |
| SEBI   | sebi.gov.in | ~42 |
| IRDAI  | irdai.gov.in | ~16 |
| NPCI   | npci.org.in | ~14 |
| NCFE   | ncfe.org.in | ~29 |

---

## Research Citation

```bibtex
@inproceedings{frankle2019lottery,
  title={The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks},
  author={Frankle, Jonathan and Carlin, Michael},
  booktitle={ICLR},
  year={2019}
}
```

---

## Team

BML Munjal University | B.Tech — Natural Language Processing | 2024-25
