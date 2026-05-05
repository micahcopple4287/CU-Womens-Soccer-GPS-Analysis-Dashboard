# CU WSOC Performance Intelligence Dashboard (`dash2.py`)

Offline Streamlit dashboard for CU Boulder Women's Soccer: Catapult GPS, schedules, game stats, injury logs, wellness, anomaly detection, next-session forecasting, availability ML, and a natural-language "Chat with Data" tab (TF-IDF RAG). **No cloud APIs required for core use.**

---

## What's in this folder

```
.
├── dash2.py                ← the entire Streamlit application (single-file)
├── requirements.txt        ← Python dependencies
├── README.md               ← you are here
├── DATA_UPDATE_GUIDE.md    ← how staff add a new season's data
└── .gitignore              ← keeps Catapult / injury / wellness data out of git
```

That's it. One Python file plus your data folder.

---

## Requirements

- **Python 3.10+** (3.10 or 3.11 recommended)
- Dependencies listed in `requirements.txt`

---

## Quick start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# or
source .venv/bin/activate        # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Drop your data next to `dash2.py`

The dashboard expects this layout (paths can also be overridden in the sidebar at run-time):

```
your-project/
├── dash2.py
├── requirements.txt
├── outputs/
│   └── clean_session_level_with_acwr.csv      ← processed Catapult sessions (required)
└── Data/
    ├── 2023/
    │   ├── 2023 Schedule Data/2023 Schedule.xlsx
    │   ├── 2023 Game Stat Data/                ← per-opponent game CSVs
    │   └── 2023 Injuries.csv
    ├── 2024/
    │   ├── 2024 Schedule Data/2024 Schedule.xlsx
    │   ├── 2024 Clean Game Stat Data/
    │   ├── 2024 Injuries.csv
    │   └── 2024 Wellness.csv
    └── 2025/
        ├── 2025 Schedule Data/2025 Schedule.xlsx
        ├── 2025 Clean Game Stat Data/
        ├── 2025 Injuries.csv
        └── 2025 Wellness.csv
```

See **`DATA_UPDATE_GUIDE.md`** for full column specs and how to add a new season.

### 5. Run

```bash
python -m streamlit run dash2.py
```

Open **http://localhost:8501**.

---

## Dashboard tabs

| Tab | Purpose |
|-----|---------|
| **Dashboard** | KPIs, injury board, wellness, smart alerts |
| **Team** | Squad trends and load over time |
| **Player** | Single-athlete deep dive |
| **Match Center** | Training vs match calendar context |
| **Game Stats** | Season box scores and charts |
| **Microcycle** | MD-window load / taper view |
| **Opponent** | Physical comparison vs opponents |
| **Health & Wellness** | Injury timelines and wellness trends |
| **Position Groups** | Load and readiness by line |
| **Anomalies** | Isolation Forest session flags |
| **Forecast** | Next-session metric prediction (multi-model leaderboard) |
| **Coach ML** | Next-session availability risk (RF, ET, AdaBoost, SVM, MLP, KNN, GBDT, XGBoost) |
| **Chat with Data** | Offline Q&A over your loaded tables |

Every tab includes a **Coach Guide** expander that summarises how to read it, what the metric means, and what decision it should drive.

---

## Optional packages

| Package | Use |
|---------|-----|
| `shap` | SHAP-style explainability where enabled |
| `python-docx` | Word export for ML reports |

```bash
pip install shap python-docx
```

The app runs without them.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| "Catapult CSV not found" | Place `outputs/clean_session_level_with_acwr.csv` as above, or paste the correct path in the sidebar |
| Schedule load error | Open **Schedule loader debug** in the app; confirm sheet name and header row |
| Empty Health / ML tabs | Confirm injury and wellness CSV paths; missing files show a warning banner |
| `streamlit` not found | Use `python -m streamlit run dash2.py` from the same environment where you ran `pip install` |
| Stale "Chat with Data" answers | Delete the auto-created `.rag_cache/` folder and restart |

---

## Pushing to GitHub

From the folder that contains `dash2.py`:

```bash
git init
git add dash2.py requirements.txt README.md .gitignore DATA_UPDATE_GUIDE.md
git commit -m "Add CU WSOC performance dashboard (dash2)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> **Data privacy:** Catapult, injury, and wellness files often contain identifiable athlete information. The provided `.gitignore` already excludes `Data/`, `outputs/`, and all `*.csv` / `*.xlsx` files so they will not be committed. Use a private repository or strip identifiers before sharing publicly.

---

## Project team

- Rishekesan S V — rishe7742@colorado.edu
- Michael Baker — michael.n.baker@colorado.edu
- Quinn Conroy — quinn.conroy-1@colorado.edu
- Micah Copple — micah.copple@colorado.edu

University of Colorado Boulder — Spring 2026

---

## More documentation

See **`DATA_UPDATE_GUIDE.md`** for how staff can refresh datasets each season.
