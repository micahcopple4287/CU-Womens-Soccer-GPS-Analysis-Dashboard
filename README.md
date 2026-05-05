# CU Women's Soccer — Catapult GPS Microcycle Load & Velocity Zone Analysis
 
A capstone data science project analyzing Catapult GPS tracking data for the **University of Colorado Women's Soccer** program. This project includes a full statistical analysis of microcycle training loads and velocity zone distributions, as well as an interactive Streamlit dashboard for real-time monitoring and reporting.
 
---
 
## Project Overview
 
Athlete load management is critical for optimizing performance and reducing injury risk. This project leverages Catapult GPS data to:
 
- Quantify **microcycle training loads** across the competitive season
- Analyze **velocity zone distributions** by player, position, and session type
- Establish **data-driven thresholds** for flagging overload and underload
- Provide coaching staff with an **interactive dashboard** for ongoing monitoring
---
 
## Contributors
 
| Name | Role |
|------|------|
| Micah Copple | Data analysis, R modeling, dashboard development |
| Rishekesan Senthil Kumar Vanathi | Data analysis, R modeling, dashboard development |
| Michael Jeffery Baker | Research, analysis, reporting |
| Quinn Conroy | Research, analysis, reporting |
 
---
 
## Tech Stack
 
**Analysis**
- R / RMarkdown
- tidyverse, ggplot2, scales
**Dashboard**
- Python / Streamlit
- pandas, plotly
---
 
## Features
 
- **Microcycle load tracking** — session-by-session and weekly load trends across the season
- **Velocity zone analysis** — distribution of time spent in each velocity band by player and session
- **Percentile-based thresholds** — statistically grounded benchmarks rather than arbitrary cutoffs
- **Interactive dashboard** — filtering by player, date range, and session type
- **Exportable visualizations** — figures ready for staff reports and presentations
---
 
## Repository Structure
 
```
├── Capstone.Rmd              # Full R analysis and write-up
├── dashboard/
│   ├── dash2.py              # Streamlit dashboard application
│   ├── requirements.txt      # Python dependencies
│   └── DATA_UPDATE_GUIDE.md  # Instructions for updating with new data
└── figures/                  # Exported visualizations
```
 
---
 
## Data
 
Raw GPS data is proprietary to **CU Athletics** and is not included in this repository. The analysis pipeline expects a Catapult export in standard CSV format. See `dashboard/DATA_UPDATE_GUIDE.md` for instructions on connecting your own data source.
 
---
 
## Running the Dashboard
 
1. Install dependencies:
```bash
pip install -r dashboard/requirements.txt
```
 
2. Launch the app:
```bash
streamlit run dashboard/dash2.py
```
 
---
 
## Academic Context
 
This project was completed as a **capstone project** for the University of Colorado Statistics and Data Science program.
