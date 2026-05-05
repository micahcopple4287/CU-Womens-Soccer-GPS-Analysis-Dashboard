# CU WSOC Dashboard — Data Update Guide

This guide explains how to add new season data so the dashboard reflects current information.

---

## Folder Structure

The dashboard expects data organized by year under the `Data/` folder:

```
Project CU_SOCCER/
├── Data/
│   ├── 2023/
│   │   ├── 2023 Injuries.csv
│   │   ├── 2023 Schedule Data/
│   │   │   └── 2023 Schedule.xlsx
│   │   └── 2023 Game Stat Data/
│   │       ├── Game1.csv
│   │       └── ...
│   ├── 2024/
│   │   ├── 2024 Injuries.csv
│   │   ├── 2024 Wellness.csv
│   │   ├── 2024 Schedule Data/
│   │   │   └── 2024 Schedule.xlsx
│   │   └── 2024 Clean Game Stat Data/
│   │       ├── Game1.csv
│   │       └── ...
│   └── 2025/
│       ├── 2025 Injuries.csv
│       ├── 2025 Wellness.csv
│       ├── 2025 Schedule Data/
│       │   └── 2025 Schedule.xlsx
│       └── 2025 Clean Game Stat Data/
│           ├── Game1.csv
│           └── ...
├── outputs/
│   └── clean_session_level_with_acwr.csv   ← Main Catapult export
└── dash2.py
```

---

## Step-by-Step: Adding a New Season (e.g., 2026)

### 1. Create the folder structure

Create these folders inside `Data/`:
```
Data/2026/
Data/2026/2026 Schedule Data/
Data/2026/2026 Clean Game Stat Data/
```

### 2. Prepare each dataset

#### Catapult GPS Data (`outputs/clean_session_level_with_acwr.csv`)

This is the main data source. When you export a new Catapult session file:

1. Export from Catapult Cloud as CSV (session-level, not raw 10 Hz).
2. Ensure these columns exist (names must match exactly):
   - `player_id` — Anonymized or real athlete ID
   - `date` — Session date
   - `total_player_load` — Accumulated load for the period
   - `player_load_per_min_est` — Load per minute
   - `total_distance` — Distance in **miles**
   - `maximum_velocity` — Max speed in **mph**
   - `total_acceleration_load` — Acceleration load
   - `explosive_efforts` — Explosive effort count
   - `max_acceleration` / `max_deceleration` — In m/s²
   - `session_code_tag` — "Match", "Practice", "Training", etc.
   - `position_name` — Player position
3. Append new rows to the existing `clean_session_level_with_acwr.csv`, or replace the entire file with all seasons combined.

**Data quality checks the dashboard applies automatically:**
- Max velocity >= 22 mph → set to NaN (GPS artifact)
- Total distance >= 10 miles → set to NaN (GPS artifact)
- Any metric beyond 3 standard deviations → set to NaN

#### Schedule (`2026 Schedule Data/2026 Schedule.xlsx`)

1. Create an Excel file with at least these columns:
   - `Date` — Match date (any standard date format)
   - `Opponent` — Team name (e.g., "Texas A&M")
2. Optional columns: `Venue` (Home/Away), `Result` (W/L/T + score)
3. The dashboard auto-detects the header row, but keeping it in row 1 is safest.

#### Injuries (`2026 Injuries.csv`)

CSV with these columns:
| Column | Description |
|--------|-------------|
| `player_id` | Athlete identifier (must match Catapult IDs) |
| `injury_status` | One of: `Available`, `Limited`, `As Tolerated`, `Out` |
| `days_in_status` | Number of days in current status |
| `injury_date` or `status_start` | Date the status began |

#### Wellness (`2026 Wellness.csv`)

CSV with these columns:
| Column | Description | Scale |
|--------|-------------|-------|
| `player_id` | Athlete identifier | — |
| `wellness_date` | Date of wellness entry | — |
| `wellness_total` | Overall wellness score | 0–20 |
| `physical_score` | Physical feeling | 1–5 |
| `mental_score` | Mental state | 1–5 |
| `sleep_score` | Sleep quality | 1–5 |
| `soreness_score` | Soreness (5 = less sore) | 1–5 |

#### Game Stats (`2026 Clean Game Stat Data/*.csv`)

One CSV per match with per-player box scores:
| Column | Description |
|--------|-------------|
| `player_id` | Athlete identifier |
| `MIN` | Minutes played |
| `G` | Goals |
| `A` | Assists |
| `Sh` | Shots |
| `SOG` | Shots on goal |
| `GA` | Goals against (GK) |
| `Saves` | Saves (GK) |

File names should include the opponent (e.g., `vs_Texas_AM.csv`).

### 3. Update `dash2.py` paths

Open `dash2.py` and find the `PATHS` section near the top. Add new entries:

```python
SCHED_2026 = _PROJECT_ROOT / "Data" / "2026" / "2026 Schedule Data" / "2026 Schedule.xlsx"
GAME_2026_DIR = _PROJECT_ROOT / "Data" / "2026" / "2026 Clean Game Stat Data"
INJ_2026 = _PROJECT_ROOT / "Data" / "2026" / "2026 Injuries.csv"
WELL_2026 = _PROJECT_ROOT / "Data" / "2026" / "2026 Wellness.csv"
```

Then update the data loading calls further down:
- Add the new schedule path to `load_all_schedules()`
- Add the new game stats folder to the game loading section
- Add the new injury/wellness files to `load_all_injuries()` and `load_all_wellness()`

### 4. Clear cached data

Delete the `.rag_cache/` folder to force the RAG system to rebuild its knowledge base with the new data:

```
del .rag_cache\*
```

### 5. Run the dashboard

```bash
streamlit run dash2.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Injury tiles show "No injury data" | Check that `player_id` values match between injury CSV and Catapult data |
| Wellness tiles show "No wellness data" | Same — ensure `player_id` matches; wellness is joined within a 3-day window |
| Schedule not loading | Check that the Excel file has `Date` and `Opponent` columns |
| Suspiciously high velocity values | The dashboard caps at 22 mph. If you see values near 20-22, verify with coaching staff |
| RAG answers seem stale | Delete `.rag_cache/` and restart |

---

## Important Notes

- **Player IDs must be consistent** across all datasets (Catapult, injuries, wellness, game stats).
- **Units**: Velocity is in mph, distance is in miles, acceleration is in m/s².
- **Do not modify column names** in source files — the dashboard expects exact matches.
- **Back up data** before making changes to existing CSVs.
