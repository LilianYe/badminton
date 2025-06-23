# Badminton ELO Rating System

A comprehensive system for tracking, analyzing, and visualizing badminton match results, player ELO ratings, and partnership statistics. Includes a Streamlit web app for interactive exploration and a FastAPI backend for programmatic access.

## Features
- Player ELO ranking and progression tracking
- Session and match history analysis
- Head-to-head statistics between any two players
- Partnership win rate analysis
- Manual match entry and new player management
- REST API for player ratings and match history (FastAPI)
- Mobile-friendly UI

## Project Structure
```
badminton/
├── streamlit_app.py         # Main Streamlit web app
├── api.py                  # FastAPI backend for API access
├── elo_system.py           # ELO calculation and match processing logic
├── data_io.py              # Data loading and saving utilities
├── scheduler.py            # Match and rest schedule generation
├── validators.py           # Schedule and data validation functions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Multi-service orchestration
├── data/                   # Match history, player stats, and session files
└── ...
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

### 3. Run the FastAPI API
```bash
uvicorn api:app --reload
```

- Streamlit UI: http://localhost:8501
- API docs: http://localhost:8000/docs

### 4. Docker Compose (Recommended for production)
```bash
docker-compose up --build
```
- Streamlit: http://localhost:8501
- API: http://localhost:8000

## API Endpoints
- `/players` — List all players and ELO ratings
- `/players/{player_name}` — Get a specific player's ELO and stats
- `/matches/{player_name}` — Get match history for a player
- `/head-to-head/{player1}/{player2}` — Get head-to-head stats

## Data Files
- `data/player_elo_ratings_new.csv` — Player ELO and stats
- `data/match_history_with_elo_*.csv` — Match history per session
- `data/session_stats_*.csv` — Session stats

## Customization
- Add new players and matches via the Streamlit UI
- Hide players from rankings by editing the `HIDDEN_PLAYERS` list
- Adjust ELO calculation logic in `elo_system.py`

## License
MIT License

## Authors
- Your Name (and contributors)
