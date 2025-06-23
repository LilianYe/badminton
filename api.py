from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import uvicorn
from pydantic import BaseModel
import pandas as pd
from data_io import (load_player_data, get_player_match_history, 
                    get_head_to_head_history)

app = FastAPI(
    title="Badminton ELO Rating API",
    description="API for accessing badminton player ratings and match history",
    version="1.0.0"
)

class PlayerRating(BaseModel):
    player: str
    elo: float
    total_games: int
    total_wins: int
    win_rate: str
    
class MatchResult(BaseModel):
    session: str
    team_a: str
    team_b: str
    score: str
    winner: str
    elo_change_a: float
    elo_change_b: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Badminton ELO Rating API"}


@app.get("/players/{player_name}", response_model=PlayerRating)
def get_player(player_name: str):
    """Get a specific player's ELO rating"""
    player_df = load_player_data()
    if player_df is None:
        raise HTTPException(status_code=404, detail="No player data found")
    
    player_data = player_df[player_df["Player"] == player_name]
    if player_data.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    
    row = player_data.iloc[0]
    return PlayerRating(
        player=row["Player"],
        elo=row["ELO"],
        total_games=row["total_games"],
        total_wins=row["total_wins"],
        win_rate=row["total_success_rate"]
    )

@app.get("/matches/{player_name}", response_model=List[MatchResult])
def get_matches(player_name: str, limit: Optional[int] = Query(None, description="Maximum number of matches to return")):
    """Get match history for a specific player"""
    matches = get_player_match_history(player_name)
    if matches is None or matches.empty:
        raise HTTPException(status_code=404, detail=f"No match history found for player '{player_name}'")
    result = []
    
    # Apply limit if specified
    if limit:
        matches = matches.head(limit)
    
    for _, row in matches.iterrows():
        result.append(MatchResult(
            session=row["Session"],
            team_a=row["Team A"],
            team_b=row["Team B"],
            score=row["Score"],
            winner=row["Winner"],
            elo_change_a=row.get("ELO Change A", 0.0),
            elo_change_b=row.get("ELO Change B", 0.0)
        ))
    return result

@app.get("/head-to-head/{player1}/{player2}")
def get_head_to_head(player1: str, player2: str):
    """Get head-to-head statistics between two players"""
    h2h_data = get_head_to_head_history(player1, player2)
    if not h2h_data or h2h_data["total_matches"] == 0:
        raise HTTPException(status_code=404, detail=f"No matches found between '{player1}' and '{player2}'")
    
    # Convert DataFrame to list of dicts for JSON serialization
    matches_list = []
    for _, row in h2h_data["matches"].iterrows():
        matches_list.append({
            "session": row["Session"],
            "team_a": row["Team A"],
            "team_b": row["Team B"],
            "score": row["Score"],
            "winner": row["Winner"]
        })
    
    return {
        "player1": player1,
        "player2": player2,
        "total_matches": h2h_data["total_matches"],
        "player1_wins": h2h_data["player1_wins"],
        "player2_wins": h2h_data["player2_wins"],
        "matches": matches_list
    }

# Run the API with: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)