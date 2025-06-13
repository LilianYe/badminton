import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from elo_system import calculate_elo_ratings
from glob import glob
import re
from datetime import datetime

# Create a platform-independent data directory path
def get_data_dir():
    """Get the appropriate data directory based on environment"""
    # For Docker deployment
    docker_path = "/app/data"
    # For Windows development
    windows_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Check if Docker path exists
    if os.path.exists(docker_path):
        return docker_path
    # Fall back to local development path
    else:
        # Create the data directory if it doesn't exist
        if not os.path.exists(windows_path):
            os.makedirs(windows_path)
        return windows_path

# Get the data directory to use
DATA_DIR = get_data_dir()

st.set_page_config(
    page_title="Badminton ELO Rating System",
    layout="centered",  # Better for mobile
    initial_sidebar_state="collapsed"  # Prevents sidebar from taking space on mobile
)

def load_match_history_files():
    """Load all match history files in the directory"""
    # Use os.path.join for cross-platform compatibility
    pattern = os.path.join(DATA_DIR, "match_history_with_elo_*.csv")
    files = glob(pattern)
    return sorted(files)

def load_player_data():
    """Load player ELO ratings and statistics"""
    try:
        # Use os.path.join for path creation
        new_ratings_path = os.path.join(DATA_DIR, "player_elo_ratings_new.csv")
        old_ratings_path = os.path.join(DATA_DIR, "player_elo_ratings.csv")
        
        # Try to read the updated ratings file
        if os.path.exists(new_ratings_path):
            df = pd.read_csv(new_ratings_path)
            file_used = "player_elo_ratings_new.csv"
        else:
            # Fallback to original ratings file
            df = pd.read_csv(old_ratings_path)
            file_used = "player_elo_ratings.csv"
        return df, file_used
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return None, None

def get_player_match_history(player):
    """Get match history for a specific player"""
    match_files = load_match_history_files()
    
    if not match_files:
        return None
    
    all_matches = []
    for file in match_files:
        try:
            match_df = pd.read_csv(file)
            filename = os.path.basename(file)
            match_df["Session"] = filename.replace("match_history_with_elo_", "").replace(".csv", "")
            # Escape special regex characters in player name and ensure exact player name matching
            # by searching for the player surrounded by non-word characters or at start/end of string
            player_escaped = re.escape(player)
            pattern = f"(?:^|[^\\w]){player_escaped}(?:[^\\w]|$)"  # Using non-capturing groups
            
            # Filter for matches involving this player using regex
            player_matches = match_df[
                (match_df["Team A"].str.contains(pattern, regex=True)) | 
                (match_df["Team B"].str.contains(pattern, regex=True))
            ].copy()
            
            # If regex approach fails (no matches), fall back to simple contains
            if len(player_matches) == 0:
                player_matches = match_df[
                    (match_df["Team A"].str.contains(player, regex=False)) | 
                    (match_df["Team B"].str.contains(player, regex=False))
                ].copy()
            
            # Add match index within the file to maintain original order
            player_matches["Match_Index"] = range(len(player_matches))

            # Add columns to identify which team the player was on and if they won
            player_matches["Player's Team"] = "A"  # Default to team A

            # For the regex approach
            if 'pattern' in locals():
                # Use the same regex pattern that was used for filtering
                matches_in_team_a = player_matches["Team A"].str.contains(pattern, regex=True)
                player_matches.loc[~matches_in_team_a, "Player's Team"] = "B"
            else:
                # Fallback to literal string matching (no regex)
                matches_in_team_a = player_matches["Team A"].str.contains(player, regex=False)
                player_matches.loc[~matches_in_team_a, "Player's Team"] = "B"
            
            player_matches["Result"] = "Won"
            player_matches.loc[
                ((player_matches["Player's Team"] == "A") & (player_matches["Winner"] != "Team A")) |
                ((player_matches["Player's Team"] == "B") & (player_matches["Winner"] != "Team B")),
                "Result"
            ] = "Lost"
            
            # Reset the index to avoid duplicate indices when concatenating
            player_matches = player_matches.reset_index(drop=True)
            
            all_matches.append(player_matches)
        except Exception as e:
            st.warning(f"Could not process file {file}: {e}")
            continue
    
    if all_matches:
        # Concatenate and reset index again to ensure unique indices
        combined_matches = pd.concat(all_matches, ignore_index=True)
        return combined_matches.sort_values(by=["Session", "Match_Index"], ascending=[False, False])
    return None

def load_session_stats_files():
    """Load all session stats files in the data directory"""
    # Use os.path.join for cross-platform compatibility
    pattern = os.path.join(DATA_DIR, "session_stats_*.csv")
    files = glob(pattern)
    return sorted(files, reverse=True)  # Most recent first

def is_mobile():
    """Detect if user is on a mobile device based on screen width"""
    try:
        # Use browser's window dimensions through iframe communication
        screen_width = st.session_state.get("screen_width", 1000)  # Default to desktop size
        print(screen_width)
        return screen_width < 768  # Common breakpoint for mobile devices
    except:
        return False  # Default to desktop view if detection fails

def main():
    st.title("🏸 Badminton ELO Rating System")
    # Add tab for session results
    tab1, tab2, tab3 = st.tabs(["Player Rankings", "Session Results", "Process New Results"])

    with tab1:
        # Load player data
        player_df, file_used = load_player_data()
        
        if player_df is not None:
            st.success(f"Loaded player data from {file_used}")
            
            # Select view mode
            view_mode = st.radio("Select view mode:", ["Summary View", "Detailed Player View"])
            
            if view_mode == "Summary View":
                # Display ELO rankings and stats
                st.subheader("📊 Player Rankings and Statistics")
                
                # Always sort by ELO Rating
                player_df = player_df.sort_values(by="ELO", ascending=False)
                
                # Format columns for better display
                display_df = player_df[['Player', 'ELO', 'total_games', 'total_wins', 'total_success_rate']]
                display_df = display_df.rename(columns={
                    'total_games': 'Games Played',
                    'total_wins': 'Wins',
                    'total_success_rate': 'Win Rate'
                })
                
                # Display as a dataframe with clickable rows
                st.dataframe(
                    display_df,
                    column_config={
                        "ELO": st.column_config.NumberColumn(format="%.1f")
                    },
                    hide_index=True
                )
                
                # Create distribution chart
                st.subheader("ELO Distribution")

                # Create figure with fixed size
                fig, ax = plt.subplots(figsize=(10, 6))

                # Calculate key statistics
                median_elo = player_df['ELO'].median()
                min_elo = player_df['ELO'].min()
                max_elo = player_df['ELO'].max()

                # Create histogram with better styling
                bin_start = int(min_elo // 25) * 25
                bin_end = int(max_elo // 25 + 1) * 25
                bins = range(bin_start, bin_end + 1, 25)
                n, bins, patches = ax.hist(
                    player_df['ELO'], 
                    bins=bins, 
                    alpha=0.7, 
                    color='#4c72b0',
                    edgecolor='white', 
                    linewidth=1
                )
                # Improve styling
                ax.set_xlabel('ELO Rating', fontsize=12)
                ax.set_ylabel('Number of Players', fontsize=12)
                ax.grid(True, alpha=0.2, linestyle=':')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=10)
                # Add legend
                ax.legend(frameon=False)

                # Add text annotations for statistics
                stats_text = (
                    f"Total Players: {len(player_df)}\n"
                    f"Median: {median_elo:.1f}\n"
                    f"Range: {min_elo:.1f} - {max_elo:.1f}"
                )
                # Position in the upper right if most values are on the left side
                x_pos = 0.95
                ax.text(
                    x_pos, 0.95, stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right' if x_pos > 0.5 else 'left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

                # Ensure proper spacing
                plt.tight_layout()

                # Display the plot
                st.pyplot(fig)
                
            else:  # Detailed Player View
                st.subheader("🔍 Player Detail View")
                
                # Create selectbox with players sorted by ELO
                sorted_players = player_df.sort_values(by="ELO", ascending=False)
                selected_player = st.selectbox(
                    "Select a player to view details:",
                    sorted_players["Player"].tolist()
                )
                
                if selected_player:
                    # Get player details
                    player_info = player_df[player_df["Player"] == selected_player].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ELO Rating", f"{player_info['ELO']:.1f}")
                    with col2:
                        st.metric("Win Rate", player_info["total_success_rate"])
                    with col3:
                        st.metric("Games Played", player_info["total_games"])
                    
                    # Create detailed stats section
                    st.subheader("Game Type Statistics")
                    
                    # Create columns for different game types
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    # Total stats
                    with stat_col1:
                        st.write("**Overall Stats**")
                        st.write(f"Games: {player_info['total_games']}")
                        st.write(f"Wins: {player_info['total_wins']}")
                        st.write(f"Win Rate: {player_info['total_success_rate']}")
                    
                    # Same gender stats
                    with stat_col2:
                        st.write("**Same Gender Doubles**")
                        st.write(f"Games: {player_info['same_gender_games']}")
                        st.write(f"Wins: {player_info['same_gender_wins']}")
                        st.write(f"Win Rate: {player_info['same_gender_success_rate']}")
                    
                    # Mixed stats
                    with stat_col3:
                        st.write("**Mixed Doubles**")
                        st.write(f"Games: {player_info['mixed_games']}")
                        st.write(f"Wins: {player_info['mixed_wins']}")
                        st.write(f"Win Rate: {player_info['mixed_success_rate']}")
                    
                    # Get match history for this player
                    st.subheader("Match History")
                    matches = get_player_match_history(selected_player)
                    
                    if matches is not None and not matches.empty:
                        # Add ELO change column for selected player
                        # Initialize as float to avoid dtype incompatibility warning
                        matches["Player's ELO Change"] = 0.0  
                        
                        # Calculate ELO change based on which team the player was on
                        matches.loc[matches["Player's Team"] == "A", "Player's ELO Change"] = matches["ELO Change A"]
                        matches.loc[matches["Player's Team"] == "B", "Player's ELO Change"] = matches["ELO Change B"]
                        
                        # Display matches with nice formatting
                        display_cols = [
                            "Session", "Team A", "Team B", "Score", 
                            "Result", "Player's ELO Change"
                        ]
                        
                        st.dataframe(
                            matches[display_cols],
                            column_config={
                                "Session": st.column_config.TextColumn(
                                    "Date",
                                    help="Date when the match was played"
                                ),
                                "Player's ELO Change": st.column_config.NumberColumn(
                                    format="%.1f",
                                    help="ELO points gained or lost in this match"
                                ),
                                "Result": st.column_config.Column(
                                    help="Whether the selected player won or lost this match"
                                )
                            },
                            hide_index=True
                        )
                        
                        # Add ELO progression chart
                        if len(matches) > 1:
                            st.subheader("ELO Progression")
                            
                            # We'll need to calculate cumulative ELO over time
                            # First, sort by session date ascending
                            elo_progress = matches.sort_values(["Session", "Match_Index"], ascending=[True, True])
                            
                            # Get current ELO and work backwards
                            current_elo = player_info['ELO']
                            elo_changes = elo_progress["Player's ELO Change"].values
                            
                            # Calculate ELO after each match
                            elo_values = [current_elo]
                            running_elo = current_elo
                            
                            for change in reversed(elo_changes):
                                running_elo -= change
                                elo_values.insert(0, running_elo)
                            
                            # Plot the progression
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Plot line with markers
                            ax.plot(range(len(elo_values)), elo_values, marker='o', markersize=8, color='steelblue', linewidth=2)

                            # Add text labels with exact values above each point
                            for i, elo in enumerate(elo_values):
                                ax.annotate(f'{elo:.0f}', 
                                            (i, elo),
                                            textcoords="offset points", 
                                            xytext=(0, 10),  # Offset text 10 points above the point
                                            ha='center',     # Center text horizontally
                                            fontsize=9,      # Smaller font for readability
                                            fontweight='bold')

                            # Improve appearance
                            ax.set_xlabel('Matches (Oldest to Newest)')
                            ax.set_ylabel('ELO Rating')
                            ax.grid(True, alpha=0.3)

                            # Set y-axis limits with some padding
                            y_min = min(elo_values) - 20
                            y_max = max(elo_values) + 20
                            ax.set_ylim(y_min, y_max)

                            # Use integer ticks for x-axis
                            ax.set_xticks(range(len(elo_values)))

                            # Add a horizontal line at starting ELO for reference
                            ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)

                            st.pyplot(fig)
                    else:
                        st.info("No match history found for this player.")
                
    with tab2:
        st.subheader("📅 Session Results")
        
        # Load available session files
        session_files = load_session_stats_files()
        
        if not session_files:
            st.info("No session history files found. Process some match results first.")
        else:
            # Extract dates from filenames for selection
            session_dates = [os.path.basename(f).replace("session_stats_", "").replace(".csv", "") for f in session_files]
            
            # Format dates for display in selectbox
            display_dates = []
            for date_str in session_dates:
                try:
                    # Try to parse and format the date
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    formatted_date = date_obj.strftime("%B %d, %Y")
                    display_dates.append(formatted_date)
                except ValueError:
                    # If parsing fails, use the original string
                    display_dates.append(date_str)
            
            # Create a mapping from display date to file path
            date_to_file = dict(zip(display_dates, session_files))
            
            # Let user select a session
            selected_date = st.selectbox(
                "Select a session date:",
                options=display_dates
            )
            
            if selected_date:
                # Load the selected session data
                session_file = date_to_file[selected_date]
                try:
                    session_df = pd.read_csv(session_file)
                    
                    # Display simple header with date
                    st.subheader(f"Results for {selected_date}")
                    
                    # Create a display dataframe with the columns we want, sorted by ELO change
                    session_df = session_df.rename(columns={
                        'total_games': 'Games Played',
                        'total_wins': 'Wins',
                        'total_success_rate': 'Win Rate',
                        'same_gender_games': 'Same Gender Games',
                        'same_gender_wins': 'Same Gender Wins',
                        'same_gender_success_rate': 'Same Gender Win Rate',
                        'mixed_games': 'Mixed Games',
                        'mixed_wins': 'Mixed Wins',
                        'mixed_success_rate': 'Mixed Win Rate',
                        'elo_change': 'ELO Change'
                    }).sort_values(by="ELO Change", ascending=False)
                    
                    # Add ELO column from player data
                    # Display the table without index
                    st.dataframe(
                        session_df,
                        column_config={
                            "ELO Change": st.column_config.NumberColumn(format="%.1f")
                        },
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"Error loading session data: {e}")
    with tab3:
        st.subheader("Process New Match Results")
        uploaded_file = st.file_uploader(
            "Upload Excel file with match results", 
            type=["xlsx", "xls"]
        )
        
        sheet_name = st.text_input("Sheet name (default: Schedule)", "Schedule")
        
        if uploaded_file is not None:
            if st.button("Process Results"):
                try:
                    # Save the uploaded file
                    temp_file = "temp_upload.xlsx"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the results
                    with st.spinner("Processing results..."):
                        elo_df, history_df, stats_df, session_df = calculate_elo_ratings(
                            temp_file, 
                            sheet_name,
                            DATA_DIR  # Pass the data directory
                        )
                        
                        st.success("Results processed successfully!")
                        
                        # Display a single combined table with all information
                        st.subheader("Session Results and Updated Rankings")
                        
                        # Merge ELO rankings with session statistics
                        # Get only the Player and ELO columns from elo_df
                        elo_info = elo_df[['Player', 'ELO']]
                        
                        # Merge with session statistics
                        combined_df = pd.merge(session_df, elo_info, on='Player')
                        
                        # Select and rename columns for display
                        display_combined = combined_df[['Player', 'ELO', 'total_games', 'total_wins', 'total_success_rate',
                                                       'same_gender_games', 'same_gender_wins', 'same_gender_success_rate',
                                                       'mixed_games', 'mixed_wins', 'mixed_success_rate', 
                                                       'elo_change']]
                        
                        # Rename columns for better readability
                        display_combined = display_combined.rename(columns={
                            'total_games': 'Games Played',
                            'total_wins': 'Wins',
                            'total_success_rate': 'Win Rate',
                            'same_gender_games': 'Same Gender Games',
                            'same_gender_wins': 'Same Gender Wins',
                            'same_gender_success_rate': 'Same Gender Win Rate',
                            'mixed_games': 'Mixed Games',
                            'mixed_wins': 'Mixed Wins',
                            'mixed_success_rate': 'Mixed Win Rate',
                            'elo_change': 'ELO Change'
                        }).sort_values(by="ELO Change", ascending=False)
                        
                        # Display the combined table
                        st.dataframe(
                            display_combined,
                            column_config={
                                "ELO": st.column_config.NumberColumn(format="%.1f"),
                                "ELO Change": st.column_config.NumberColumn(format="%.1f")
                            },
                            hide_index=True
                        )
                        
                        # Clean up temporary file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                
                except Exception as e:
                    st.error(f"Error processing results: {e}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        else:
            st.info("Please upload an Excel file with match results to process.")

if __name__ == "__main__":
    main()
