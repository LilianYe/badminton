import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from elo_system import calculate_elo_ratings, load_existing_player_data, process_match, create_stats_dataframe
import re
from datetime import datetime
import matplotlib as mpl
import logging
import plotly.graph_objects as go
from collections import defaultdict
from data_io import load_player_data, load_session_stats_files, get_head_to_head_history, get_player_match_history, get_player_opponents, save_player_data, save_match_history, save_session_stats, get_partnership_statistics

HIDDEN_PLAYERS = ['疏朗(F)']

st.set_page_config(
    page_title="Badminton ELO Rating System",
    layout="centered",  # Better for mobile
    initial_sidebar_state="collapsed"  # Prevents sidebar from taking space on mobile
)


def configure_matplotlib_fonts():
    """Configure matplotlib to use fonts that support CJK characters"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('badminton-app')
    
    logger.info("Configuring fonts for CJK support")
    
    try:
        # Get list of system fonts
        import matplotlib.font_manager as fm
        
        # List of font directories to check
        font_dirs = []
        
        # Add Docker-specific font paths
        if os.path.exists('/usr/share/fonts/truetype/noto'):
            font_dirs.append('/usr/share/fonts/truetype/noto')
        
        # Find all system fonts + fonts in our specific directories
        if font_dirs:
            # Find font files in our specified directories
            font_files = fm.findSystemFonts(fontpaths=font_dirs)
            
            # Register each font file with matplotlib
            for font_file in font_files:
                if 'cjk' in font_file.lower() or 'noto' in font_file.lower():
                    try:
                        fm.fontManager.addfont(font_file)
                        logger.info(f"Added font: {os.path.basename(font_file)}")
                    except Exception as e:
                        logger.warning(f"Failed to add font {os.path.basename(font_file)}: {e}")
        
        # Get available fonts after registration
        available_fonts = sorted([f.name for f in fm.fontManager.ttflist])
        logger.info(f"Total fonts available: {len(available_fonts)}")
        logger.info(f"Sample fonts: {available_fonts[:10]}")
        
        # Check for CJK font names in available fonts
        cjk_patterns = ['cjk', 'noto sans', 'noto serif', 'simhei', 'simsun', 'microsoft yahei']
        cjk_fonts = [f for f in available_fonts if any(pattern in f.lower() for pattern in cjk_patterns)]
        
        logger.info(f"Found {len(cjk_fonts)} potential CJK fonts: {cjk_fonts[:5]}")
        
        # Set font configuration
        mpl.rcParams['font.family'] = 'sans-serif'
        
        # Add CJK fonts to the beginning of the sans-serif list if found
        if cjk_fonts:
            mpl.rcParams['font.sans-serif'] = cjk_fonts[:3] + ['DejaVu Sans', 'Arial', 'sans-serif']
            logger.info(f"Using CJK fonts: {cjk_fonts[:3]}")
        else:
            mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            logger.warning("No CJK fonts found")
            
        # Fix minus sign display
        mpl.rcParams['axes.unicode_minus'] = False
        
        logger.info("Font configuration completed")
    except Exception as e:
        logger.error(f"Error configuring fonts: {e}")
        st.warning("No font with CJK support detected. Some characters may not display correctly.")


def main():
    global HIDDEN_PLAYERS
    st.title("🏸 Badminton ELO Rating System")
    # Add a new tab for Best Partnerships
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Player Rankings", "Session Results", "Head-to-Head", 
                                                "Best Partnerships", "Add Single Match", "Process New Results"])

    with tab1:
        # Load player data
        player_df = load_player_data()
        
        if player_df is not None:
            st.success(f"Loaded player data with {len(player_df)} players.")
            
            # Filter out players who opted to hide their rankings
            public_players = player_df[~player_df["Player"].isin(HIDDEN_PLAYERS)].copy()
                        
            # Select view mode
            view_mode = st.radio("Select view mode:", ["Summary View", "Detailed Player View"])
            
            if view_mode == "Summary View":
                # Display ELO rankings and stats for visible players only
                st.subheader("📊 Player Rankings and Statistics")
                
                # Always sort by ELO Rating
                public_players = public_players.sort_values(by="ELO", ascending=False)
                
                # Format columns for better display
                display_df = public_players[['Player', 'ELO', 'total_games', 'total_wins', 'total_success_rate']]
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
                    linewidth=1,
                    label='ELO Distribution'  # Add this label
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
                # Use public_players instead of player_df to filter out hidden players
                sorted_players = public_players.sort_values(by="ELO", ascending=False)
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
                            
                            # Sort by session date ascending
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
                            
                            # Get match dates for better x-axis labels
                            match_dates = elo_progress["Session"].values
                            # Ensure unique x-axis values by adding match number to duplicate dates
                            x_labels = []
                            date_counts = {}
                            for date in reversed(match_dates):
                                if date in date_counts:
                                    date_counts[date] -= 1
                                else:
                                    date_counts[date] = len([d for d in match_dates if d == date])
                                x_labels.insert(0, f"{date}-{date_counts[date]}")
                            # Add the initial point label
                            x_labels.insert(0, "Start")
                            
                            mobile_view = True
                            # Create a Plotly figure
                            fig = go.Figure()
                            
                            # Add line trace with markers
                            fig.add_trace(go.Scatter(
                                x=list(range(len(elo_values))),
                                y=elo_values,
                                mode='lines+markers+text' if not mobile_view else 'lines+markers',
                                marker=dict(size=8 if mobile_view else 10, color='steelblue'),
                                line=dict(width=2, color='steelblue'),
                                text=[f'{elo:.0f}' for elo in elo_values],
                                textposition='top center',
                                textfont=dict(size=10, color='black', family='Arial, sans-serif'),
                                hoverinfo='text',
                                hovertext=[f'Match: {x_labels[i]}<br>ELO: {elo:.0f}' for i, elo in enumerate(elo_values)],
                                name='ELO Rating'
                            ))
                            
                            # Add a horizontal line at starting ELO (1500) for reference
                            fig.add_shape(
                                type='line',
                                x0=0,
                                y0=1500,
                                x1=len(elo_values)-1,
                                y1=1500,
                                line=dict(color='gray', dash='dash', width=1)
                            )
                            
                            # Update layout for better appearance
                            fig.update_layout(
                                xaxis=dict(
                                    title='Matches',
                                    tickmode='array',
                                    tickvals=list(range(len(elo_values))) if len(elo_values) < 8 or not mobile_view else list(range(0, len(elo_values), max(1, len(elo_values) // 5))),
                                    ticktext=[str(i) for i in range(len(elo_values))] if len(elo_values) < 8 or not mobile_view else [str(i) for i in range(0, len(elo_values), max(1, len(elo_values) // 5))],
                                    tickangle=45 if mobile_view else 0  # Angle ticks on mobile
                                ),
                                yaxis=dict(
                                    title=None if mobile_view else 'ELO Rating',
                                    range=[min(elo_values) - 20, max(elo_values) + 20]
                                ),
                                showlegend=False,
                                hovermode='closest',
                                plot_bgcolor='white',
                                margin=dict(l=10 if mobile_view else 40, 
                                            r=10 if mobile_view else 20, 
                                            t=30,  
                                            b=40 if mobile_view else 50),
                            )
                            
                            # Add grid lines
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                            
                            # Enable panning, zooming and other interactive features
                            fig.update_layout(
                                dragmode='pan',  # Enable panning by default
                            )
                            height = None if not mobile_view else 400
                            # Display the interactive plot
                            st.plotly_chart(fig, use_container_width=True, height=height)
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
                    
                    # Filter out hidden players
                    public_session_df = session_df[~session_df["Player"].isin(HIDDEN_PLAYERS)].copy()    
                    # Display simple header with date
                    st.subheader(f"Results for {selected_date}")
                    
                    # Create a display dataframe with the columns we want, sorted by ELO change
                    public_session_df = public_session_df.rename(columns={
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
                        public_session_df,
                        column_config={
                            "ELO Change": st.column_config.NumberColumn(format="%.1f")
                        },
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"Error loading session data: {e}")
    
    with tab6:
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
                            sheet_name
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
    
    with tab3:
        st.subheader("🤺 Head-to-Head Comparison")
        # Load player data for the dropdowns
        player_df = load_player_data()
        if player_df is not None:
            # Create two columns for selecting players
            col1, col2 = st.columns(2)
            
            with col1:
                player1 = st.selectbox(
                    "Select first player:",
                    options=sorted(player_df["Player"].tolist()),
                    key="player1_select"
                )
                
            with col2:
                # Filter options to only players who have played against player1
                if player1:
                    player2_options = get_player_opponents(player1)
                    if not player2_options:
                        st.info(f"No opponents found for {player1}.")
                        player2 = None
                    else:
                        player2 = st.selectbox(
                            "Select second player:",
                            options=player2_options,
                            key="player2_select"
                        )
                else:
                    player2 = None
                    st.info("Please select a player first.")
            
            if player1 and player2:
                if st.button("Show Head-to-Head Stats"):
                    # Get head-to-head history
                    h2h_data = get_head_to_head_history(player1, player2)
                    
                    if h2h_data and h2h_data["total_matches"] > 0:
                        # Display summary statistics
                        st.subheader(f"{player1} vs {player2}")
                        
                        # Create metrics for easy viewing
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("Total Matches", h2h_data["total_matches"])
                        with metric_cols[1]:
                            st.metric(f"{player1} Wins", h2h_data["player1_wins"])
                        with metric_cols[2]:
                            st.metric(f"{player2} Wins", h2h_data["player2_wins"])
                        
                        # Display the actual match history
                        st.subheader("Match History")
                        
                        # Select and rename columns for better display
                        display_cols = ["Session", "Team A", "Team B", "Score", "Winner"]
                        matches_display = h2h_data["matches"][display_cols].copy()
                        
                        # Format the display to highlight which player was on which team
                        matches_display["Result"] = "Draw"
                        for idx, row in matches_display.iterrows():
                            if row["Winner"] == "Team A":
                                if row["Team A"].find(player1) >= 0:
                                    matches_display.at[idx, "Result"] = f"{player1} Won"
                                else:
                                    matches_display.at[idx, "Result"] = f"{player2} Won"
                            elif row["Winner"] == "Team B":
                                if row["Team B"].find(player1) >= 0:
                                    matches_display.at[idx, "Result"] = f"{player1} Won"
                                else:
                                    matches_display.at[idx, "Result"] = f"{player2} Won"
                        
                        # Display the match history
                        st.dataframe(
                            matches_display[["Session", "Team A", "Team B", "Score", "Result"]],
                            column_config={
                                "Session": st.column_config.TextColumn(
                                    "Date",
                                    help="Date when the match was played"
                            )
                        },
                        hide_index=True
                    )
                    
                    # Check if we have enough data for a visualization
                    if h2h_data and h2h_data["total_matches"] > 1:
                        st.subheader("Head-to-Head Summary")
                        
                        # Create a pie chart showing win distribution
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels = [f"{player1} Wins", f"{player2} Wins"]
                        sizes = [h2h_data["player1_wins"], h2h_data["player2_wins"]]
                        
                        # Only include non-zero values
                        non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
                        non_zero_sizes = [size for size in sizes if size > 0]
                        
                        if non_zero_sizes:
                            # Create the pie chart
                            ax.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%',
                                   shadow=False, startangle=90)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            st.pyplot(fig)
        else:
            st.error("Could not load player data. Please check if player data files exist.")

    with tab4:
        st.subheader("🤝 Best Partnerships")
        st.write("Discover which player combinations have the highest win rates.")
        
        # Get partnership statistics
        partnership_df = get_partnership_statistics()
        
        if partnership_df is not None:
            # Filter options
            min_games = st.slider("Minimum Games Played", 3, 20, 3)
            
            # Filter partnerships
            filtered_df = partnership_df[partnership_df["Games Played"] >= min_games].copy()
            
            if not filtered_df.empty:
                # Sort by win rate (descending) and then by games played (descending)
                filtered_df = filtered_df.sort_values(
                    by=["Win Rate Value", "Games Played"], 
                    ascending=[False, False]
                )
                
                # Remove the sorting column before display
                display_df = filtered_df.drop(columns=["Win Rate Value"]).head(20)
                
                # Display the table
                st.dataframe(
                    display_df,
                    column_config={
                        "Partnership": st.column_config.TextColumn("Partnership"),
                        "Games Played": st.column_config.NumberColumn("Games", help="Total games played by this partnership"),
                        "Wins": st.column_config.NumberColumn("Wins", help="Total wins by this partnership"),
                        "Win Rate": st.column_config.TextColumn("Win Rate", help="Percentage of games won")
                    },
                    hide_index=True
                )    
            else:
                st.info(f"No partnerships found with at least {min_games} games played.")
        else:
            st.info("No partnership data available. Please play some matches first!")

    # Add the new tab for manual match entry
    with tab5:
        st.subheader("✏️ Add Single Match Result")
        
        # Load player data for selection
        player_df = load_player_data()
        player_list = []
        
        if player_df is not None:
            player_list = sorted(player_df["Player"].tolist())
            
            # Step 1: Add new players if needed
            st.subheader("Step 1: Add New Players")
            st.write("Add any new players before entering match results.")
            
            # Create two columns for adding new players
            col1, col2 = st.columns(2)
            
            with col1:
                new_player_name = st.text_input("New Player Name", key="new_player_name")
            
            with col2:
                new_player_gender = st.selectbox("Gender", ["Male", "Female"], key="new_player_gender")
                
            # Button to add the player
            if st.button("Add Player"):
                if not new_player_name:
                    st.error("Please enter a player name.")
                else:
                    # Add (F) suffix if female
                    player_full_name = f"{new_player_name}(F)" if new_player_gender == "Female" else new_player_name
                    
                    # Check if player already exists
                    if player_full_name in player_list:
                        st.error(f"Player '{player_full_name}' already exists.")
                    else:
                        # Add default entry to player ELO ratings file
                        try:
                            from elo_system import load_existing_player_data, DEFAULT_ELO
                            from data_io import add_new_player
                            add_new_player(player_full_name, DEFAULT_ELO)
                            # After successful addition, set a flag to clear the form
                            st.session_state["player_added"] = True
                            st.success(f"Player '{player_full_name}' added successfully!")
                            # Reload player list to include the new player
                            updated_df = load_player_data()
                            if updated_df is not None:
                                player_list = sorted(updated_df["Player"].tolist())
                        except Exception as e:
                            st.error(f"Error adding new player: {e}")
        
        # Display current player list
        with st.expander("Current Player List"):
            if player_list:
                st.write(", ".join(player_list))
            else:
                st.write("No players found.")
        
        # Step 2: Enter match details
        st.subheader("Step 2: Enter Match Details")
        
        # Create form for match entry
        with st.form("match_form"):
            st.write("### Team A")
            col1a, col2a = st.columns(2)
            with col1a:
                a1_player = st.selectbox("Player 1", player_list, key="a1_select")
            with col2a:
                a2_player = st.selectbox("Player 2", player_list, key="a2_select", index=1 if len(player_list) > 1 else 0)
            
            st.write("### Team B")
            col1b, col2b = st.columns(2)
            with col1b:
                b1_player = st.selectbox("Player 1", player_list, key="b1_select", index=2 if len(player_list) > 2 else 0)
            with col2b:
                b2_player = st.selectbox("Player 2", player_list, key="b2_select", index=3 if len(player_list) > 3 else 0)
            
            # Match details
            st.write("### Match Details")
            col1c, col2c = st.columns(2)
            with col1c:
                st.write("First score is always Team A's score")
                score = st.text_input("Score (e.g., 21-19 means Team A won)", "21-0")
            
            # Submit button
            submit_button = st.form_submit_button("Add Match")
        
        # Process form submission
        if submit_button:
            # Validate inputs
            error = None
            
            # Check for duplicate players
            players = [a1_player, a2_player, b1_player, b2_player]
            if len(set(players)) < 4:
                error = "All players must be different"
            
            # Validate score format
            if not re.match(r'^\d+-\d+$', score):
                error = "Score must be in format: 21-19"
            
            if error:
                st.error(error)
            else:
                # Process the match
                team_a = [a1_player, a2_player]
                team_b = [b1_player, b2_player]
                
                success, message = process_single_match(team_a, team_b, score)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)


def process_single_match(team_a, team_b, score):
    """Process a single match entered manually."""
    try:
        # Load existing player data using the function from elo_system
        player_elos, player_stats = load_existing_player_data()
        
        # Store initial ELOs to calculate changes after the match
        initial_elos = dict(player_elos)
        
        # Initialize session stats tracker (needed for process_match)
        session_stats = defaultdict(lambda: {
            "total": {"games": 0, "wins": 0},
            "male_double": {"games": 0, "wins": 0},
            "female_double": {"games": 0, "wins": 0},
            "mixed": {"games": 0, "wins": 0}
        })
        
        # Format the match data to match what process_match expects
        team_a_str = f"{team_a[0]}/{team_a[1]}"
        team_b_str = f"{team_b[0]}/{team_b[1]}"
        match_str = f"{team_a_str} vs {team_b_str}"
        
        # Format score to match what process_match expects
        score_parts = score.split("-")
        if len(score_parts) != 2:
            return False, "Score must be in format: 21-19"
            
        score_str = f"{score_parts[0]}:{score_parts[1]}"
        
        # Court is arbitrary for manual entry
        court = "Manual"
        
        # Process the match using the existing function from elo_system
        match_record = process_match(
            court, match_str, score_str,
            player_elos, player_stats, session_stats
        )
        
        if not match_record:
            return False, "Failed to process match. Please check input data."
            
        # Create a match history dataframe with this one match
        match_history = [match_record]
        history_df = pd.DataFrame(match_history)
        save_match_history(history_df)
            
        # Create the player ratings dataframe
        elo_df = pd.DataFrame(player_elos.items(), columns=['Player', 'ELO'])
        
        # Create the player statistics dataframe
        stats_df = create_stats_dataframe(player_stats)
        
        # Merge player stats with ELO ratings
        combined_df = pd.merge(elo_df, stats_df, on="Player")
        save_player_data(combined_df)

        # Calculate ELO changes for this session
        elo_changes = {}
        for player, final_elo in player_elos.items():
            # If player is new, initial ELO was DEFAULT_ELO
            initial_elo = initial_elos.get(player, 1500)
            elo_changes[player] = round(final_elo - initial_elo, 1)
        
        # Create and save session stats
        session_df = create_stats_dataframe(session_stats)
        session_df['elo_change'] = session_df['Player'].map(elo_changes)
        save_session_stats(session_df)        
        return True, "Match processed successfully. Player ratings have been updated."
        
    except Exception as e:
        logging.error(f"Error processing single match: {e}")
        return False, f"Error processing match: {e}"


if __name__ == "__main__":
    configure_matplotlib_fonts()  # Configure fonts at the start
    main()
