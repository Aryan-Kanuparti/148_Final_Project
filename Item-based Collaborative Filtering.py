import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv(r"C:\Users\yosub\Downloads\limited_dataset.csv")

# Clean the 'date_added' column to handle invalid dates
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')

# Remove rows with invalid 'date_added'
data = data.dropna(subset=['date_added'])

# Count the number of unique playlists
num_playlists = data['playlist_name'].nunique()
print(f"Total number of playlists: {num_playlists}")

# Sort the data by playlist and date_added
data = data.sort_values(by=['playlist_name', 'date_added'])

# Create a list to store the training data (after removing the 3 most recent songs)
train_data = []
removed_songs = []

# Iterate over playlists and remove the last 3 songs
for playlist, group in data.groupby('playlist_name'):
    if len(group) > 3:
        # Remove the 3 most recent songs
        removed = group.tail(3)
        train = group.head(len(group) - 3)

        removed_songs.append(removed)
        train_data.append(train)

# Concatenate training data back into a single DataFrame
train_data = pd.concat(train_data)

# Concatenate removed songs into a single DataFrame
removed_songs = pd.concat(removed_songs)

# Create a song-playlist interaction matrix (binary: 1 if the song is in the playlist, 0 otherwise)
song_playlist_matrix = pd.pivot_table(train_data, index='track_name', columns='playlist_name', aggfunc='size',
                                      fill_value=0)

# Compute cosine similarity between songs based on the interaction matrix
song_similarity_matrix = cosine_similarity(song_playlist_matrix)

# Convert similarity matrix to a DataFrame for easier handling
song_similarity_df = pd.DataFrame(song_similarity_matrix, index=song_playlist_matrix.index,
                                  columns=song_playlist_matrix.index)

# Clean song names by stripping extra spaces and handling encoding
song_similarity_df.index = song_similarity_df.index.str.strip()
song_similarity_df.columns = song_similarity_df.columns.str.strip()


# Function to compute the cosine similarity between two songs
def compute_similarity(song_a, song_b, song_similarity_df):
    if song_a in song_similarity_df.index and song_b in song_similarity_df.index:
        return song_similarity_df.loc[song_a, song_b]
    else:
        return 0  # Return 0 if the song is not found in the similarity matrix


# Function to generate recommendations for each playlist
def get_recommendations(playlist_songs, song_similarity_df, top_n=1):
    recommended_songs = {}
    for song in playlist_songs:
        if song in song_similarity_df.index:
            similar_songs = song_similarity_df[song].sort_values(ascending=False)[
                            1:top_n + 1]  # Exclude the song itself
            recommended_songs[song] = similar_songs.index.tolist()
        else:
            recommended_songs[song] = []  # If song not found, return empty recommendation
    return recommended_songs


# Function to calculate RMSE
def calculate_rmse(recommended_songs, removed_songs, song_similarity_df):
    all_errors = []
    for _, removed_group in removed_songs.groupby('playlist_name'):
        playlist_songs = removed_group['track_name'].tolist()

        for song in playlist_songs:
            recommended_song = recommended_songs.get(song, None)
            if recommended_song:
                # Calculate the similarity between the recommended song and the removed song(s)
                distances = [compute_similarity(song, recommended, song_similarity_df) for recommended in
                             recommended_song]
                all_errors.append(min(distances))  # Use the minimum distance as the error

    # Calculate RMSE from the distances (convert distances to errors)
    rmse = sqrt(
        mean_squared_error(np.ones(len(all_errors)), all_errors))  # Using ones as true values (ideal similarity)
    return rmse


# Generate recommendations for all playlists
all_recommended_songs = {}

# Iterate over playlists and generate recommendations for each
for playlist, group in data.groupby('playlist_name'):
    playlist_songs = group['track_name'].tolist()
    all_recommended_songs[playlist] = get_recommendations(playlist_songs, song_similarity_df, top_n=1)

# Calculate RMSE
rmse_value = calculate_rmse(all_recommended_songs, removed_songs, song_similarity_df)
print(f"Root Mean Squared Error (RMSE): {rmse_value}")
