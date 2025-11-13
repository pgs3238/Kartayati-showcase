import pandas as pd
import numpy as np
from itertools import combinations
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load Data (2927-2930 -> 5001-5006) (5056-5555)
df = pd.read_csv("game_data_10000.csv")

# Filter relevant columns (dots, partnerships, circles, and new points) 52 Features
feature_cols = [
    'Dot ID0', 'X0', 'Y0', 'Dot ID1', 'X1', 'Y1', 'Dot ID2', 'X2', 'Y2',
    'Dot ID3', 'X3', 'Y3', 'Dot ID4', 'X4', 'Y4', 'Dot ID5', 'X5', 'Y5',
    'Dot ID6', 'X6', 'Y6', 'Dot ID7', 'X7', 'Y7',
    'HasPartners0', 'HasPartners1', 'HasPartners2', 'HasPartners3',
    'HasPartners4', 'HasPartners5', 'HasPartners6', 'HasPartners7',
    'Circle1 ID A', 'Circle1 ID B', 'Circle1 X', 'Circle1 Y', 'Circle1 Radius',
    'Circle2 ID A', 'Circle2 ID B', 'Circle2 X', 'Circle2 Y', 'Circle2 Radius',
    'Circle3 ID A', 'Circle3 ID B', 'Circle3 X', 'Circle3 Y', 'Circle3 Radius',
    'Circle4 ID A', 'Circle4 ID B', 'Circle4 X', 'Circle4 Y', 'Circle4 Radius',
    #'NewPoints Player 1', 'NewPoints Player 2'
]
# Takes data and creates new Features for when circles overlap each other (+6 Features)
def calculate_circle_overlap(df):
    for i in range(4):  # Assuming max 4 circles
        for j in range(i + 1, 4):  # Compare each circle with the others
            x1, y1, r1 = df[f'Circle{i+1} X'], df[f'Circle{i+1} Y'], df[f'Circle{i+1} Radius']
            x2, y2, r2 = df[f'Circle{j+1} X'], df[f'Circle{j+1} Y'], df[f'Circle{j+1} Radius']

            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            df[f'Overlap Circle{i+1}-{j+1}'] = (distance <= (r1 + r2)).astype(int)  # 1 if touching, else 0 <--??
            df[f'Overlap Circle{j+1}-in-{i+1}'] = (distance <= (r1 + r2)).astype(int)  # Check if j overlaps i

    return df

df = calculate_circle_overlap(df)

feature_cols.extend([f'Overlap Circle{j+1}-in-{i+1}' for i in range(4) for j in range(i+1, 4)])

# # Print the columns for overlap for testing purposes
# for i in range(4):
#     for j in range(i + 1, 4):
#         col_name = f'Overlap Circle{j+1}-in-{i+1}'
#         print(f"Contents of {col_name}:")
#         print(df[col_name], "\n")

#feature_cols.extend([f'Overlap Circle{i+1}-{j+1}' for i in range(4) for j in range(i+1, 4)])

def count_touching_circles(df):
    for i in range(4):
        overlap_cols = [f'Overlap Circle{i+1}-{j+1}' for j in range(4) if i != j and f'Overlap Circle{i+1}-{j+1}' in df.columns]
        df[f'Touching Count Circle{i+1}'] = df[overlap_cols].sum(axis=1)

    return df

df = count_touching_circles(df)
feature_cols.extend([f'Touching Count Circle{i+1}' for i in range(4)])

# #Print the columns for overlap for testing purposes
# for i in range(4):
#     col_name = f'Touching Count Circle{i+1}'
#     print(f"Contents of {col_name}:")
#     print(df[col_name], "\n")

# def count_touching_circles(df):
#     for i in range(4):
#         overlap_cols = [f'Overlap Circle{j+1}-in-{i+1}' for j in range(4) if i != j and f'Overlap Circle{j+1}-in-{i+1}' in df.columns]
#         df[f'Touching Count Circle{i+1}'] = df[overlap_cols].sum(axis=1)
#
#     return df
#
# df = count_touching_circles(df)
# feature_cols.extend([f'Touching Count Circle{i+1}' for i in range(4)])

df['Previous Points Player 1'] = df['Total Points Player 1'] - df['NewPoints Player 1']
df['Previous Points Player 2'] = df['Total Points Player 2'] - df['NewPoints Player 2']
feature_cols.extend(['Previous Points Player 1', 'Previous Points Player 2'])

# def find_new_circle(df):
#     # Count the number of existing circles (non-null X-coordinates)
#     df['Num Circles'] = df.apply(lambda row: sum(pd.notna(row[col]) for col in ['Circle1 X', 'Circle2 X', 'Circle3 X', 'Circle4 X']), axis=1)
#
#     # Assign the new circle based on count (or mark as 'Game Start' if no circles exist)
#     df['New Circle ID'] = df['Num Circles'].map({
#         1: 'Circle1',
#         2: 'Circle2',
#         3: 'Circle3',
#         4: 'Circle4'
#     }).fillna('Game Start')  # If no circles exist, mark as 'Game Start'
#
#     return df
#
# df = find_new_circle(df)
#
# df['Is New Circle'] = 0
# # Assign 1 only to the last created circle
# df.loc[df['Num Circles'] > 0, 'Is New Circle'] = 1
#
# feature_cols.extend(['Is New Circle'])
#
# # # Drop 'Num Circles' column if no longer needed
# # df.drop(columns=['Num Circles'], inplace=True)
#
# print(df[['Circle1 X', 'Circle2 X', 'Circle3 X', 'Circle4 X', 'New Circle ID', 'Is New Circle']].head())



# Handle missing values (fill with 0)
df.fillna(0, inplace=True)

# Standardize input features
scaler = StandardScaler()
#scaler = MinMaxScaler()
X = scaler.fit_transform(df[feature_cols])

plt.boxplot(X)
plt.show()

# Target Variable (NewPoints for both players) 2 Output Features
#y = df[['NewPoints Player 1', 'NewPoints Player 2', 'Total Points Player 1', 'Total Points Player 2']]
y = df[['NewPoints Player 1', 'NewPoints Player 2']]
#y = df[['Total Points Player 1', 'Total Points Player 2']]

print(y.shape)
print("Y Min Before Scaling:", np.min(y.values, axis=0))
print("Y Max Before Scaling:", np.max(y.values, axis=0))
#Todo New
# Standardize the target variable (y)
#scaler_y = MinMaxScaler()
#scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLP Model / Input Features: (52+6+6+4=68) (was pre new features: 52, 35 / now 68, 45)
mlp = MLPRegressor(hidden_layer_sizes=(45), activation='relu', solver='adam', max_iter=500)
# mlp.fit(X_train, y_train)

# Define the ShuffleSplit cross-validation
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# Lists to store MSE and R² for each split
mse_scores = []
r2_scores = []

#Perform cross-validation
for i, (train_index, test_index) in enumerate(ss.split(X, y)):
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc()[test_index]

    # Train the model
    mlp.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(mlp, 'trained_mlp_model_02.pkl')
    joblib.dump(scaler, 'scaler_X.pkl')  # Save the input feature scaler
    joblib.dump(scaler_y, 'scaler_y.pkl')  # Save the output scaler



    # Predict on the test set
    y_pred = mlp.predict(X_test)

    #TODO new
    # Inverse transform the predicted values to get them back to original scale
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)

    print("Min-Werte nach Inverse Transform:", np.min(y_pred_original, axis=0))
    print("Max-Werte nach Inverse Transform:", np.max(y_pred_original, axis=0))

    #TODO new
    # Evaluate the model
    # opponent_weight = 2  # Increase the importance of Player 2's score
    # mse = mean_squared_error(y_test_original[:, 0], y_pred_original[:, 0]) + \
    #     opponent_weight * mean_squared_error(y_test_original[:, 1], y_pred_original[:, 1])

    mse = mean_squared_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    # #Todo Old
    # # Calculate Mean Squared Error and R²
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # Store the results
    mse_scores.append(mse)
    r2_scores.append(r2)


    print(f"Split {i + 1} - Predicted vs Target Values (First 10 samples):")
    for pred, target in zip(y_pred_original[:10], y_test.values[:10]):  # Limiting to 10 samples
        print(f"Predicted: {pred}, Target: {target}")
    print("\n" + "-"*50 + "\n")

    # Print the results for this split
    print(f"Split {i + 1} - MSE: {mse:.4f}, R²: {r2:.4f}")

# After all splits, you can print the average of the metrics
print(f"\nAverage MSE across 10 splits: {np.mean(mse_scores):.4f}")
print(f"Average R² across 10 splits: {np.mean(r2_scores):.4f}")

plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Mean Training Loss Curve')
plt.show()

# # Predictions
# y_pred = mlp.predict(X_test)
#
# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")

# --------------------------------
# Move Recommendation Function
# --------------------------------
def recommend_best_move(df_row, model):
    """
    Recommend the best move by simulating all possible partnerships
    and choosing the move that maximizes predicted new points.
    """
    # current_player = df_row['CurrentPlayer']
    # current_player = int(current_player)
    # print(f"Current Player: {current_player}")

    next_player = df_row['NextPlayer']
    next_player = int(next_player)
    print(f"Next Player: {next_player}")

    # Extract dots and their existing partnerships
    dots = [(df_row[f'Dot ID{i}'], df_row[f'X{i}'], df_row[f'Y{i}']) for i in range(8)]
    partnerships = [df_row[f'HasPartners{i}'] for i in range(8)]

    # Identify available dots (dots that have no partner)
    available_dots = [dots[i] for i in range(8) if partnerships[i] == 0]
    #available_dots = [dots[i] for i in range(8)]

    # Generate all possible new partnerships
    possible_moves = list(combinations(available_dots, 2))

    best_move = None
    best_score = -np.inf
    #best_score = 0

    for move in possible_moves:
        (dotA_id, dotA_x, dotA_y), (dotB_id, dotB_x, dotB_y) = move

        # Simulate the new move: create a new circle
        new_circle_x = (dotA_x + dotB_x) / 2
        new_circle_y = (dotA_y + dotB_y) / 2
        new_radius = np.sqrt((dotA_x - dotB_x) ** 2 + (dotA_y - dotB_y) ** 2) / 2

        # Create a temporary feature set for prediction
        temp_features = df_row[feature_cols].copy()
        temp_features['Circle1 ID A'] = dotA_id
        temp_features['Circle1 ID B'] = dotB_id
        temp_features['Circle1 X'] = new_circle_x
        temp_features['Circle1 Y'] = new_circle_y
        temp_features['Circle1 Radius'] = new_radius

        # Standardize the input features
        temp_X = scaler.transform([temp_features])

        predictions = model.predict(temp_X)

        # Round predictions to the nearest whole number
        #predictions = np.round(predictions).astype(int)  # Ensures integer values

        #Todo new
        # Inverse transform the predictions
        predictions_original = scaler_y.inverse_transform(predictions)
        print(f"Predicted points for this move: {predictions_original}")

        # print(predictions.shape)  # Print shape of predictions
        # print(predictions)        # Print the actual values
        # print(f"Move: Connect Dot {dotA_id} with Dot {dotB_id} → Predictions: {predictions}")
        print(f"Move: Connect Dot {dotA_id} with Dot {dotB_id} → Predictions: {predictions_original}")

        #print(f"Current Player: {current_player}")

        # Predict the new points for this move
        if next_player == 9:
            predicted_points = 0  # Special case for game has ended, no points for moves after last move
        else:
            #predicted_points = model.predict(temp_X)[0][current_player - 1]  # Get current player's points (0 for Player 1, 1 for Player 2)
            #predicted_points = model.predict(temp_X)[0][next_player - 1]  #  Todo Old Get current player's points (0 for Player 1, 1 for Player 2) <-- was old
            #predicted_points = predictions[0][next_player - 1]  # Get current player's points (0 for Player 1, 1 for Player 2)<-- is new but not perfect
        #    predicted_points = model.predict(temp_X)[0][2 - current_player]  # Get current player's points (0 for Player 1, 1 for Player 2)
            predicted_points = predictions_original[0][next_player - 1] # Todo New

        print(f"Predicted_points: {predicted_points}")

        # Check if this move is the best so far
        if predicted_points > best_score:
            best_score = predicted_points
            best_move = move

    return best_move, best_score

# --------------------------------
# Test the Move Recommendation on a Sample Game
# --------------------------------
sample_game_row = df.iloc[2]  # Select any game row
best_move, best_points = recommend_best_move(sample_game_row, mlp)

print(f"Best Move: Connect Dot {best_move[0][0]} with Dot {best_move[1][0]}")
print(f"Predicted Points Gained: {best_points}")

def predict_best_move(df_row, model, scaler, scaler_y, num_moves=5):
    """
    Predicts the best possible move by identifying two unpartnered dots,
    predicting a new circle, and selecting the best one based on the model's score.
    """
    next_player = int(df_row['NextPlayer'])  # Next player ID
    feature_set = df_row[feature_cols].copy()

    # Identify dots with no partners
    available_dots = [i for i in range(8) if df_row[f'HasPartners{i}'] == 0]

    if len(available_dots) < 2:
        return None, None, None  # Not enough free dots to create a circle

    # Generate all possible pair combinations
    dot_pairs = list(combinations(available_dots, 2))

    best_score = -np.inf
    best_circle = None
    best_pair = None

    for dotA, dotB in dot_pairs:
        # Extract coordinates
        x1, y1 = df_row[f'X{dotA}'], df_row[f'Y{dotA}']
        x2, y2 = df_row[f'X{dotB}'], df_row[f'Y{dotB}']

        for _ in range(num_moves):  # Generate variations for each pair

            # Predict circle center and radius (basic heuristic)
            circle_x = (x1 + x2) / 2
            circle_y = (y1 + y2) / 2
            radius = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2  # Half the distance between dots

            # Introduce slight variations (random noise)
            circle_x += np.random.uniform(-0.05, 0.05) * radius
            circle_y += np.random.uniform(-0.05, 0.05) * radius
            radius *= np.random.uniform(0.95, 1.05)  # 5% variation in size

            # Create a modified feature set with this new circle
            temp_features = feature_set.copy()
            temp_features.update({
                'Circle1 ID A': dotA, 'Circle1 ID B': dotB,
                'Circle1 X': circle_x, 'Circle1 Y': circle_y,
                'Circle1 Radius': radius
            })

            # Standardize features
            X_test = scaler.transform([temp_features])

            # Predict outcome
            y_pred = model.predict(X_test)
            y_pred_original = scaler_y.inverse_transform(y_pred)[0]

            # Check if this move is better
            if y_pred_original[next_player - 1] > best_score:
                best_score = y_pred_original[next_player - 1]
                best_circle = (circle_x, circle_y, radius)
                best_pair = (dotA, dotB)

    return best_pair, best_circle, best_score

# Example Usage
sample_game_row = df.iloc[2]  # Select any game row
dot_pair, predicted_circle, predicted_score = predict_best_move(sample_game_row, mlp, scaler, scaler_y)

print("Best Dot Pair for Circle:", dot_pair)
print("Predicted Circle (X, Y, Radius):", predicted_circle)
print("Predicted Score (Player 1 & 2):", predicted_score)




def plot_circles(df_row, best_move, new_circle_x, new_circle_y, new_radius):
    """
    Plots all existing circles in a given game state and highlights the new move.
    - Player 1's circles are blue.
    - Player 2's circles are red.
    - The newly created circle is green.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Colors alternate: even-indexed circles (0,2,4...) are blue, odd-indexed (1,3,5...) are red
    colors = ['blue', 'red']

    # Plot existing circles
    for i in range(4):  # Assuming max 4 circles
        x = df_row[f'Circle{i+1} X']
        y = df_row[f'Circle{i+1} Y']
        r = df_row[f'Circle{i+1} Radius']

        if r > 0:  # Only plot circles that have a valid radius
            color = colors[i % 2]  # Alternate colors: even = blue, odd = red
            circle = plt.Circle((x, y), r, color=color, alpha=0.3, label=f"Circle {i+1}")
            ax.add_patch(circle)

    # Plot new move in green
    new_circle = plt.Circle((new_circle_x, new_circle_y), new_radius, color='green', alpha=0.6, linestyle='dashed')
    ax.add_patch(new_circle)

    # Plot the dots
    for i in range(8):
        dot_x = df_row[f'X{i}']
        dot_y = df_row[f'Y{i}']
        dot_id = df_row[f'Dot ID{i}']
        plt.scatter(dot_x, dot_y, color='black', marker='o')
        plt.text(dot_x + 1, dot_y + 1, str(dot_id), fontsize=10, color='black', ha='center', va='center', fontweight='bold')

    # Set axis limits dynamically based on data range
    all_x = [df_row[f'X{i}'] for i in range(8)] + [new_circle_x]
    all_y = [df_row[f'Y{i}'] for i in range(8)] + [new_circle_y]

    ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
    ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
    ax.set_aspect('equal', adjustable='datalim')

    plt.title("Circle Game Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Extract best move circle's properties
(dotA_id, dotA_x, dotA_y), (dotB_id, dotB_x, dotB_y) = best_move
new_circle_x = (dotA_x + dotB_x) / 2
new_circle_y = (dotA_y + dotB_y) / 2
new_radius = np.sqrt((dotA_x - dotB_x) ** 2 + (dotA_y - dotB_y) ** 2) / 2

# Call the plotting function
plot_circles(sample_game_row, best_move, new_circle_x, new_circle_y, new_radius)