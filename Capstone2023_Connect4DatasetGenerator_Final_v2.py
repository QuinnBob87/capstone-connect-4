"""
Building an AI to Play Strategy Games
Champlain College DATA x MATH Capstone 2023

Authors:
    Ryan Burns
    Quinn Agabob
"""

# Imports
import numpy as np
import random
import pandas as pd

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

# Variables and Lists
counter = 0
game_num = 1
move_num = 0

game = []
games = []
game_num_list = []
move_count_list = []
turn_list = []
score_p1 = []
score_p2 = []
current_winner = []
result_list = []
scores = []
columns = []


# Functions

# Creates an empty board
def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


# Drops a piece at a position for a given player
def drop_piece(board, row, col, piece):
    board[row][col] = piece


# Check if a piece can be placed in a certain location
def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


# Finds the next available row for new moves given a column
def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


# Check if the winning move has been made by either player
def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece \
            and board[r][c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece \
            and board[r + 3][c] == piece:
                return True

    # Check positively sloped diagonals for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece \
            and board[r + 3][c + 3] == piece:
                return True

    # Check negatively sloped diagonals for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece \
            and board[r - 3][c + 3] == piece:
                return True


# Determine scores based on where pieces have been placed
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    # Award points for consecutive pieces
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 10
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 5

    # Subtract points for opponent getting consecutive pieces
    if window.count(opp_piece) == 4:
        score -= 1000
    elif window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 12
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 1

    return score


# Award extra points based on column placement
def score_position(board, piece):
    score = 0

    # Column 1 & 7 (edges, no extra points)
    column1 = [int(i) for i in list(board[:, 0])]
    column1count = column1.count(piece)
    score += column1count * 0

    column7 = [int(i) for i in list(board[:, 6])]
    column7count = column7.count(piece)
    score += column7count * 0

    # Column 2 & 6 (1 point per piece)
    column2 = [int(i) for i in list(board[:, 1])]
    column2count = column2.count(piece)
    score += column2count * 1

    column6 = [int(i) for i in list(board[:, 5])]
    column6count = column6.count(piece)
    score += column6count * 1

    # Column 3 & 5 (2 points per piece)
    column3 = [int(i) for i in list(board[:, 2])]
    column3count = column3.count(piece)
    score += column3count * 2

    column5 = [int(i) for i in list(board[:, 4])]
    column5count = column5.count(piece)
    score += column5count * 2

    # Column 4 (center, 3 points per piece)
    column4 = [int(i) for i in list(board[:, 3])]
    column4count = column4.count(piece)
    score += column4count * 3

    # Score horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += (evaluate_window(window, piece) * 1.25)

    # Score negative sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += (evaluate_window(window, piece) * 1.25)

    return score


# Find valid locations for the next move to be placed
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


# Choose next moves randomly for dataset generation
def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_col = random.choice(valid_locations)

    return best_col


# Setting Up
file_name = input("\nWhat would you like the file to be saved as? (.csv will be added automatically):  ")

board = create_board()
game_over = False

# Pick a random player to go first
turn = random.randint(PLAYER, AI)

# Number of games to be played
NUMBER_OF_GAMES = 5

# Main Game Loop
while (game_num-1) < NUMBER_OF_GAMES:
    print("Running Game " + str(game_num) + "/" + str(NUMBER_OF_GAMES) + "...")

    # Add current player scores to each score list
    score_p1.append(score_position(board, PLAYER_PIECE))
    score_p2.append(score_position(board, AI_PIECE))

    # Add current game state and game number
    game.append([])
    game_num_list.append(game_num)

    # Game has not ended yet
    result_list.append("N/A")

    # Add board values for the current game state
    for x in range(6):
        game[counter].append([])
        for y in range(7):
            game[counter][x].append(board[abs(x - 5)][y])

    counter += 1

    # Add current move number and increase it by 1
    move_count_list.append(move_num)
    move_num += 1

    while not game_over:

        # Player 1's Turn
        if turn == PLAYER:
            try:
                # Makes a move and changes turns
                col = pick_best_move(board, PLAYER_PIECE)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                turn_list.append(turn + 1)
                turn += 1
                turn = turn % 2
            except:
                # Check if there are any moves left to make
                try:
                    pick_best_move(board, PLAYER_PIECE)
                except:
                    # If not, there are no moves left meaning that the game ended in a draw
                    turn_list.append(turn + 1)
                    turn += 1
                    turn = turn % 2

                    result_list.append("Draw")
                    game_over = True

            # Check for a win
            if winning_move(board, PLAYER_PIECE):
                game_over = True

            # Add current game state and game number
            game.append([])
            game_num_list.append(game_num)

            # If there was a winning move made, add this to the results list
            if winning_move(board, PLAYER_PIECE):
                result_list.append("P1 Win")
            else:
                result_list.append("N/A")

            # Add board values for the current game state
            for x in range(6):
                game[counter].append([])
                for y in range(7):
                    game[counter][x].append(board[abs(x - 5)][y])

            counter += 1

            # Add current player scores to each score list
            score_p1.append(score_position(board, PLAYER_PIECE))
            score_p2.append(score_position(board, AI_PIECE))

            # Add current move number and increase it by 1
            move_count_list.append(move_num)
            move_num += 1

        # Player 2's Turn
        if turn == AI and not game_over:
            try:
                # Makes a move and changes turns
                col = pick_best_move(board, AI_PIECE)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                turn_list.append(turn + 1)
                turn += 1
                turn = turn % 2
            except:
                # Check if there are any moves left to make
                try:
                    pick_best_move(board, AI_PIECE)
                except:
                    # If not, there are no moves left meaning that the game ended in a draw
                    turn_list.append(turn + 1)
                    turn += 1
                    turn = turn % 2

                    result_list.append("Draw")
                    game_over = True

            # Check for a win
            if winning_move(board, AI_PIECE):
                game_over = True

            # Add current game state and game number
            game.append([])
            game_num_list.append(game_num)

            # If there was a winning move made, add this to the results list
            if winning_move(board, AI_PIECE):
                result_list.append("P2 Win")
            else:
                result_list.append("N/A")

            # Add board values for the current game state
            for x in range(6):
                game[counter].append([])
                for y in range(7):
                    game[counter][x].append(board[abs(x - 5)][y])

            counter += 1

            # Add current player scores to each score list
            score_p1.append(score_position(board, PLAYER_PIECE))
            score_p2.append(score_position(board, AI_PIECE))

            # Add current move number and increase it by 1
            move_count_list.append(move_num)
            move_num += 1

        # Game is finished, add it to the list of finished games
        if game_over:
            games.append(game)

    # Reset board and update variables for the next game
    game_num += 1
    move_num = 0
    turn_list.append(turn + 1)

    game_over = False
    board = create_board()


# Creating the dataset
print("Preparing Dataset... \n")

# Adding column headers to the dataset
columns.append("game")
columns.append("move")
columns.append("turn")
columns.append("p1_score")
columns.append("p2_score")
columns.append("winning")
columns.append("result")

for y in range(6):
    for z in range(7):
        columns.append('(' + str(y) + ',' + str(z) + ')')

# Making the dataframe and adding the collected game state data
df = pd.DataFrame(columns=columns)

for x in range(len(games)):
    for x2 in range(len(games[x])):
        for y in range(6):
            for z in range(7):
                df.at[x2, '(' + str(y) + ',' + str(z) + ')'] = games[x][x2][y][z]

# Adding the other collected information and putting it under the appropriate columns
for x in range(len(score_p1)):
    if score_p1[x] > score_p2[x]:
        current_winner.append(1)
    elif score_p1[x] < score_p2[x]:
        current_winner.append(2)
    else:
        current_winner.append(0)

df["game"] = game_num_list
df["move"] = move_count_list
df["turn"] = turn_list
df["p1_score"] = score_p1
df["p2_score"] = score_p2
df["winning"] = current_winner
df["result"] = result_list

# Sending the dataframe to a .csv file
df.to_csv(file_name + ".csv", index=False, index_label=False)
print("The dataset has been saved as: " + file_name + ".csv")
