"""
Building an AI to Play Strategy Games
Champlain College DATA x MATH Capstone 2023

Authors:
    Ryan Burns
    Quinn Agabob
"""

# Imports
import numpy as np
import pygame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from PIL import ImageGrab
import mouse


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
game_turn = 0

columns = []
initial_colors = []

# Reading data
df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('data2.csv')
df3 = pd.read_csv('data3.csv')
df4 = pd.read_csv('data4.csv')
df5 = pd.read_csv('data5.csv')
df6 = pd.read_csv('data6.csv')
df7 = pd.read_csv('data7.csv')
df8 = pd.read_csv('data8.csv')
df9 = pd.read_csv('data9.csv')
df10 = pd.read_csv('data10.csv')

# Combining dataframes
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])


# Machine Learning

# Features for machine learning part: board and turn
Features = df[['move', '(0,0)', '(0,1)', '(0,2)', '(0,3)', '(0,4)', '(0,5)', '(0,6)'
                     , '(1,0)', '(1,1)', '(1,2)', '(1,3)', '(1,4)', '(1,5)', '(1,6)'
                     , '(2,0)', '(2,1)', '(2,2)', '(2,3)', '(2,4)', '(2,5)', '(2,6)'
                     , '(3,0)', '(3,1)', '(3,2)', '(3,3)', '(3,4)', '(3,5)', '(3,6)'
                     , '(4,0)', '(4,1)', '(4,2)', '(4,3)', '(4,4)', '(4,5)', '(4,6)'
                     , '(5,0)', '(5,1)', '(5,2)', '(5,3)', '(5,4)', '(5,5)', '(5,6)']]

# Target for machine learning part: score
Target = df['p1_score']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=.2)

# Decision Tree Regressor code
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Random Forest Regressor code (this is the one we use)
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(Features, Target)
y_pred = clf.predict(Features)

# Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred2 = reg.predict(X_test)
r2 = r2_score(y_test, y_pred2)
mse = mean_squared_error(y_test, y_pred2)


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


# Prints the board
def print_board(board):
    print(np.flip(board, 0))
    print()


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

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece \
            and board[r + 3][c + 3] == piece:
                return True

    # Check negatively sloped diagonals
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
# (this is a combination of basic strategy and our Linear Regression Model)
def score_position2(board, piece):
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

    # Assigns the basic strategy to traditional score
    trad_score = score

    # Machine learning Linear Regression score
    ml_code = reg.intercept_ + reg.coef_[0] * game_turn

    if board[0][0] == piece:
        ml_code += reg.coef_[1]
    if board[0][1] == piece:
        ml_code += reg.coef_[2]
    if board[0][2] == piece:
        ml_code += reg.coef_[3]
    if board[0][3] == piece:
        ml_code += reg.coef_[4]
    if board[0][4] == piece:
        ml_code += reg.coef_[5]
    if board[0][5] == piece:
        ml_code += reg.coef_[6]
    if board[0][6] == piece:
        ml_code += reg.coef_[7]
    if board[1][0] == piece:
        ml_code += reg.coef_[8]
    if board[1][1] == piece:
        ml_code += reg.coef_[9]
    if board[1][2] == piece:
        ml_code += reg.coef_[10]
    if board[1][3] == piece:
        ml_code += reg.coef_[11]
    if board[1][4] == piece:
        ml_code += reg.coef_[12]
    if board[1][5] == piece:
        ml_code += reg.coef_[13]
    if board[1][6] == piece:
        ml_code += reg.coef_[14]
    if board[2][0] == piece:
        ml_code += reg.coef_[15]
    if board[2][1] == piece:
        ml_code += reg.coef_[16]
    if board[2][2] == piece:
        ml_code += reg.coef_[17]
    if board[2][3] == piece:
        ml_code += reg.coef_[18]
    if board[2][4] == piece:
        ml_code += reg.coef_[19]
    if board[2][5] == piece:
        ml_code += reg.coef_[20]
    if board[2][6] == piece:
        ml_code += reg.coef_[21]
    if board[3][0] == piece:
        ml_code += reg.coef_[22]
    if board[3][1] == piece:
        ml_code += reg.coef_[23]
    if board[3][2] == piece:
        ml_code += reg.coef_[24]
    if board[3][3] == piece:
        ml_code += reg.coef_[25]
    if board[3][4] == piece:
        ml_code += reg.coef_[26]
    if board[3][5] == piece:
        ml_code += reg.coef_[27]
    if board[3][6] == piece:
        ml_code += reg.coef_[28]
    if board[4][0] == piece:
        ml_code += reg.coef_[29]
    if board[4][1] == piece:
        ml_code += reg.coef_[30]
    if board[4][2] == piece:
        ml_code += reg.coef_[31]
    if board[4][3] == piece:
        ml_code += reg.coef_[32]
    if board[4][4] == piece:
        ml_code += reg.coef_[33]
    if board[4][5] == piece:
        ml_code += reg.coef_[34]
    if board[4][6] == piece:
        ml_code += reg.coef_[35]
    if board[5][0] == piece:
        ml_code += reg.coef_[36]
    if board[5][1] == piece:
        ml_code += reg.coef_[37]
    if board[5][2] == piece:
        ml_code += reg.coef_[38]
    if board[5][3] == piece:
        ml_code += reg.coef_[39]
    if board[5][4] == piece:
        ml_code += reg.coef_[40]
    if board[5][5] == piece:
        ml_code += reg.coef_[41]
    if board[5][6] == piece:
        ml_code += reg.coef_[42]

    # Machine learning code can generate large numbers, this code evens it out
    if ml_code > trad_score:
        if trad_score > 0:
            if ml_code/trad_score > 2:
                trad_score = ((ml_code/trad_score)/2) * trad_score

    score = trad_score + ml_code

    return score


# This is the score that utilizes the Random Forest Regressor model
def score_position(board, piece):
    colsss = []
    colsss.append('move')

    # Our previous game data is stored in a dataframe format
    # Our current game data is stored in a 2D list format
    # The code below converts the current game from a 2D list to a dataframe
    for y in range(6):
        for z in range(7):
            colsss.append('(' + str(y) + ',' + str(z) + ')')

    de_board = pd.DataFrame(columns=colsss)

    # Assigning the column titles
    de_board.at[0, 'move'] = game_turn
    for y in range(6):
        for z in range(7):
            de_board.at[0, '(' + str(y) + ',' + str(z) + ')'] = board[y][z]

    prediction = clf.predict(de_board)

    # Return the predicted score of a given board state
    return prediction


# Find valid locations for the next move to be placed
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


# Strategy for finding the best move
def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)

    # Sets scores to an unobtainable value
    highest = -100000000

    # List for scores
    scores1 = []

    # List for column numbers
    poss = []

    for col in valid_locations:
        # Sets oppiece to the opponents corresponding piece number
        if piece == 1:
            oppiece = 2
        else:
            oppiece = 1

        # Player Move #1, gets the next spot to place piece
        row = get_next_open_row(board, col)

        # Creates a temporary board
        temp_board = board.copy()

        # Drops hypothetical piece #1
        drop_piece(temp_board, row, col, piece)

        # Opponent move, gets possible moves after player's first move
        valid_locations2 = get_valid_locations(temp_board)
        scores2 = []

        for col2 in valid_locations2:
            # Gets the next spot to place piece
            row2 = get_next_open_row(temp_board, col2)

            # Creates a temporary board
            temp_board2 = temp_board.copy()

            # Drops hypothetical piece #2
            drop_piece(temp_board2, row2, col2, oppiece)

            # Player move #2
            valid_locations3 = get_valid_locations(temp_board2)
            scores3 = []

            for col3 in valid_locations3:
                # Gets the next spot to place piece
                row3 = get_next_open_row(temp_board2, col3)

                # Creates a temporary board
                temp_board3 = temp_board2.copy()

                # Drops hypothetical piece #3
                drop_piece(temp_board3, row3, col3, piece)

                # Assigns the score to this possibility
                score = score_position(temp_board3, piece)

                # Gets list of the best moves of 2nd player move
                scores3.append(score)

            # Gets best move of 2nd player move
            scores2.append(max(scores3))

        # Gets best move after the opponent plays their best move
        scores1.append(min(scores2))

        # Assigns the position of the moves
        poss.append(col)

    # Looks through the list of the best moves
    index = 0
    for x in range(len(scores1)):
        # Finds the highest score of the list
        if scores1[x] > highest:
            highest = scores1[x]
            # Sets the column based off the position of scores in score list
            index = x

    best_col = poss[index]

    # Returns the location of the best move
    return best_col


# Determines where on the screen to click given a column
def position2click(col):
    mouse_x = 660 + (col * 100)

    mouse.move(mouse_x, 835)
    pygame.time.wait(100)
    mouse.click()


# Compares screenshots to determine where the opponent made their move
def find_opponent_move(im1):
    move_made = False
    valid_locations = get_valid_locations(board)

    # Runs while the opponent has not made their move
    while not move_made:
        pygame.time.wait(10)
        im2 = ImageGrab.grab()

        for col in valid_locations:
            row = get_next_open_row(board, col)
            screen_pos = (640+(col*100), 800-(row*100))

            # Compares past and present board states
            rgb1 = im1.getpixel(screen_pos)
            rgb2 = im2.getpixel(screen_pos)

            # If one of the possible next moves is a different color now, that must be where the opponent moved
            if rgb1 != rgb2 or rgb1 != (16, 27, 39):
                drop_piece(board, row, col, AI_PIECE)
                # Move has been made
                move_made = True


# Setting Up
board = create_board()
game_over = False

# Determining the initial turn order
turn_image = ImageGrab.grab()
for col in range(COLUMN_COUNT):
    screen_color = turn_image.getpixel((640+(col*100), 800))
    initial_colors.append(screen_color)

# Assume by default that it is the player's turn
turn = PLAYER

# If a piece has already been placed in the bottom row, it must have been the opponent's turn first
for color in initial_colors:
    if color != (16, 27, 39):
        turn = AI
        # Save a screenshot of the opponent's first move
        image1 = ImageGrab.grab()

print_board(board)

# Main Game Loop
while not game_over:
    # Player 1 (Capstone AI)
    if turn == PLAYER:
        # Pick the best move
        col = pick_best_move(board, PLAYER_PIECE)
        row = get_next_open_row(board, col)

        # Save a screenshot before making the move
        image1 = ImageGrab.grab()
        pygame.time.wait(10)

        # Update the board in the output and click to make the same move on the website
        drop_piece(board, row, col, PLAYER_PIECE)
        position2click(col)
        print_board(board)

        # Check for a win
        if winning_move(board, PLAYER_PIECE):
            game_over = True

        # Change turns
        turn += 1
        turn = turn % 2

    # Player 2 (Website AI)
    if turn == AI and not game_over:
        # Check if opponent has made their move
        find_opponent_move(image1)
        pygame.time.wait(10)
        print_board(board)

        # Check for a win
        if winning_move(board, AI_PIECE):
            game_over = True

        # Change turns
        turn += 1
        turn = turn % 2
