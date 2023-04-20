from re import X
import numpy as np
import random
import pygame
import sys
import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image, ImageGrab, ImageDraw
import mouse


# Import pillow and mouse if you haven't already
# Go to https://papergames.io/en/connect4
# Start a game against the bot with unlimited time
# Starting player can be random
# When the countdown finishes, run the code and manually click back to the game window
# The extra time below is there so the board will be on screen when the game loop starts
# Then, the capstone AI should be able to play against the bot
# It takes a while to make a decision, so just be patient at the beginning especially
# When the game is over, go back to the code and hopefully the output should have the same board state
# Also, you can play against our AI by starting a game against a friend and scanning the QR code on your phone
# If you do that, and you go first, make sure you make a move before running the code, or it might not work
pygame.time.wait(5000)


BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
game_turn = 0

WINDOW_LENGTH = 4
counter2 = 0
game_num = 1
move_num = 1
game = []
games = []
game_num_list = []
move_count_list = []
turn_list = []
score_p1 = []
score_p2 = []
current_winner = []
scores = []
columns = []
initial_colors = []


df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('data2.csv')
df3 = pd.read_csv('data3.csv')
df4 = pd.read_csv('data4.csv')
df5 = pd.read_csv('data5.csv')

df = df1
df = df1.append(df2)
df = df1.append(df3)
df = df1.append(df4)
df = df1.append(df5)

Features = df[['move', '(0,0)', '(0,1)', '(0,2)', '(0,3)', '(0,4)', '(0,5)', '(0,6)'
, '(1,0)', '(1,1)', '(1,2)', '(1,3)', '(1,4)', '(1,5)', '(1,6)'
, '(2,0)', '(2,1)', '(2,2)', '(2,3)', '(2,4)', '(2,5)', '(2,6)'
, '(3,0)', '(3,1)', '(3,2)', '(3,3)', '(3,4)', '(3,5)', '(3,6)'
, '(4,0)', '(4,1)', '(4,2)', '(4,3)', '(4,4)', '(4,5)', '(4,6)'
, '(5,0)', '(5,1)', '(5,2)', '(5,3)', '(5,4)', '(5,5)', '(5,6)']]

Target = df['p1_score']
# target should also be: 
# test score by predicting the score of all possible moves (up to 7)

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=.2)
# use decision tree
tree = DecisionTreeRegressor()
#tree = DecisionTreeClassifier()
# pass in training data as the trained data
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
#mse determines accuracy
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)
print(y_test.shape)
print(y_pred.shape)
# utilize link
# y test and y pred needs to be different shapes before score function
# .reshape(1,-1)
#accuracy = tree.score(y_test, y_pred)
#print('Accuracy:', accuracy)

# use random forrest regressor

clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(Features, Target)
y_pred = clf.predict(Features)
print(clf.score(X_test, y_test))

"""
#use random forrest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, df)
"""

# linear regression moodel
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred2 = reg.predict(X_test)
r2 = r2_score(y_test, y_pred2)
mse = mean_squared_error(y_test, y_pred2)
"""
print(f"R-squared: {r2}")
print(f"Mean squared error: {mse}")
print(reg.intercept_)
print(reg.coef_)
"""


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
  board[row][col] = piece
    # board_df = pd.DataFrame(board)
    #  note: write to csv for every game state
    # board_df.to_csv("./data.csv")


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 10
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 5

    if window.count(opp_piece) == 4:
        score -= 1000
    elif window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 12
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 1

    return score


def score_position2(board, piece):
    score = 0

    ## Score columns
    # 1 and 7 (edges)
    column1 = [int(i) for i in list(board[:, 0])]
    column7 = [int(i) for i in list(board[:, 6])]
    column1count = column1.count(piece)
    column7count = column7.count(piece)
    score += column1count * 0
    score += column7count * 0

    # 2 and 6
    column2 = [int(i) for i in list(board[:, 1])]
    column6 = [int(i) for i in list(board[:, 5])]
    column2count = column2.count(piece)
    column6count = column6.count(piece)
    score += column2count * 1
    score += column6count * 1

    # 3 and 5
    column3 = [int(i) for i in list(board[:, 2])]
    column5 = [int(i) for i in list(board[:, 4])]
    column3count = column3.count(piece)
    column5count = column5.count(piece)
    score += column3count * 2
    score += column5count * 2

    # 4 (center)
    column4 = [int(i) for i in list(board[:, 3])]
    column4count = column4.count(piece)
    score += column4count * 3


    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += (evaluate_window(window, piece) * 1.25)

    ## Score negative sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += (evaluate_window(window, piece) * 1.25)

    trad_score = score

# ml lin reg score
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

    # ml code can generate large numbers, this codes evens it out
    if ml_code > trad_score:
      if trad_score > 0:
        if ml_code/trad_score > 2:
          trad_score = ((ml_code/trad_score)/2) * trad_score

    score = trad_score + ml_code
    

    return score


def score_position(board, piece):
  colsss = []
  colsss.append('move')
  for y in range(6):
          for z in range(7):
              colsss.append('(' + str(y) + ',' + str(z) + ')')
  de_board = pd.DataFrame(columns= colsss)
  de_board.at[0, 'move'] = game_turn
  for y in range(6):
            for z in range(7):
                de_board.at[0, '(' + str(y) + ',' + str(z) + ')'] = board[y][z]
  #print('RE: \n' + str(Features))
  #print('DE: \n' + str(de_board))
  prediction = clf.predict(de_board)
  #print('prediction:' + str(prediction))
  return prediction

"""
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score
"""

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            if turn == AI:
                return (None, score_position(board, AI_PIECE))
            #else:
                #return (None, score_position2(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            print()
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    tmpbest_score = 10000000
    scores1=[]
    poss = []
    highest= -100000000
    #best_col = random.choice(valid_locations)
    for col in valid_locations:
        if piece == 1:
          oppiece = 2
        elif piece == 2:
          oppiece = 1
        #player move #1
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        # opponent move
        valid_locations2 = get_valid_locations(temp_board)
        scores2 = [] 
        for col2 in valid_locations2:
          row2 = get_next_open_row(temp_board, col2)
          temp_board2 = temp_board.copy()
          drop_piece(temp_board2, row, col2, oppiece)
          #player move #2
          valid_locations3 = get_valid_locations(temp_board2)
          scores3 = []
          for col3 in valid_locations2:
            row3 = get_next_open_row(temp_board2, col3)
            temp_board3 = temp_board2.copy()
            drop_piece(temp_board3, row, col3, piece)
            score = score_position(temp_board3, piece)
            #print('position: ;' + str(col) + ' : ' + str(col2) + ' : ' + str(col3) + ' : ')
            #print('score: ;' + str(score))
            #gets list of the best moves of 2nd player move
            scores3.append(score)
          #gets best move of 2nd player move
          scores2.append(max(scores3))
        scores1.append(min(scores2))
        poss.append(col)
    
    for x in range(len(scores1)):
        if scores1[x] > highest:
            highest = scores1[x]
            index = x
    tmprng = randint(1, 10)
    if tmprng == 5:
      scores1.remove(highest)
      poss.remove(poss[index])
    for x in range(len(scores1)):
        if scores1[x] > highest:
            highest = scores1[x]
            index = x
    best_col = poss[index]
    return best_col

"""
    for col in valid_locations:
        if piece == 1:
          oppiece = 2
        elif piece == 2:
          oppiece = 1
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        #score = score_position(temp_board, piece)

        valid_locations2 = get_valid_locations(temp_board)
        for col in valid_locations2:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)

        #main_scores['scores' + str(col)]
        #scores[str(col)] = str(col)
        #print('This score is: ' + str(score))
        if score > best_score:
            best_score = score
            best_col = col


    return best_col
"""
"""
def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            #pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            #pygame.draw.circle(screen, BLACK, (
            #int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                #pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                #pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    #pygame.display.update()
"""

def position2click(col):
    mouse_x = 660 + (col * 100)

    mouse.move(mouse_x, 835)
    pygame.time.wait(100)
    mouse.click()


def find_opponent_move(im1):
    move_made = False
    valid_locations = get_valid_locations(board)
    while not move_made:
        pygame.time.wait(1000)
        im2 = ImageGrab.grab()
        for col in valid_locations:
            row = get_next_open_row(board, col)
            screen_pos = (640+(col*100), 800-(row*100))

            rgb1 = im1.getpixel(screen_pos)
            #print(rgb1)
            rgb2 = im2.getpixel(screen_pos)
            #print(rgb2)
            #print(str(row) + " " + str(col))

            if rgb1 != rgb2 or rgb1 != (16, 27, 39):
                drop_piece(board, row, col, AI_PIECE)
                move_made = True



board = create_board()
#print_board(board)
game_over = False

#pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

#screen = #pygame.display.set_mode(size)
#draw_board(board)
#pygame.display.update()

#myfont = #pygame.font.SysFont("monospace", 75)

turn_image = ImageGrab.grab()
for col in range(COLUMN_COUNT):
    screen_color = turn_image.getpixel((640+(col*100), 800))
    initial_colors.append(screen_color)

turn = PLAYER

for c in initial_colors:
    if c != (16, 27, 39):
        turn = AI
        image1 = ImageGrab.grab()

print(turn)

NUMBER_OF_GAMES = 1

while (game_num-1) < NUMBER_OF_GAMES:
    print_board(board)

    score_p1.append(score_position(board, PLAYER_PIECE))
    score_p2.append(score_position(board, AI_PIECE))

    game.append([])
    game_num_list.append(game_num)

    for x in range(6):
        game[counter2].append([])
        for y in range(7):
            game[counter2][x].append(board[abs(x - 5)][y])

    counter2 = counter2 + 1

    move_count_list.append(move_num)
    move_num += 1

    while not game_over:

        if turn == PLAYER:

            """
                    # posx = event.pos[0]
                    # col = int(math.floor(posx / SQUARESIZE))
            col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)
            """

            col = pick_best_move(board, PLAYER_PIECE)
            row = get_next_open_row(board, col)
            image1 = ImageGrab.grab()
            pygame.time.wait(100)
            drop_piece(board, row, col, PLAYER_PIECE)

            position2click(col)

            if winning_move(board, PLAYER_PIECE):
                #label = myfont.render("Player 1 wins!!", 1, RED)
                #screen.blit(label, (40, 10))
                game_over = True
            game_turn = game_turn + 1
            turn_list.append(turn + 1)
            turn += 1
            turn = turn % 2

            print_board(board)
            #draw_board(board)

            game.append([])
            game_num_list.append(game_num)

            for x in range(6):
                game[counter2].append([])
                for y in range(7):
                    game[counter2][x].append(board[abs(x - 5)][y])

            counter2 = counter2 + 1

            score_p1.append(score_position(board, PLAYER_PIECE))
            score_p2.append(score_position(board, AI_PIECE))

            move_count_list.append(move_num)
            move_num += 1

        # # Ask for Player 2 Input
        if turn == AI and not game_over:

            """
            col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
            if is_valid_location(board, col):
                #pygame.time.wait(500)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)
            """

            """
            # selects a random column
            col = pick_best_move(board, AI_PIECE)
            row = get_next_open_row(board, col)
            #pygame.time.wait(100)
            """
            pygame.time.wait(1000)
            #image2 = ImageGrab.grab()

            find_opponent_move(image1)

            pygame.time.wait(500)

            #drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                #label = myfont.render("Player 2 wins!!", 1, YELLOW)
                #screen.blit(label, (40, 10))
                game_over = True

            print_board(board)
            #draw_board(board)

            game_turn = game_turn + 1
            turn_list.append(turn + 1)
            turn += 1
            turn = turn % 2

            game.append([])
            game_num_list.append(game_num)

            for x in range(6):
                game[counter2].append([])
                for y in range(7):
                    game[counter2][x].append(board[abs(x - 5)][y])

            counter2 += 1

            score_p1.append(score_position(board, PLAYER_PIECE))
            score_p2.append(score_position(board, AI_PIECE))

            move_count_list.append(move_num)
            move_num += 1

        if game_over:
            ##pygame.time.wait(30000)

            #print(score_position(board, AI_PIECE))
            #print(score_position2(board, PLAYER_PIECE))
            #print(game)
            games.append(game)
            scores.append(score_p1)

    game_num += 1
    move_num = 1
    turn_list.append(turn + 1)
    game_over = False
    board = create_board()

"""
# making the dataset
columns.append("game")
columns.append("move")
columns.append("turn")
columns.append("p1_score")
columns.append("p2_score")
columns.append("winner")

#for x in range(len(games)):
    #columns.append('score state | ' + str(x))
for y in range(6):
    for z in range(7):
        columns.append('(' + str(y) + ',' + str(z) + ')')
        #columns.append(str(y) + ':' + str(z) + ' | ' + str(x))

df = pd.DataFrame(columns=columns)

for x in range(len(games)):
    #df['score state | ' + str(x)] = scores[x]
    for x2 in range(len(games[x])):
        for y in range(6):
            for z in range(7):
                #df.at[x2, str(y) + ':' + str(z) + ' | ' + str(x)] = games[x][x2][y][z]
                df.at[x2, '(' + str(y) + ',' + str(z) + ')'] = games[x][x2][y][z]

df["game"] = game_num_list
df["move"] = move_count_list
df["turn"] = turn_list
df["p1_score"] = score_p1
df["p2_score"] = score_p2

for x in range(len(score_p1)):
    if score_p1[x] > score_p2[x]:
        current_winner.append(1)
    elif score_p1[x] < score_p2[x]:
        current_winner.append(2)
    else:
        current_winner.append(0)

df["winner"] = current_winner

print(df)

df.to_csv("./data.csv", index=False, index_label=False)
"""
