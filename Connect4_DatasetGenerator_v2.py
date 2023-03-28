import numpy as np
import random
#import #pygame
import sys
import math
import pandas as pd
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

WINDOW_LENGTH = 4
counter2 = 0
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


def score_position(board, piece):
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

    return score

def score_position2(board, piece):
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
            else:
                return (None, score_position2(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
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
    # best_score = -10000
    best_col = random.choice(valid_locations)

    """
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    """

    return best_col

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

turn = random.randint(PLAYER, AI)

# change this to the number of games you want to run
NUMBER_OF_GAMES = 30

while (game_num-1) < NUMBER_OF_GAMES:
    #print_board(board)
    print("Running Game " + str(game_num) + "/" + str(NUMBER_OF_GAMES) + "...")

    score_p1.append(score_position(board, PLAYER_PIECE))
    score_p2.append(score_position(board, AI_PIECE))

    game.append([])
    game_num_list.append(game_num)

    result_list.append("N/A")

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

            # selects a random column
            try:
                col = pick_best_move(board, PLAYER_PIECE)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                turn_list.append(turn + 1)
                turn += 1
                turn = turn % 2
            except:
                # check if the next move can be made
                try:
                    pick_best_move(board, PLAYER_PIECE)
                except:
                    turn_list.append(turn + 1)
                    turn += 1
                    turn = turn % 2
                    result_list.append("Draw")
                    game_over = True

            if winning_move(board, PLAYER_PIECE):
                #label = myfont.render("Player 1 wins!!", 1, RED)
                #screen.blit(label, (40, 10))
                game_over = True

            #print_board(board)
            #draw_board(board)

            game.append([])
            game_num_list.append(game_num)

            if winning_move(board, PLAYER_PIECE):
                result_list.append("P1 Win")
            elif game_over == False:
                result_list.append("N/A")

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

            # selects a random column
            try:
                col = pick_best_move(board, AI_PIECE)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                turn_list.append(turn + 1)
                turn += 1
                turn = turn % 2
            except:
                # check if the next move can be made
                try:
                    pick_best_move(board, AI_PIECE)
                except:
                    turn_list.append(turn + 1)
                    turn += 1
                    turn = turn % 2
                    result_list.append("Draw")
                    game_over = True

            if winning_move(board, AI_PIECE):
                #label = myfont.render("Player 2 wins!!", 1, YELLOW)
                #screen.blit(label, (40, 10))
                game_over = True

            #print_board(board)
            #draw_board(board)

            game.append([])
            game_num_list.append(game_num)

            if winning_move(board, AI_PIECE):
                result_list.append("P2 Win")
            elif game_over == False:
                result_list.append("N/A")

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
            #scores.append(score_p1)

    game_num += 1
    move_num = 0
    turn_list.append(turn + 1)
    game_over = False
    board = create_board()


print("Preparing Dataset... \n")

# making the dataset
columns.append("game")
columns.append("move")
columns.append("turn")
columns.append("p1_score")
columns.append("p2_score")
columns.append("winning")
columns.append("result")

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

df["winning"] = current_winner
df["result"] = result_list

print(df)

df.to_csv("./data.csv", index=False, index_label=False)
