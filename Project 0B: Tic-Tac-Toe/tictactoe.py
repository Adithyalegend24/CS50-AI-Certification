import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    return [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]


def player(board):
    x_count1 = 0
    o_count1 = 0

    for row in board:
        for cell in row:
            if cell == X:
                x_count1 += 1
            elif cell == O:
                o_count1 += 1

    return X if x_count1 <= o_count1 else O


def actions(board):
    moves = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.add((i, j))

    return moves


def result(board, action):
    i, j = action

    if i < 0 or i > 2 or j < 0 or j > 2 or board[i][j] is not EMPTY:
        raise Exception("Invalid move")

    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    lines = []

    for row in board:
        lines.append(row)

    for j in range(3):
        lines.append([board[i][j] for i in range(3)])

    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line[0] is not EMPTY and line.count(line[0]) == 3:
            return line[0]

    return None


def terminal(board):

    if winner(board) is not None:
        return True

    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False

    return True


def utility(board):
    win = winner(board)

    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):

    if terminal(board):
        return None

    curr_player = player(board)

    if curr_player == X:
        best_value = -math.inf
        best_move = None

        for action in actions(board):
            value = min_value(result(board, action))
            if value > best_value:
                best_value = value
                best_move = action

        return best_move

    else:
        best_value = math.inf
        best_move = None

        for action in actions(board):
            value = max_value(result(board, action))
            if value < best_value:
                best_value = value
                best_move = action

        return best_move


def max_value(board):
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
