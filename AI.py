import UI
import numpy as np
import math

# p, kn, b, r, q
eval_matrix = [1, 3, 5, 7, 9]

def evaluation(current_state) :
    
    np_board, white_pieces, black_pieces, castling_rights = current_state

    if UI.board_check(np_board, UI.white >> 3) : return -math.inf
    elif UI.board_check(np_board, UI.black >> 3) : return math.inf

    score = 0

    for wpiece in white_pieces :
        if np_board[wpiece] == (UI.white | UI.pawn) :
            score += eval_matrix[0]
        elif np_board[wpiece] == (UI.white | UI.knight) :
            score += eval_matrix[1]
        elif np_board[wpiece] == (UI.white | UI.rook) :
            score += eval_matrix[2]
        elif np_board[wpiece] == (UI.white | UI.bishop) :
            score += eval_matrix[3]
        elif np_board[wpiece] == (UI.white | UI.queen) :
            score += eval_matrix[4]
    
    for bpiece in black_pieces :
        if np_board[bpiece] == (UI.black | UI.pawn) :
            score -= eval_matrix[0]
        elif np_board[bpiece] == (UI.black | UI.knight) :
            score -= eval_matrix[1]
        elif np_board[bpiece] == (UI.black | UI.rook) :
            score -= eval_matrix[2]
        elif np_board[bpiece] == (UI.black | UI.bishop) :
            score -= eval_matrix[3]
        elif np_board[bpiece] == (UI.black | UI.queen) :
            score -= eval_matrix[4]

    return score

def Minimax(current_state, isMaximizing = True, depth = 3) :

    np_board, white_pieces, black_pieces, castling_rights = current_state

    best_move = None
    best_score = [math.inf, -math.inf][isMaximizing]

    Moves = UI.getLegalMoves(np_board, [UI.black >> 3, UI.white >> 3][isMaximizing])

    for move in Moves :
        init, dest = move
        
        buffer = np_board[dest]
        np_board[dest] = np_board[init]
        np_board[init] = UI.empty

        current_state = np_board, white_pieces, black_pieces, castling_rights

        if depth :
            next_best_move, next_best_score = Minimax(current_state, not isMaximizing, depth - 1)
        else :
            next_best_move, next_best_score = None, evaluation(current_state)

        np_board[init] = np_board[dest]
        np_board[dest] = buffer
        buffer = UI.empty

        current_state = np_board, white_pieces, black_pieces, castling_rights
        
        if (isMaximizing and best_score < next_best_score) or  (not isMaximizing and best_score > next_best_score) :
            best_move = move
            best_score = next_best_score       

    return best_move, best_score

