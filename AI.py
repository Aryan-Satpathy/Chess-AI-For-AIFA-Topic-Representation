import UI
import numpy as np
import math
import time
import PIL

# p, kn, b, r, q
eval_matrix = [1, 3, 5, 7, 9]

def evaluation(current_state) :
    
    np_board, white_pieces, black_pieces, castling_rights = current_state

    if UI.board_check(np_board, UI.white >> 3) : return -500
    elif UI.board_check(np_board, UI.black >> 3) : return 500

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

def Minimax(current_state, isMaximizing = True, alpha = -math.inf, beta = math.inf, depth = 3) :

    np_board, white_pieces, black_pieces, castling_rights = current_state

    best_move = []
    best_score = [math.inf, -math.inf][isMaximizing]

    Moves = UI.getLegalMoves(np_board, [UI.black >> 3, UI.white >> 3][isMaximizing])

    if len(Moves) == 0 : 
        print("Doomsday !!!")

    for move in Moves :
        i, dest = move
        if isMaximizing :
            init = white_pieces[i]
        else :
            init = black_pieces[i]
        
        buffer = np_board[dest]
        np_board[dest] = np_board[init]
        np_board[init] = UI.empty

        if isMaximizing :
            white_pieces[i] = dest
            try : 
                ind = black_pieces.index(dest)
                black_pieces.pop(ind)
            except :
                ind = -1
        else :
            black_pieces[i] = dest
            try : 
                ind = white_pieces.index(dest)
                white_pieces.pop(ind)
            except :
                ind = -1

        current_state = np_board, white_pieces, black_pieces, castling_rights

        if depth :
            next_best_move, next_best_score = Minimax(current_state, not isMaximizing, alpha, beta, depth - 1)
        else :
            next_best_move, next_best_score = [], evaluation(current_state)

        if isMaximizing :
            white_pieces[i] = init
            if ind >= 0 :
                black_pieces.insert(ind, dest)  
        else :
            black_pieces[i] = init
            if ind >= 0 :
                UI.white_pieces.insert(ind, dest)  
        
        np_board[init] = np_board[dest]
        np_board[dest] = buffer
        buffer = UI.empty

        current_state = np_board, white_pieces, black_pieces, castling_rights
        
        if (isMaximizing and best_score < next_best_score) or (not isMaximizing and best_score > next_best_score) :
            best_move = [(init, dest)] + next_best_move
            best_score = next_best_score 
        
        if (isMaximizing and best_score > alpha) :
            alpha = best_score
        if (not isMaximizing and best_score < beta) :
            beta = best_score

        if (isMaximizing and best_score >= beta) or (not isMaximizing and best_score <= alpha) :
            break     

    return best_move, best_score

current_state = UI.np_board, UI.white_pieces, UI.black_pieces, UI.castleRights

print("Evaluation = ", evaluation(current_state))

start = time.time()
best_move, best_score = Minimax(current_state, depth = 5)
end = time.time()
print(best_move, best_score, end - start)

UI.cv2.waitKey(0)