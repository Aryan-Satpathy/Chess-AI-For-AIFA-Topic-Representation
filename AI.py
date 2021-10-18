import UI
import numpy as np
import math
import time



# p, kn, b, r, q, ki
eval_matrix = [1, 3, 5, 7, 9, 4]

def getIndexofPiecesInEvalMatrix(piece) :
    
    if piece == 0b0000 or piece == 0b1000 :
        return 0
    elif piece == (UI.white | UI.rook) or piece == (UI.black | UI.rook) :
        return 3
    elif piece == UI.white | UI.bishop or piece == UI.black | UI.bishop :
        return 2
    elif piece == UI.white | UI.knight or piece == UI.black | UI.knight :
        return 1
    elif piece == UI.white | UI.queen or piece == UI.black | UI.queen :
        return 4
    elif piece == UI.white | UI.king or piece == UI.black | UI.king :
        return 5
    

def evaluation(current_state) :

    ## Total evaluation will be based on the weighted sum of evaluation of each of the factors.
    ## Weightages will be fractions of 1
    

    ## Weightage Factors

    ## Checked State
    c_state = 0.3

    ## Number of pieces
    n_pieces = 0.5

    ## Mobility of pieces
    m_pieces = 0.2



    np_board, white_pieces, black_pieces, castling_rights = current_state

    ## Check State
    score3 = 0
    if UI.board_check(np_board, UI.white >> 3) : score3 += 500
    elif UI.board_check(np_board, UI.black >> 3) : score3 -= 500

    score1 = 0
    

    for wpiece in white_pieces :
        if np_board[wpiece] == (UI.white | UI.pawn) :
            score1 += eval_matrix[0]
        elif np_board[wpiece] == (UI.white | UI.knight) :
            score1 += eval_matrix[1]
        elif np_board[wpiece] == (UI.white | UI.rook) :
            score1 += eval_matrix[2]
        elif np_board[wpiece] == (UI.white | UI.bishop) :
            score1 += eval_matrix[3]
        elif np_board[wpiece] == (UI.white | UI.queen) :
            score1 += eval_matrix[4]
    
    for bpiece in black_pieces :
        if np_board[bpiece] == (UI.black | UI.pawn) :
            score1 -= eval_matrix[0]
        elif np_board[bpiece] == (UI.black | UI.knight) :
            score1 -= eval_matrix[1]
        elif np_board[bpiece] == (UI.black | UI.rook) :
            score1 -= eval_matrix[2]
        elif np_board[bpiece] == (UI.black | UI.bishop) :
            score1 -= eval_matrix[3]
        elif np_board[bpiece] == (UI.black | UI.queen) :
            score1 -= eval_matrix[4]


    ## Mobility based evaluation

    score2 = 0
    
    localLegalMoves1 = UI.getLegalMoves(np_board, UI.white >> 3)
    localLegalMoves2 = UI.getLegalMoves(np_board, UI.black >> 3)

    for move1 in localLegalMoves1 :

        index = move1[0]
        piece = np_board[white_pieces[index]]
        indexEvalMatrix = getIndexofPiecesInEvalMatrix(piece)
        score2 += eval_matrix[indexEvalMatrix]

    for move2 in localLegalMoves2 :
        
        index = move2[0]
        piece = np_board[black_pieces[index]]
        indexEvalMatrix = getIndexofPiecesInEvalMatrix(piece)
        score2 -= eval_matrix[indexEvalMatrix]



    ##print("finalScore2", score2)



    eval = (score3 * c_state) + (score2 * m_pieces)   + (score1 * n_pieces)
    return eval

def Minimax(current_state, isMaximizing = True, alpha = -math.inf, beta = math.inf, depth = 1) :

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
best_move, best_score = Minimax(current_state, depth = 3)
end = time.time()
print(best_move, best_score, end - start)

UI.cv2.waitKey(0)