import time
import math
import random
import chess

N = 0



def isQuiet(current_state, isMaximizing) :
    # Evaluation ain't changing much from this state
    np_board, white_pieces, black_pieces, castling_rights = current_state

    Moves = UI.getLegalMoves(np_board, white_pieces, black_pieces, [UI.black >> 3, UI.white >> 3][isMaximizing])

    # if UI.board_check(np_board, int(isMaximizing)) :
    #     return False

    for move in Moves :
        if np_board[move[1]] != UI.empty :
            return False
    
    return True

def qsearch(current_state, isMaximizing, alpha, beta, depth = 3) :
    np_board, white_pieces, black_pieces, castling_rights = current_state

    best_move = []
    best_score = [math.inf, -math.inf][isMaximizing]

    Moves = UI.getLegalMoves(np_board, white_pieces, black_pieces, [UI.black >> 3, UI.white >> 3][isMaximizing])

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

        if np_board[dest] == UI.empty :
            if not UI.board_check(np_board, white_pieces, black_pieces, not isMaximizing) :
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
        
                continue

        if depth == 0 or isQuiet(current_state, not isMaximizing) :
            next_best_move, next_best_score = [], evaluation(current_state)
        else :
            next_best_move, next_best_score = qsearch(current_state, not isMaximizing, alpha, beta, depth - 1)

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

    if abs(best_score) > 500 : 
        best_score = evaluation(current_state)

    return best_move, best_score

'''
def evaluation(current_state) :
    ## Total evaluation will be based on the weighted sum of evaluation of each of the factors.
    ## Weightages will be fractions of 1
    ## Weightage Factors
    ## Checked State
    c_state = 0.8
    ## Number of pieces
    n_pieces = 0
    ## Mobility of pieces
    m_pieces = 0.2

    np_board, white_pieces, black_pieces, castling_rights = current_state

    # score1 : material
    # score2 : mobility
    # score3 : check
    score1, score2, score3 = 0, 0, 0

    ## Check State
    if UI.board_check(np_board, white_pieces, black_pieces, UI.white >> 3) : score3 = -500
    elif UI.board_check(np_board, white_pieces, black_pieces, UI.black >> 3) : score3 = 500

    ## Mobility based evaluation    
    localLegalMoves1 = UI.getFreeBoardLegalMoves(np_board, white_pieces, black_pieces, UI.white >> 3)
    localLegalMoves2 = UI.getFreeBoardLegalMoves(np_board, white_pieces, black_pieces, UI.black >> 3)

    for move1 in localLegalMoves1 :
        index = move1[0]
        piece = np_board[white_pieces[index]]
        piece = piece & ((1 << 3) - 1)
        if piece  == UI.pawn :
            score1 += eval_matrix[0]
            score2 += eval_matrix[0]
        elif piece == UI.knight :
            score1 += eval_matrix[1]
            score2 += eval_matrix[1]
        elif piece == UI.bishop :
            score1 += eval_matrix[2]
            score2 += eval_matrix[2]
        elif piece == UI.rook :
            score1 += eval_matrix[3]
            score2 += eval_matrix[3]
        elif piece == UI.queen :
            score1 += eval_matrix[4]
            score2 += eval_matrix[4]
        elif piece == UI.king :
            score2 += eval_matrix[5]

    for move2 in localLegalMoves2 :        
        index = move2[0]
        piece = np_board[black_pieces[index]]
        piece = piece & ((1 << 3) - 1)
        if piece  == UI.pawn :
            score1 -= eval_matrix[0]
            score2 -= eval_matrix[0]
        elif piece == UI.knight :
            score1 -= eval_matrix[1]
            score2 -= eval_matrix[1]
        elif piece == UI.bishop :
            score1 -= eval_matrix[2]
            score2 -= eval_matrix[2]
        elif piece == UI.rook :
            score1 -= eval_matrix[3]
            score2 -= eval_matrix[3]
        elif piece == UI.queen :
            score1 -= eval_matrix[4]
            score2 -= eval_matrix[4]
        elif piece == UI.king :
            score2 -= eval_matrix[5]

    return score3 * c_state + score2 * m_pieces + score1 * n_pieces
'''

# p, kn, b, r, q, ki
eval_matrix = [10, 30, 30, 50, 90, 900]

king_mob_matrix = [
    -3,-4,-4,-5,-5,-4,-4,-3,
    -3,-4,-4,-5,-5,-4,-4,-3,
    -3,-4,-4,-5,-5,-4,-4,-3,
    -3,-4,-4,-5,-5,-4,-4,-3,
    -2,-3,-3,-4,-4,-3,-3,-2,
    -1,-2,-2,-2,-2,-2,-2,-1,
    2,2,0,0,0,0,2,2,
    2,3,1,0,0,1,3,2
]
queen_mob_matrix = [
    -2,-1,-1,-0.5,-0.5,-1,-1,-2,
    -1,0,0,0,0,0,0,-1,
    -1,0,0.5,0.5,0.5,0.5,0,-1,
    -0.5,0,0.5,0.5,0.5,0.5,0,-0.5,
    0,0,0.5,0.5,0.5,0.5,0,-0.5,
    -1,0.5,0.5,0.5,0.5,0.5,0,-1,
    -1,0,0.5,0,0,0,0,-1,
    -2,-1,-1,-0.5,-0.5,-1,-1,-2
]
bishop_mob_matrix = [
    -2,-1,-1,-1,-1,-1,-1,-2,
    -1,0,0,0,0,0,0,-1,
    -1,0,0.5,1,1,0.5,0,-1,
    -1,0.5,0.5,1,1,0.5,0.5,-1,
    -1,0,1,1,1,1,0,-1,
    -1,1,1,1,1,1,1,-1,
    -1,0.5,0,0,0,0,0.5,-1,
    -2,-1,-1,-1,-1,-1,-1,-2
]
knight_mob_matrix = [
    -5,-4,-3,-3,-3,-3,-4,-5,
    -4,-2,0,0,0,0,-2,-4,
    -3,0,1,1.5,1.5,1,0,-3,
    -3,0.5,1.5,2,2,1.5,0.5,-3,
    -3,0,1.5,2,2,1.5,0,-3,
    -3,0.5,1,1.5,1.5,1,0.5,-3,
    -4,-2,0,0.5,0.5,0,-2,-4,
    -5,-4,-3,-3,-3,-3,-4,-5
]
rook_mob_matrix = [
    0,0,0,0,0,0,0,0,
    0.5,1,1,1,1,1,1,0.5,
    -0.5,0,0,0,0,0,0,-0.5,
    -0.5,0,0,0,0,0,0,-0.5,
    -0.5,0,0,0,0,0,0,-0.5,
    -0.5,0,0,0,0,0,0,-0.5,
    -0.5,0,0,0,0,0,0,-0.5,
    0,0,0,0.5,0.5,0,0,0
]
pawn_mob_matrix = [
    -2,-1,-1,-1,-1,-1,-1,-2,
    -1,0,0,0,0,0,0,-1,
    -1,0,0.5,1,1,0.5,0,-1,
    -1,0.5,0.5,1,1,0.5,0.5,-1,
    -1,0,1,1,1,1,0,-1,
    -1,1,1,1,1,1,1,-1,
    -1,0.5,0,0,0,0,0.5,-1,
    -2,-1,-1,-1,-1,-1,-1,-2
]

mob_matrix = [pawn_mob_matrix, knight_mob_matrix,  bishop_mob_matrix, rook_mob_matrix,  queen_mob_matrix, king_mob_matrix]

def evaluation(board : chess.Board) : 
    score1 = 0
    score2 = 0

    moves = board.generate_pseudo_legal_moves()
    PieceMap = board.piece_map(mask = chess.BB_ALL)
    print("PieceMap", PieceMap)

    for key in PieceMap:
        square_number = key
        piece = PieceMap[key]
        
        if piece.color :
            index = 56 - 8*(int(square_number/8)) + square_number%8
            ##print("index", index)
            computing_matrix = mob_matrix[piece.piece_type - 1]     
            score2 += computing_matrix[index] * eval_matrix[piece.piece_type - 1]
            ##print("score2White", score2)  
        else :
            square_number = 63 - key
            index = 56 - 8*(int(square_number/8)) + square_number%8
            ##print("index", index)
            computing_matrix = mob_matrix[piece.piece_type - 1]      
            score2 -= computing_matrix[index] * eval_matrix[piece.piece_type - 1] 
            ##print("score2Black", score2)
    ##print("score2", score2)
        
    '''
    for move in moves :
        piece = board.piece_at(move.from_square)
        for
        score += [-1, 1][piece.color] * eval_matrix[piece.piece_type - 1]
    '''
    
    score1 += 200 * board.is_attacked_by(chess.WHITE, board.king(chess.BLACK))
    score1 -= 200 * board.is_attacked_by(chess.BLACK, board.king(chess.WHITE))
    
    
    return score1 + score2

def Minimax(board, isMaximizing = True, alpha = -math.inf, beta = math.inf, depth = 5) :
    if depth == 0 : 
        return [], evaluation(board)
    
    if board.is_checkmate() : 
        return [], -1000
    if board.is_stalemate() : 
        return [], -400
    
    best_move, best_score = [], [-1000, 1000][not isMaximizing]

    for move in board.legal_moves :
        board.push(move)

        bm, bs = Minimax(board, not isMaximizing, alpha, beta, depth - 1)

        board.pop()

        if (isMaximizing and best_score < bs) or (not isMaximizing and best_score > bs) :
            best_move = [move] + bm
            best_score = bs 
        
        if (isMaximizing and best_score > alpha) :
            alpha = best_score
        if (not isMaximizing and best_score < beta) :
            beta = best_score

        if (isMaximizing and best_score >= beta) or (not isMaximizing and best_score <= alpha) :
            break     

    return best_move, best_score

# current_state = UI.np_board, UI.white_pieces, UI.black_pieces, UI.castleRights
# print("Evaluation = ", evaluation(current_state))

board = chess.Board('r2bkqn1/2p5/4p3/8/8/4P3/2P5/R2B1KN1 w - - 0 1')
##board = chess.Board()
start = time.time()
score = evaluation(board)
print("score",score)

best_move, best_score = Minimax(board, depth = 3) 
end = time.time()
print("best_move, score, time = ", best_move, best_score, end - start)
'''
print(end - start)
start = time.time()
print("free board moves = ", UI.getFreeBoardLegalMoves(UI.np_board, UI.white_pieces, UI.black_pieces, 1))
end = time.time()
print(end - start)
# print(best_move, best_score, end - start)
'''