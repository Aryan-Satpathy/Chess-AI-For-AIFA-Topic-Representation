import time
import math
import chess
import random
import numpy as np

N = 0

king_mob_matrix = [
    -3,-4,-4,-5,-5,-4,-4,-3,
    -3,-4,-4,-5,-5,-4,-4,-3,
    -3,-4,-4,-5,-5,-4,-4,-3,
    -3,-4,-4,-5,-5,-4,-4,-3,
    -2,-3,-3,-4,-4,-3,-3,-2,
    -1,-2,-2,-2,-2,-2,-2,-1,
    2,2,0,0,0,0,2,2,
    2,3,2,0,0,1,3,2
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

# Phase Variables 
PawnPhase = 0
KnightPhase = 1
BishopPhase = 1
RookPhase = 2
QueenPhase = 4
TotalPhase = PawnPhase*16 + KnightPhase*4 + BishopPhase*4 + RookPhase*4 + QueenPhase*2

# Piece raw values 
p_mg_value, n_mg_value, b_mg_value, r_mg_value, q_mg_value = 82, 337, 365, 477, 1025
p_eg_value, n_eg_value, b_eg_value, r_eg_value, q_eg_value = 94, 281, 297, 512,  936

# Piece Tables 
mg_w_pawn_table = np.array([
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
])
mg_b_pawn_table = np.array([mg_w_pawn_table[2 * (i % 8) + 56 - i ] for i in range(64)])

eg_w_pawn_table = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
])
eg_b_pawn_table = np.array([eg_w_pawn_table[2 * (i % 8) + 56 - i ] for i in range(64)])

mg_w_knight_table = np.array([
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
])
mg_b_knight_table = np.array([mg_w_knight_table[2 * (i % 8) + 56 - i ] for i in range(64)])

eg_w_knight_table = np.array([
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
])
eg_b_knight_table = np.array([eg_w_knight_table[2 * (i % 8) + 56 - i ] for i in range(64)])

mg_w_bishop_table = np.array([
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
])
mg_b_bishop_table = np.array([mg_w_bishop_table[2 * (i % 8) + 56 - i ] for i in range(64)])

eg_w_bishop_table = np.array([
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
])
eg_b_bishop_table = np.array([eg_w_bishop_table[2 * (i % 8) + 56 - i ] for i in range(64)])

mg_w_rook_table = np.array([
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
])
mg_b_rook_table = np.array([mg_w_rook_table[2 * (i % 8) + 56 - i ] for i in range(64)])

eg_w_rook_table = np.array([
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
])
eg_b_rook_table = np.array([eg_w_rook_table[2 * (i % 8) + 56 - i ] for i in range(64)])

mg_w_queen_table = np.array([
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
])
mg_b_queen_table = np.array([mg_w_queen_table[2 * (i % 8) + 56 - i ] for i in range(64)])

eg_w_queen_table = np.array([
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
])
eg_b_queen_table = np.array([eg_w_queen_table[2 * (i % 8) + 56 - i ] for i in range(64)])

mg_w_king_table = np.array([
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
])
mg_b_king_table = np.array([mg_w_king_table[2 * (i % 8) + 56 - i ] for i in range(64)])

eg_w_king_table = np.array([
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
])
eg_b_king_table = np.array([eg_w_king_table[2 * (i % 8) + 56 - i ] for i in range(64)])


# p, kn, b, r, q, ki
mobility_matrix = [1, 2, 2.5, 2, 2.5, 0.5]
material_matrix = [1, 3, 3, 5, 9, 90]
complex_material_matrix = [pawn_mob_matrix, knight_mob_matrix, bishop_mob_matrix, rook_mob_matrix, queen_mob_matrix, king_mob_matrix]
attack_matrix = [5, 1, 1, 1, 1, 0.2]

TT = [[int(random.getrandbits(64)) for i in range(64)] for j in range(12)]
CTT = [int(random.getrandbits(64)) for i in range(4)]

def init_hash(board : chess.Board) -> int : 
    x = 0

    cstr = board.castling_xfen()

    if 'k' in cstr :
        x ^= CTT[0]
    if 'K' in cstr :
        x ^= CTT[1]
    if 'q' in cstr :
        x ^= CTT[2]
    if 'Q' in cstr :
        x ^= CTT[3]

    for i in range(64) : 
        piece = board.piece_at(i)
        if piece != None : 
            x ^= TT[piece.piece_type - 1 + 6 * piece.color][i]
    
    return x

def single_move_hash(move : chess.Move, init_board : chess.Board, init_hash : int) -> int :

    init_piece = board.piece_at(move.from_square)
    dest_piece = board.piece_at(move.to_square)
    f_type = move.promotion
    f_type = [init_piece.piece_type, f_type][f_type != None]

    init_hash ^= TT[init_piece.piece_type - 1 + 6 * init_piece.color][move.from_square] ^ TT[f_type - 1 + 6 * init_piece.color][move.to_square]

    if dest_piece != None : 
        if dest_piece.color == init_piece.color :
            if move.to_square > move.from_square : 
                init_hash ^= TT[chess.ROOK - 1 + 6 * init_piece.color][move.from_square + 1] ^ TT[chess.ROOK - 1 + 6 * init_piece.color][move.to_square + 1] ^ CTT[2 * init_piece.color]
            else : 
                init_hash ^= TT[chess.ROOK - 1 + 6 * init_piece.color][move.to_square + 1] ^ TT[chess.ROOK - 1 + 6 * init_piece.color][move.to_square - 1] ^ CTT[1 + 2 * init_piece.color]
        elif dest_piece.color != None :
            init_hash ^= TT[dest_piece.piece_type - 1 + 6 * dest_piece.color][move.to_square]
        
    return init_hash

Small_Lut = {}

# small_lut_size = 5000000
# Small_Lut = [None for i in range(small_lut_size)]

'''
def hash_hashkeys(hash_key : int) :
    returns a hash for the hash key, in the range of [0, small_lut_size), which can be directly used as index for Lut
    return (hash_key % small_lut_size)
'''
'''
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
'''
'''
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

TT_len = 0

def bb_mask(bb : int) -> np.array :
    x = (bb >> position_array).astype(np.uint8)
    x = np.unpackbits(x, bitorder = 'little')
    return x

def evaluation2_0(board : chess.Board, hash_ind : int) -> float :

    global N
    global TT_len
    N += 1


    if Small_Lut.get(hash_ind, None) != None :
        return Small_Lut[hash_ind]


    '''
    bpawn_mask = bb_mask(board.pawns & black)
    wpawn_mask = bb_mask(board.pawns & white)
    bknight_mask = bb_mask(board.knights & black)
    wknight_mask = bb_mask(board.knights & white)
    bbishop_mask = bb_mask(board.bishops & black)
    wbishop_mask = bb_mask(board.bishops & white)
    brook_mask = bb_mask(board.rooks & black)
    wrook_mask = bb_mask(board.rooks & white)
    bqueen_mask = bb_mask(board.queens & black)
    wqueen_mask = bb_mask(board.queens & white)
    bking_mask = bb_mask(board.kings & black)
    wking_mask = bb_mask(board.kings & white)
    '''

    pieces = board.piece_map()

    mg_material_value = 0
    eg_material_value = 0

    mg_protection_value = 0
    eg_protection_value = 0

    wp, bp, wn, bn, wb, bb, wr, br, wq, bq = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for pos in pieces :
        piece = pieces[pos]

        piece_type = piece.piece_type
        piece_color = piece.color

        if piece_type == chess.PAWN :
            if piece_color == chess.WHITE :
                # phase calculation
                wp += 1
                ## material
                # mg calculation
                mg_material_value += mg_w_pawn_table[pos] + p_mg_value
                # eg calculation
                eg_material_value += eg_w_pawn_table[pos] + p_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value += _protection_value * p_mg_value
                # eg calculation
                eg_protection_value += _protection_value * p_eg_value
            if piece_color == chess.BLACK :
                # phase calculation
                bp += 1
                # mg calculation
                mg_material_value -= mg_b_pawn_table[pos] + p_mg_value
                # eg calculation
                eg_material_value -= eg_b_pawn_table[pos] + p_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value -= _protection_value * p_mg_value
                # eg calculation
                eg_protection_value -= _protection_value * p_eg_value
        if piece_type == chess.KNIGHT :
            if piece_color == chess.WHITE :
                # phase calculation
                wn += 1
                # mg calculation
                mg_material_value += mg_w_knight_table[pos] + n_mg_value
                # eg calculation
                eg_material_value += eg_w_knight_table[pos] + n_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value += _protection_value * n_mg_value
                # eg calculation
                eg_protection_value += _protection_value * n_eg_value
            if piece_color == chess.BLACK :
                # phase calculation
                bn += 1
                # mg calculation
                mg_material_value -= mg_b_knight_table[pos] + n_mg_value
                # eg calculation
                eg_material_value -= eg_b_knight_table[pos] + n_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value -= _protection_value * n_mg_value
                # eg calculation
                eg_protection_value -= _protection_value * n_eg_value 
        if piece_type == chess.BISHOP :
            if piece_color == chess.WHITE :
                # phase calculation
                wb += 1
                # mg calculation
                mg_material_value += mg_w_bishop_table[pos] + b_mg_value
                # eg calculation
                eg_material_value += eg_w_bishop_table[pos] + b_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value += _protection_value * b_mg_value
                # eg calculation
                eg_protection_value += _protection_value * b_eg_value
            if piece_color == chess.BLACK :
                # phase calculation
                bb += 1
                # mg calculation
                mg_material_value -= mg_b_bishop_table[pos] + b_mg_value
                # eg calculation
                eg_material_value -= eg_b_bishop_table[pos] + b_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value -= _protection_value * b_mg_value
                # eg calculation
                eg_protection_value -= _protection_value * b_eg_value
        if piece_type == chess.ROOK :
            if piece_color == chess.WHITE :
                # phase calculation
                wr += 1
                # mg calculation
                mg_material_value += mg_w_rook_table[pos] + r_mg_value
                # eg calculation
                eg_material_value += eg_w_rook_table[pos] + r_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value += _protection_value * r_mg_value
                # eg calculation
                eg_protection_value += _protection_value * r_eg_value
            if piece_color == chess.BLACK :
                # phase calculation
                br += 1
                # mg calculation
                mg_material_value -= mg_b_rook_table[pos] + r_mg_value
                # eg calculation
                eg_material_value -= eg_b_rook_table[pos] + r_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value -= _protection_value * r_mg_value
                # eg calculation
                eg_protection_value -= _protection_value * r_eg_value
        if piece_type == chess.QUEEN :
            if piece_color == chess.WHITE :
                # phase calculation
                wq += 1
                # mg calculation
                mg_material_value += mg_w_queen_table[pos] + q_mg_value
                # eg calculation
                eg_material_value += eg_w_queen_table[pos] + q_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value += _protection_value * q_mg_value
                # eg calculation
                eg_protection_value += _protection_value * q_eg_value
            if piece_color == chess.BLACK :
                # phase calculation
                bq += 1
                # mg calculation
                mg_material_value -= mg_b_queen_table[pos] + q_mg_value
                # eg calculation
                eg_material_value -= eg_b_queen_table[pos] + q_eg_value
                ## protection
                attackers = board.attackers(not piece_color, pos)
                protectors = board.attackers(piece_color, pos)
                _protection_value = (1.5 * len(protectors) - len(attackers))
                # mg calculation
                mg_protection_value -= _protection_value * q_mg_value
                # eg calculation
                eg_protection_value -= _protection_value * q_eg_value


    '''
    wp, bp = bpawn_mask.sum(), wpawn_mask.sum()
    wn, bn = bknight_mask.sum(), wknight_mask.sum()
    wb, bb = bbishop_mask.sum(), wbishop_mask.sum()
    wr, br = brook_mask.sum(), wrook_mask.sum()
    wq, bq = bqueen_mask.sum(), wqueen_mask.sum()
    '''

    phase = TotalPhase

    phase -= (wp + bp) * PawnPhase  # Where wp is the number of white pawns currently on the board
    phase -= (wn + bn) * KnightPhase    # White knights
    phase -= (wb + bb) * BishopPhase
    phase -= (wr + br) * RookPhase
    phase -= (wq + bq) * QueenPhase

    phase = 256 * phase / TotalPhase

    '''
    #  mg_eval
    mg_material_value += -(bpawn_mask * mg_b_pawn_table).sum() + p_eg_value + (wpawn_mask * mg_w_pawn_table).sum() 
    mg_material_value += -(bknight_mask * mg_b_knight_table).sum() + (wknight_mask * mg_w_knight_table).sum()
    mg_material_value += -(bbishop_mask * mg_b_bishop_table).sum() + (wbishop_mask * mg_w_bishop_table).sum() 
    mg_material_value += -(brook_mask * mg_b_rook_table).sum() + (wrook_mask * mg_w_rook_table).sum() 
    mg_material_value += -(bqueen_mask * mg_b_queen_table).sum() + (wqueen_mask * mg_w_queen_table).sum() 
    mg_material_value += -(bking_mask * mg_b_king_table).sum() + (wking_mask * mg_w_king_table).sum() 
    #  eg_eval
    eg_material_value += -(bpawn_mask * eg_b_pawn_table).sum() + (wpawn_mask * eg_w_pawn_table).sum() 
    eg_material_value += -(bknight_mask * eg_b_knight_table).sum() + (wknight_mask * eg_w_knight_table).sum()
    eg_material_value += -(bbishop_mask * eg_b_bishop_table).sum() + (wbishop_mask * eg_w_bishop_table).sum() 
    eg_material_value += -(brook_mask * eg_b_rook_table).sum() + (wrook_mask * eg_w_rook_table).sum() 
    eg_material_value += -(bqueen_mask * eg_b_queen_table).sum() + (wqueen_mask * eg_w_queen_table).sum() 
    eg_material_value += -(bking_mask * eg_b_king_table).sum() + (wking_mask * eg_w_king_table).sum() 
    '''

    material_value = (mg_material_value * (256 - phase) + eg_material_value * phase) / 256

    material_value /= 1e4

    protection_value = (mg_protection_value * (256 - phase) + eg_protection_value * phase) / 256

    original = board.turn
    board.turn = chess.WHITE
    n_white = board.legal_moves.count() 
    board.turn = chess.BLACK
    n_black = board.legal_moves.count() 
    board.turn = original

    mobility_value = (n_white - n_black) / 4

    score = material_value * protection_value / 2e3 + mobility_value * protection_value / 5e6

    Small_Lut[hash_ind] = score
    TT_len += 1

    return score

def ordering_heuristics(board : chess.Board, moves : chess.LegalMoveGenerator) :
    L_moves = []
    L_evals = []
    i = 0

    for move in moves :
        i += 1
        score = 0
        mypiece = board.piece_at(move.from_square)
        enemypiece = board.piece_at(move.to_square)
        myColor = mypiece.color
        enemyColor = not myColor

        # Encourage promotion moves
        if move.promotion != None : 
            score += 25 * material_matrix[move.promotion - 1]
        
        # Encourage castles
        if mypiece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2 :
            score += 40

        # Captures evaluation (Encourage capturing high value piece with a low value piece)
        if enemypiece != None :
            score += 3 * (material_matrix[enemypiece.piece_type - 1] - material_matrix[mypiece.piece_type - 1])
        
        '''
        if mypiece.piece_type != chess.KING :
            # Attack on the piece
            attackers = board.attackers(enemyColor, move.to_square)
            for attacker in attackers :
                score -= 2.5 * attack_matrix[board.piece_type_at(attacker) - 1]
            # Protection of the piece
        
            attackers = board.attackers(myColor, move.to_square)
            for attacker in attackers :
                if attacker != move.from_square : 
                    score += 2.5 * attack_matrix[board.piece_type_at(attacker) - 1]
        '''

        L_moves.append(move)
        L_evals.append((i - 1), score / 500)
        # L.append((score / 500, move))
    
    L_evals.sort(key = lambda tup : tup[1], reverse=True)

    return i, L_moves, L_evals

def qsearch(board : chess.Board, board_hash : int, isMaximizing, alpha = -math.inf, beta = math.inf, depth = 3) :

    if depth == 0 :
        return [], evaluation2_0(board, board_hash)

    best_move, best_score = [], evaluation2_0(board, board_hash)
    
    moves = board.generate_legal_captures()

    l, moves, evals = ordering_heuristics(board, moves)

    i = l

    if l == 0 :
        return best_move, best_score

    for eval in evals :
        ind, waste = eval
        move = moves[ind]

        new_hash = single_move_hash(move, board, board_hash)

        board.push(move)

        '''
        if 2 * i > l :
            bm, bs = qsearch(board, not isMaximizing, alpha, beta, depth - 1)
        elif 3 * i > l :
            bm, bs = qsearch(board, not isMaximizing, alpha, beta, max(depth - 2, 0))
        else :
            bm, bs = qsearch(board, not isMaximizing, alpha, beta, max(depth - 3, 0))
        '''
        '''
        if 3 * i > l :
            bm, bs = qsearch(board, new_hash, not isMaximizing, alpha, beta, depth - 1)
        elif 5 * i > l :
            bm, bs = qsearch(board, new_hash, not isMaximizing, alpha, beta, max(depth - 2, 0))
        else :
            bm, bs = qsearch(board, new_hash, not isMaximizing, alpha, beta, max(depth - 3, 0))
        '''

        bm, bs = qsearch(board, new_hash, not isMaximizing, alpha, beta, depth - 1)
        bs += [-1, 1][isMaximizing] * waste

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
            
        i -= 1

    return best_move, best_score

def evaluation(board : chess.Board, hash_ind : int) : 
    global N
    global TT_len
    N += 1
    score = 0

    attack_score = 0
    protection_score = 0
    material_score = 0

    # hash_ind = init_hash(board)
    if Small_Lut.get(hash_ind, None) != None :
        return Small_Lut[hash_ind]
    

    '''
    moves = board.generate_pseudo_legal_moves()
    for move in moves :
        piece = board.piece_at(move.from_square)
        score += [-1, 1][piece.color] * mobility_matrix[piece.piece_type - 1]    
    board.turn = not board.turn
    moves = board.generate_pseudo_legal_moves()
    for move in moves :
        piece = board.piece_at(move.from_square)
        score += [-1, 1][piece.color] * mobility_matrix[piece.piece_type - 1]    
    
    board.turn = not board.turn
    '''

    pieces = board.piece_map()

    for pos in pieces :
        piece = pieces[pos]
        
        
        # Pieces that it attacks 
        attacks = board.attacks(pos)
        # attacks = board.generate_legal_captures()
        # len_attacks = sum(1 for x in attacks)
        
        for square in attacks :
            en = board.piece_at(square)
            if en == None :
                attack_score += [-1, 1][piece.color] * mobility_matrix[piece.piece_type - 1]
            elif en.color != piece.color :
                attack_score += 0.5 * [-1, 1][piece.color] * attack_matrix[piece.piece_type - 1] * (material_matrix[en.piece_type - 1] * complex_material_matrix[en.piece_type - 1][square] - material_matrix[piece.piece_type - 1] * complex_material_matrix[piece.piece_type - 1][pos])       
        
        # score += [-1, 1][piece.color] * mobility_matrix[piece.piece_type - 1] * len(attacks)
        
        # Material value of the piece based on position
        material_score += [-1, 1][piece.color] * material_matrix[piece.piece_type - 1] * complex_material_matrix[piece.piece_type - 1][[-1 - pos, pos][piece.color]]
        
        
        # Protection value of the piece
        if piece.piece_type != chess.KING :
            attackers = board.attackers(piece.color, pos)
            for attacker in attackers :
                attacker = board.piece_type_at(attacker)
                protection_score += [-1, 1][piece.color] * attack_matrix[attacker - 1] * material_matrix[piece.piece_type - 1] * complex_material_matrix[piece.piece_type - 1][[-1 - pos, pos][piece.color]]
            attackers = board.attackers(not piece.color, pos)
            for attacker in attackers :
                attacker = board.piece_type_at(attacker)
                protection_score -= [-1, 1][piece.color] * attack_matrix[attacker - 1] * material_matrix[piece.piece_type - 1] * complex_material_matrix[piece.piece_type - 1][[-1 - pos, pos][piece.color]]
        
        
        # board.
        # score += [-1, 1][piece.color] * len(board.attacks(pos)) * mobility_matrix[piece.piece_type - 1]

    # score += 200 * board.is_attacked_by(chess.WHITE, board.king(chess.BLACK))
    # score -= 200 * board.is_attacked_by(chess.BLACK, board.king(chess.WHITE))

    score = attack_score + material_score + protection_score

    Small_Lut[hash_ind] = score
    # Small_Lut[hash_ind] = (attack_score, material_score, protection_score)
    TT_len += 1
    
    return score
    # return (attack_score, material_score, protection_score)

def Minimax(board, board_hash : int, isMaximizing = True, alpha = -math.inf, beta = math.inf, depth = 6, move_sequence_from_previous_iteration = None) :
    if depth == 0 : 
        return qsearch(board, board_hash, isMaximizing, alpha, beta, 4)
    
    if board.is_checkmate() : 
        return [], -math.inf
    if board.is_stalemate() : 
        return [], -1e6
    
    best_move, best_score = [], [-math.inf, math.inf][not isMaximizing]
    
    moves = board.legal_moves
    l, moves, evals = ordering_heuristics(board, moves)

    i = l

    prev_iteration_best_move = None
    new_move_sequence = None
    if depth > 1 and move_sequence_from_previous_iteration is not None :
        prev_iteration_best_move = move_sequence_from_previous_iteration[0]
        
        waste, move = prev_iteration_best_move
        ind, move = move
        waste = evals[ind][1]
        
        new_hash = single_move_hash(move, board, board_hash)
        
        board.push(move)

        '''
        if 2 * i > l :
            bm, bs = Minimax(board, not isMaximizing, alpha, beta, (depth - 1))
        elif 3 * i > l :
            bm, bs = Minimax(board, not isMaximizing, alpha, beta, max(depth - 2, 0))
        else :
            bm, bs = Minimax(board, not isMaximizing, alpha, beta, max(depth - 3, 0))
        '''
        '''
        if 3 * i > l :
            bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, (depth - 1))
        elif 5 * i > l :
            bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, max(depth - 2, 0))
        else :
            bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, max(depth - 3, 0))
        '''

        new_move_sequence = move_sequence_from_previous_iteration[1 : ]
        bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, (depth - 1), new_move_sequence)
        bs += [-1, 1][isMaximizing] * waste

        board.pop()

        if (isMaximizing and best_score < bs) or (not isMaximizing and best_score > bs) :
            best_move = [move] + bm
            best_score = bs 
        
        if (isMaximizing and best_score > alpha) :
            alpha = best_score
        if (not isMaximizing and best_score < beta) :
            beta = best_score

        i -= 1


    for eval in evals :
        ind, waste = eval
        move = moves[ind]
        if prev_iteration_best_move != None and move[1] == prev_iteration_best_move[1] : 
            continue
        new_hash = single_move_hash(move, board, board_hash)
        
        board.push(move)

        '''
        if 2 * i > l :
            bm, bs = Minimax(board, not isMaximizing, alpha, beta, (depth - 1))
        elif 3 * i > l :
            bm, bs = Minimax(board, not isMaximizing, alpha, beta, max(depth - 2, 0))
        else :
            bm, bs = Minimax(board, not isMaximizing, alpha, beta, max(depth - 3, 0))
        '''
        '''
        if 3 * i > l :
            bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, (depth - 1))
        elif 5 * i > l :
            bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, max(depth - 2, 0))
        else :
            bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, max(depth - 3, 0))
        '''

        bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, (depth - 1), None)
        bs += [-1, 1][isMaximizing] * waste

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

        i -= 1

    return best_move, best_score

# current_state = UI.np_board, UI.white_pieces, UI.black_pieces, UI.castleRights
# print("Evaluation = ", evaluation(current_state))

# OG Fen : r2bkqn1/2p5/4p3/8/8/4P3/2P5/R2B1KN1 w - - 0 1
# interesting fen : rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4
board = chess.Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 1 5')
# board = chess.Board('r2bkqn1/2p5/4p3/8/8/4P3/2P5/R2B1KN1 w - - 0 1')
# board = chess.Board('1rbqkb1r/4pppp/2Bp4/p3P3/3NP3/2N5/PPP3PP/R1BQ1RK1 b k - 0 11')

black, white = board.occupied_co
position_array = np.arange(56, -1, -8, dtype=np.uint64)

# board.push_uci('b1c3')
# board.push_uci('a7a6')

board.push_uci('f1e2')
board.push_uci('b7b5')

print(board.fen())

hash = init_hash(board)

print("before : ")
print(board)
print('Evaluation : ', evaluation2_0(board, hash))


for i in range(1) : 
    if i % 2 == 0 : 
        Small_Lut.clear()
        start = time.time()
        # score = evaluation(board)
        move_sequence_from_previous_iteration = None
        for depth in range(1, 5) : 
            best_move, best_score = Minimax(board, board_hash = hash, isMaximizing = True, depth = depth, move_sequence_from_previous_iteration = move_sequence_from_previous_iteration) 
            move_sequence_from_previous_iteration = [(best_score, move) for move in best_move]
            print('Iteration {} over.'.format(depth))
        best_move, best_score = Minimax(board, board_hash = hash, isMaximizing = True, depth = 4, move_sequence_from_previous_iteration = move_sequence_from_previous_iteration)
        end = time.time()
        print("best_move, score, time = ", best_move, best_score, end - start)
        hash = single_move_hash(best_move[0], board, hash)
        board.push(best_move[0])
        print('After : ')
        print(board)
    else :
        response = input('Enter enemy move : ')
        hash = single_move_hash(chess.Move.from_uci(response), board, hash)
        move = board.push_uci(response)

print('After : ')
print(board)

print("number of positions evaluated : ", N)
print("number of transposition entries : ", TT_len)


'''
move = chess.Move.from_uci('c1g5')
hash = single_move_hash(move, board, hash)
board.push_uci('c1g5')
print('After')
print(board)
print('Evaluation : ', evaluation2_0(board, hash))
'''

'''
start = time.time()
for i in range(8000) :
    mask = bb_mask(board.pawns & white)
end = time.time()
print('Time taken = ', end - start, 'ms')
'''

'''
# print('current evaluation : ', evaluation(board, init_hash(board)))
start = time.time()
for i in range(1000) :
    l, moves = ordering_heuristics(board, board.legal_moves)
end = time.time()
# print('time : ', (end - start))
print('best move eval : ', moves)
'''

'''
l, moves = ordering_heuristics(board, board.legal_moves)
for move in moves :
    me, move = move
    board.push(move)
    # print('Board : ')
    # print(board)
    print(move, ' :')
    print('Move Eval, board eval : ', me, evaluation2_0(board, init_hash(board)))
    print('--------')
    board.pop()
# print('eval : ', evaluation(board, 0))
'''

'''
start = time.time()
for i in range(1) :
    n = evaluation2_0(board, init_hash)
end = time.time()
print("time taken, evaluation = ", (end - start), n)
'''

'''
print("Stages of Board : \n")
board.push_uci('c3b1')
print(board)
print('--------')
board.push_uci('e8d7')
print(board)
print('--------')
board.push_uci('f1d3')
print(board)
print('--------')
board.push_uci('g7g6')
print(board)
print('--------')
board.push_uci('d1f3')
print(board)
print('--------')
board.push_uci('f6e4')
print(board)
print('--------')
board.push_uci('f3f7')
print(board)
print('--------')
board.push_uci('e4f2')
print(board)
print('--------')
print()
'''

'''
start = time.time()
moves = ordering_heuristics(board, board.legal_moves)
end = time.time()
print('moves, time : ', moves, end - start)
'''

'''
print("after : ")
print(board)
'''

'''
print(end - start)
start = time.time()
print("free board moves = ", UI.getFreeBoardLegalMoves(UI.np_board, UI.white_pieces, UI.black_pieces, 1))
end = time.time()
print(end - start)
# print(best_move, best_score, end - start)
'''