from re import X
import time
import math
import chess
import random

N = 0

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

# p, kn, b, r, q, ki
mobility_matrix = [1, 2, 2.5, 2, 2.5, 0.5]
material_matrix = [1, 3, 3, 5, 7, 90]
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

def ordering_heuristics(board : chess.Board, moves : chess.LegalMoveGenerator) :
    L = []
    i = 0

    for move in moves :
        i += 1
        score = 0
        myColor = board.color_at(move.from_square)
        enemyColor = not myColor

        if move.promotion != None : 
            score += 50 * material_matrix[move.promotion - 1]
        
        if abs(move.from_square - move.to_square) in [2, 3] :
            if board.piece_type_at(move.from_square) == chess.KING :
                score += 15

        if board.color_at(move.to_square) == enemyColor :
            score += 5 * (material_matrix[board.piece_type_at(move.to_square) - 1] - material_matrix[board.piece_type_at(move.from_square) - 1])

        attackers = board.attackers(enemyColor, move.to_square)

        for attacker in attackers :
            score -= 2 * attack_matrix[board.piece_type_at(attacker) - 1]

        if board.piece_type_at(move.from_square) != chess.KING :
            attackers = board.attackers(myColor, move.to_square)

            for attacker in attackers :
                score += 2 * attack_matrix[board.piece_type_at(attacker) - 1]
        
        L.append((score, move))
    
    L.sort(key = lambda tup : tup[0], reverse=True)

    return i, L

def qsearch(board : chess.Board, board_hash : int, isMaximizing, alpha = -math.inf, beta = math.inf, depth = 3) :

    if depth == 0 :
        return [], evaluation(board, board_hash)

    best_move, best_score = [], evaluation(board, board_hash)
    
    moves = board.generate_legal_captures()

    l, moves = ordering_heuristics(board, moves)

    i = l

    if l == 0 :
        return best_move, best_score

    for move in moves :
        waste, move = move

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
        score += [-1, 1][piece.color] * mobility_matrix[piece.piece_type - 1] * len(board.attacks(pos))
        score += [-1, 1][piece.color] * material_matrix[piece.piece_type - 1] * complex_material_matrix[piece.piece_type - 1][[-1 - pos, pos][piece.color]]
        if piece.piece_type != chess.KING :
            attackers = board.attackers(piece.color, pos)
            for attacker in attackers :
                attacker = board.piece_type_at(attacker)
                score += attack_matrix[attacker - 1]
            attackers = board.attackers(not piece.color, pos)
            for attacker in attackers :
                attacker = board.piece_type_at(attacker)
                score -= attack_matrix[attacker - 1]
        # board.
        # score += [-1, 1][piece.color] * len(board.attacks(pos)) * mobility_matrix[piece.piece_type - 1]

    # score += 200 * board.is_attacked_by(chess.WHITE, board.king(chess.BLACK))
    # score -= 200 * board.is_attacked_by(chess.BLACK, board.king(chess.WHITE))

    Small_Lut[hash_ind] = score
    TT_len += 1
    
    return score

def Minimax(board, board_hash : int, isMaximizing = True, alpha = -math.inf, beta = math.inf, depth = 6) :
    if depth == 0 : 
        return qsearch(board, board_hash, isMaximizing, alpha, beta, 3)
    
    if board.is_checkmate() : 
        return [], -1000
    if board.is_stalemate() : 
        return [], -400
    
    best_move, best_score = [], [-1000, 1000][not isMaximizing]

    moves = board.legal_moves
    l, moves = ordering_heuristics(board, moves)

    i = l

    for move in moves :
        waste, move = move
        
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
        bm, bs = Minimax(board, new_hash, not isMaximizing, alpha, beta, (depth - 1))
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

# board.push_uci('b1c3')
# board.push_uci('a7a6')

board.push_uci('f1e2')
# board.push_uci('e7e6')

print("before : ")
print(board)

for i in range(1) : 
    # if board.turn : 
    start = time.time()
    # score = evaluation(board)
    best_move, best_score = Minimax(board, False, depth = 5) 
    end = time.time()
    print("best_move, score, time = ", best_move, best_score, end - start)
    board.push(best_move[0])
    # else :
    #     response = input('Enter enemy move : ')
    #     board.push_uci(response)

print('After : ')
print(board)

print("number of positions evaluated : ", N)
print("number of transposition entries : ", TT_len)


'''
l, moves = ordering_heuristics(board, board.legal_moves)
print('best move eval : ', moves)
'''

'''
start = time.time()
for i in range(1) :
    n = evaluation(board)
end = time.time()

print("time taken = ", (end - start))
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