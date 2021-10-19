Skip to content
Search or jump to…
Pulls
Issues
Marketplace
Explore
 
@Aryan-Satpathy 
Aryan-Satpathy
/
Chess-AI-For-AIFA-Topic-Representation
Public
2
20
Code
Issues
Pull requests
1
Actions
Projects
Wiki
Security
Insights
More
Chess-AI-For-AIFA-Topic-Representation/UI.py /
@Aryan-Satpathy
Aryan-Satpathy CoMpLeTe ReWoRk
Latest commit 1c5e375 9 minutes ago
 History
 1 contributor
609 lines (531 sloc)  23 KB
   
# Change 1

import cv2
import numpy as np
import time

fen_board = ''                                                          # FEN string

# Pieces
pawn = 0b0
rook = 0b1
knight = 0b10
bishop = 0b11
queen = 0b100
king = 0b101
empty = 0b110

# Colors
black = 0b0 << 3
white = 0b1 << 3

# Direction
bdirection = int(((black >> 3) - 1 / 2) * 2)                                              # Returns -1 for black == 0 and 1 for black == 1 << 3
wdirection = int(((white >> 3) - 1 / 2) * 2)                                              # Returns -1 for white == 0 and 1 for white == 1 << 3

# Numpy Board
np_board = np.ones((8, 8), dtype = np.uintc)

# Black And White Pieces
black_pieces = []
white_pieces = []

# Castle Rights
castleRights = []

# UI stuff
scale = 111
canvas = np.zeros((1000, 1800, 4), np.uint8)
UIpawn = [cv2.resize(cv2.imread(r'bpawn.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA), cv2.resize(cv2.imread(r'wpawn.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA)]
UIrook = [cv2.resize(cv2.imread(r'brook.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA), cv2.resize(cv2.imread(r'wrook.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA)]
UIknight = [cv2.resize(cv2.imread(r'bknight.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA), cv2.resize(cv2.imread(r'wknight.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA)]
UIbishop = [cv2.resize(cv2.imread(r'bbishop.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA), cv2.resize(cv2.imread(r'wbishop.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA)]
UIqueen = [cv2.resize(cv2.imread(r'bqueen.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA), cv2.resize(cv2.imread(r'wqueen.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA)]
UIking = [cv2.resize(cv2.imread(r'bking.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA), cv2.resize(cv2.imread(r'wking.png', cv2.IMREAD_UNCHANGED), (scale, scale), interpolation=cv2.INTER_AREA)]
Colors = [(40, 40, 40, 255), (200, 200, 200, 255)]                                                                                                   # BGRA

print(UIpawn[0].shape)

# Functions
def getBoardFromFen(fen_board, np_board) :
    global castleRights
    current_pos = 0
    
    while current_pos < 64 : 
        if '1' <= fen_board[current_pos] <= '8' : 
            pass
        elif fen_board[current_pos] == '/' :
            pass
        else : 
            pass

######### Facing Some Difficulties With FEN, Imma Do It Later #########

def getPawnMoves(np_board, start, turn, LegalMoves, i = None, dMul = 1, for_check_board = False) : 
    dMul *= [-1, 1][turn == (black >> 3)]
    direction = dMul * bdirection
    step1 = False
    _start = [i, start][i == None]
    x, y = start
    if 0 <= (y + direction) < 8 and np_board[x, y + direction] == empty : 
        LegalMoves.append((_start, (x, y + direction)))
        step1 = True
    if 0 <= (x + 1) < 8 and 0 <= (y + direction) < 8 and np_board[x + 1, y + direction] != empty and (np_board[x + 1, y + direction] >> 3) != turn : 
        LegalMoves.append((_start, (x + 1, y + direction)))
    if 0 <= (x - 1) < 8 and 0 <= (y + direction) < 8 and np_board[x - 1, y + direction] != empty and (np_board[x - 1, y + direction] >> 3) != turn : 
        LegalMoves.append((_start, (x - 1, y + direction)))
    if step1 and (not for_check_board) and y == int(3.5 - 2.5 * direction) and np_board[x, y + 2 * direction] == empty :
        LegalMoves.append((_start, (x, y + 2 * direction)))

def getRookMoves(np_board, start, turn, LegalMoves, i = None) : 
    _start = [i, start][i == None]
    x, y = start
    for i in range(1, 8) :                                                  # Vertically Down
        if y + i >= 8 : 
            break
        if np_board[x, y + i] == empty:
            LegalMoves.append((_start, (x, y + i)))
        elif np_board[x, y + i] >> 3 != turn : 
            LegalMoves.append((_start, (x, y + i)))
            break
        else : 
            break
    for i in range(1, 8) :                                                  # Vertically Up
        if y - i < 0 : 
            break
        if np_board[x, y - i] == empty:
            LegalMoves.append((_start, (x, y - i)))
        elif np_board[x, y - i] >> 3 != turn : 
            LegalMoves.append((_start, (x, y - i)))
            break
        else : 
            break
    for i in range(1, 8) :                                                  # Horizontally Left
        if x - i < 0 : 
            break
        if np_board[x - i, y] == empty:
            LegalMoves.append((_start, (x - i, y)))
        elif np_board[x - i, y] >> 3 != turn : 
            LegalMoves.append((_start, (x - i, y)))
            break
        else : 
            break
    for i in range(1, 8) :                                                  # Horizontally Right
        if x + i >= 8 : 
            break
        if np_board[x + i, y] == empty:
            LegalMoves.append((_start, (x + i, y)))
        elif np_board[x + i, y] >> 3 != turn : 
            LegalMoves.append((_start, (x + i, y)))
            break
        else : 
            break

def getKnightMoves(np_board, start, turn, LegalMoves, i = None) : 
    _start = [i, start][i == None]
    x, y = start
    relPos = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
    for pos in relPos :
        delX, delY = pos
        if not (0 <= x + delX <= 7 and 0 <= y + delY <= 7) : 
            continue
        if (np_board[x + delX, y + delY] >> 3 != turn) or np_board[x + delX, y + delY] == empty : 
            LegalMoves.append((_start, (x + delX, y + delY)))

def getBishopMoves(np_board, start, turn, LegalMoves, i = None) : 
    _start = [i, start][i == None]
    x, y = start
    for i in range(1, 8) :                                                  # Bottom Right
        if max(x + i, y + i) > 7 : 
            break
        if np_board[x + i, y + i] == empty:
            LegalMoves.append((_start, (x + i, y + i)))
        elif np_board[x + i, y + i] >> 3 != turn : 
            LegalMoves.append((_start, (x + i, y + i)))
            break
        else : 
            break
    for i in range(1, 8) :                                                  # Top Right
        if x + i > 7 or y - i < 0 : 
            break
        if np_board[x + i, y - i] == empty:
            LegalMoves.append((_start, (x + i, y - i)))
        elif np_board[x + i, y - i] >> 3 != turn : 
            LegalMoves.append((_start, (x + i, y - i)))
            break
        else : 
            break
    for i in range(1, 8) :                                                  # Bottom Left
        if x - i < 0 or y + i > 7 : 
            break
        if np_board[x - i, y + i] == empty:
            LegalMoves.append((_start, (x - i, y + i)))
        elif np_board[x - i, y + i] >> 3 != turn : 
            LegalMoves.append((_start, (x - i, y + i)))
            break
        else : 
            break
    for i in range(1, 8) :                                                  # Top Left
        if min(x - i, y - i) < 0 : 
            break
        if np_board[x - i, y - i] == empty:
            LegalMoves.append((_start, (x - i, y - i)))
        elif np_board[x - i, y - i] >> 3 != turn : 
            LegalMoves.append((_start, (x - i, y - i)))
            break
        else : 
            break

def getQueenMoves(np_board, start, turn, LegalMoves, i = None) : 
    getBishopMoves(np_board, start, turn, LegalMoves, i)
    getRookMoves(np_board, start, turn, LegalMoves, i)

def getKingMoves(np_board, start, turn, LegalMoves, i = None) : 
    _start = [i, start][i == None]
    x, y = start
    relPos = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for pos in relPos : 
        delX, delY = pos
        if not (0 <= x + delX <= 7 and 0 <= y + delY <= 7) : 
            continue
        if (np_board[x + delX, y + delY] >> 3 != turn) or np_board[x + delX, y + delY] == empty : 
            LegalMoves.append((_start, (x + delX, y + delY)))

def getFreeBoardLegalMoves(np_board, white_pieces, black_pieces, turn) :
    LegalMoves = [] 
    if turn == (black >> 3) :                                                           # Black
        for i in range(len(black_pieces)) :
            start = black_pieces[i] 
            x, y = start
            if np_board[start] == black | pawn : 
                getPawnMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == black | rook : 
                getRookMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == black | knight :
                getKnightMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == black | bishop : 
                getBishopMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == black | queen : 
                getQueenMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == black | king :
                getKingMoves(np_board, start, turn, LegalMoves, i)
            else : 
                print('Invalid Piece {} {}, Imma Yeet'.format(['black', 'white'][0], np_board[start]))
    else :                                                                              # White
        for i in range(len(white_pieces)) : 
            start = white_pieces[i]
            x, y = start
            if np_board[start] == white | pawn : 
                getPawnMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == white | rook : 
                getRookMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == white | knight :
                getKnightMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == white | bishop : 
                getBishopMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == white | queen : 
                getQueenMoves(np_board, start, turn, LegalMoves, i)
            elif np_board[start] == white | king :
                getKingMoves(np_board, start, turn, LegalMoves, i)
            else : 
                print('Invalid Piece {} {}, Imma Yeet'.format(['black', 'white'][1], np_board[start]))

    return LegalMoves

def board_check(np_board, white_pieces, black_pieces, turn) : 
    # Idea is to check in a reverse way instead of checking all available moves of !turn (i.e, 1 - turn) and checking if turn king lies in any end position in the list
    # So we will do knight move generation from king and check if a !turn knight is in any of the end positions
    # We will do rook move generation from king and check if a !turn rook / queen is in any of the end positions
    # We will do bishop move generation from king and check if a !turn bishop / queen is in any of the end positions
    # We will do pawn move generation from king and check if a !turn pawn is in any of the end positions
    # We will do king move generation from king and check if !turn king is in any of the end positions
    # Now logically checking for king is logically not required, but we are using it to make sure legal moves generated are correct

    # First we find the position of king

    KingPos = None
    if turn == (black >> 3) : 
        for piece in black_pieces : 
            if np_board[piece] == (black | king) : 
                KingPos = piece
                break
        else : 
            print("Warning, king of color black not found!")
    else : 
        for piece in white_pieces : 
            if np_board[piece] == (white | king) : 
                KingPos = piece
                break
        else : 
            print("Warning, king of color white not found!")
    
    # Now that we have it, we check available moves with each type of pieces
    moves = []
    getRookMoves(np_board, KingPos, turn, moves)

    for move in moves : 
        if np_board[move[1]] in [((1 - turn) << 3) | rook, ((1 - turn) << 3) | queen] : 
            return True
    
    moves = []
    getBishopMoves(np_board, KingPos, turn, moves)

    for move in moves : 
        if np_board[move[1]] in [((1 - turn) << 3) | bishop, ((1 - turn) << 3) | queen] : 
            return True
        
    moves = []
    getKnightMoves(np_board, KingPos, turn, moves)

    for move in moves : 
        if np_board[move[1]] == (((1 - turn) << 3) | knight) : 
            return True

    moves = []
    getKingMoves(np_board, KingPos, turn, moves)

    for move in moves : 
        if np_board[move[1]] == (((1 - turn) << 3) | king) : 
            return True
    
    # Now here's the catch. The logic would work because all of these pieces, if can go from A to B, then can also come from B to A
    # But that's not true for Pawns.
    # So gotta do something different about it.

    # Pawn Code Here
    moves = []
    getPawnMoves(np_board, KingPos, turn, moves, -1)

    for move in moves : 
        if np_board[move[1]] == (((1 - turn) << 3) | pawn) : 
            return True

    return False

def getLegalMoves(np_board, white_pieces, black_pieces, turn) : 
    # We will check separately for Free Board Moves and Special Moves
    # Free Board Moves
    tempLegalMoves = getFreeBoardLegalMoves(np_board, white_pieces, black_pieces, turn)
    LegalMoves = []

    if turn == (black >> 3) : 
        pieces = black_pieces
        marker = 0
    else : 
        pieces = white_pieces
        marker = 1

    for move in tempLegalMoves : 
        i, end = move
        start = pieces[i]
        # We simulate the move to make sure that !turn (i.e 1 - turn) doesnt kill the king of turn side
        # This ensures that if turn king is in check, he mustn't be in check after the move
        
        # Simulation Code Here
        piece, piecef = np_board[start], np_board[end]
        np_board[start] = empty
        np_board[end] = piece
        
        if marker : 
            white_pieces[i] = end
        else :
            black_pieces[i] = end

        if not board_check(np_board, white_pieces, black_pieces, turn) : 
            # LegalMoves.append((start, end))
            LegalMoves.append(move)
        
        if marker : 
            white_pieces[i] = start
        else :
            black_pieces[i] = start
        
        np_board[start], np_board[end] = piece, piecef
    
    # Special Moves
    # Code Here

    return LegalMoves

def gridIndexToIRL(x, y) : 
    # print(scale)
    x, y = x - 4, y - 4
    x *= scale
    y *= scale
    return y + 500, x + 900

def overlayImage1(back, front) : 
    x, y = back.shape[ : -1]
    canvas = back.copy()
    for i in range(x) :
        for j in range(y) :
            canvas[i, j] = (front[i, j] * (front[i, j][-1] / 255) + canvas[i, j] * (1 - front[i, j][-1] / 255)) 
    return canvas

def overlayImage(back, front) : 
    front = front.astype(np.float)
    alphaChannel = (front[ : , : , -1]).copy()
    alphaChannel /= 255.0
    invAlphachannel = 1 - alphaChannel

    front = front * alphaChannel[ : , : , None]

    back = back.astype(np.float)

    back =  back * invAlphachannel[ : , : , None]

    canvas = front + back

    canvas =  canvas.astype(np.uint8)
    '''
    cv2.imshow('test', canvas)
    cv2.waitKey(50)
    '''
    return canvas

def generateBoard() :
    global canvas
    canvas = np.zeros((1000, 1800, 4), np.uint8)
    for x in range (8) :
        for y in range (8) :
            X, Y = gridIndexToIRL(x, y)
            Xf, Yf = gridIndexToIRL(x + 1, y + 1)
            canvas[X : Xf, Y : Yf] = np.reshape(np.array(Colors[(x + y) % 2] * (Xf - X) * (Yf - Y)), (Xf - X, Yf - Y, 4))

def _updateCanvas(np_board) : 
    global canvas
    start = time.time()
    canvas = np.zeros((1000, 1800, 4), np.uint8)
    for x in range (8) : 
        for y in range (8) :
            merge = True
            if np_board[x, y] == black | pawn : 
                img = UIpawn[0]
            elif np_board[x, y] == white | pawn : 
                img = UIpawn[1]
            elif np_board[x, y] == black | rook : 
                img = UIrook[0]
            elif np_board[x, y] == white | rook : 
                img = UIrook[1]
            elif np_board[x, y] == black | knight : 
                img = UIknight[0]
            elif np_board[x, y] == white | knight : 
                img = UIknight[1]
            elif np_board[x, y] == black | bishop : 
                img = UIbishop[0]
            elif np_board[x, y] == white | bishop : 
                img = UIbishop[1]
            elif np_board[x, y] == black | king : 
                img = UIking[0]
            elif np_board[x, y] == white | king : 
                img = UIking[1]
            elif np_board[x, y] == black | queen : 
                img = UIqueen[0]
            elif np_board[x, y] == white | queen : 
                img = UIqueen[1]
            else :
                # print('Unidentified Piece')
                merge = False
            X, Y = gridIndexToIRL(x, y)
            Xf, Yf = gridIndexToIRL(x + 1, y + 1)
            canvas[X : Xf, Y : Yf] = np.reshape(np.array(Colors[(x + y) % 2] * (Xf - X) * (Yf - Y)), (Xf - X, Yf - Y, 4))
            if merge : 
                # canvas[X : Xf, Y : Yf] = cv2.addWeighted(canvas[X : Xf, Y : Yf], 1, img, 1, 0.0)
                canvas[X : Xf, Y : Yf] = overlayImage(canvas[X : Xf, Y : Yf], img)
            # cv2.multiply(canvas, Colors)
    cv2.imshow('Chess Board', canvas)
    end = time.time()
    print('Time taken to make canvas : ', end - start)
    cv2.waitKey(0)
 
def placePiecesOnBoard(np_board) : 
    global canvas
    start = time.time()
    for x in range (8) : 
        for y in range (8) :
            merge = True
            if np_board[x, y] == black | pawn : 
                img = UIpawn[0]
            elif np_board[x, y] == white | pawn : 
                img = UIpawn[1]
            elif np_board[x, y] == black | rook : 
                img = UIrook[0]
            elif np_board[x, y] == white | rook : 
                img = UIrook[1]
            elif np_board[x, y] == black | knight : 
                img = UIknight[0]
            elif np_board[x, y] == white | knight : 
                img = UIknight[1]
            elif np_board[x, y] == black | bishop : 
                img = UIbishop[0]
            elif np_board[x, y] == white | bishop : 
                img = UIbishop[1]
            elif np_board[x, y] == black | king : 
                img = UIking[0]
            elif np_board[x, y] == white | king : 
                img = UIking[1]
            elif np_board[x, y] == black | queen : 
                img = UIqueen[0]
            elif np_board[x, y] == white | queen : 
                img = UIqueen[1]
            else :
                # print('Unidentified Piece')
                merge = False
            X, Y = gridIndexToIRL(x, y)
            Xf, Yf = gridIndexToIRL(x + 1, y + 1)
            if merge : 
                # canvas[X : Xf, Y : Yf] = cv2.addWeighted(canvas[X : Xf, Y : Yf], 1, img, 1, 0.0)
                canvas[X : Xf, Y : Yf] = overlayImage(canvas[X : Xf, Y : Yf], img)
    end = time.time()
    print('Time taken to place pieces on canvas : ', end - start)

def animatePieceOnBoard(np_board, initPos, finalPos) : 
    global canvas
    start = time.time()
    if np_board[initPos] == black | pawn : 
        img = UIpawn[0]
    elif np_board[initPos] == white | pawn : 
        img = UIpawn[1]
    elif np_board[initPos] == black | rook : 
        img = UIrook[0]
    elif np_board[initPos] == white | rook : 
        img = UIrook[1]
    elif np_board[initPos] == black | knight : 
        img = UIknight[0]
    elif np_board[initPos] == white | knight : 
        img = UIknight[1]
    elif np_board[initPos] == black | bishop : 
        img = UIbishop[0]
    elif np_board[initPos] == white | bishop : 
        img = UIbishop[1]
    elif np_board[initPos] == black | king : 
        img = UIking[0]
    elif np_board[initPos] == white | king : 
        img = UIking[1]
    elif np_board[initPos] == black | queen : 
        img = UIqueen[0]
    elif np_board[initPos] == white | queen : 
        img = UIqueen[1]
    else :
        print('Unidentified Piece')
    for x in range (8) : 
        for y in range (8) :
            merge = True
            if np_board[x, y] == black | pawn : 
                img = UIpawn[0]
            elif np_board[x, y] == white | pawn : 
                img = UIpawn[1]
            elif np_board[x, y] == black | rook : 
                img = UIrook[0]
            elif np_board[x, y] == white | rook : 
                img = UIrook[1]
            elif np_board[x, y] == black | knight : 
                img = UIknight[0]
            elif np_board[x, y] == white | knight : 
                img = UIknight[1]
            elif np_board[x, y] == black | bishop : 
                img = UIbishop[0]
            elif np_board[x, y] == white | bishop : 
                img = UIbishop[1]
            elif np_board[x, y] == black | king : 
                img = UIking[0]
            elif np_board[x, y] == white | king : 
                img = UIking[1]
            elif np_board[x, y] == black | queen : 
                img = UIqueen[0]
            elif np_board[x, y] == white | queen : 
                img = UIqueen[1]
            else :
                # print('Unidentified Piece')
                merge = False
            X, Y = gridIndexToIRL(x, y)
            Xf, Yf = gridIndexToIRL(x + 1, y + 1)
            if merge : 
                # canvas[X : Xf, Y : Yf] = cv2.addWeighted(canvas[X : Xf, Y : Yf], 1, img, 1, 0.0)
                canvas[X : Xf, Y : Yf] = overlayImage(canvas[X : Xf, Y : Yf], img)
    end = time.time()
    print('Time taken to animate piece on canvas : ', end - start)

canvas = cv2.imread('board.png', cv2.IMREAD_UNCHANGED)

if canvas is None : 
    start = time.time()
    generateBoard() 

    end = time.time()
    print('Time Taken = ', end - start)
    
    cv2.imshow('Board', canvas)
    
    cv2.waitKey(0)

    cv2.imwrite('board.png', canvas)
    print('writing')

# Board Arrangement
for x in range(8) :
    for y in range(8) :
        np_board[x, y] = empty

np_board[0, 0] = white | rook
np_board[3, 0] = white | bishop
np_board[5, 0] = white | king
np_board[6, 0] = white | knight
np_board[2, 1] = white | pawn
np_board[4, 2] = white | pawn
np_board[0, -1] = black | rook
np_board[3, -1] = black | bishop
np_board[4, -1] = black | king
np_board[5, -1] = black | queen
np_board[6, -1] = black | knight
np_board[2, -2] = black | pawn
np_board[4, -3] = black | pawn

black_pieces = [(0, 7), (3, 7), (4, 7), (5, 7), (6, 7), (2, 6), (4, 5)] 
white_pieces = [(0, 0), (3, 0), (5, 0), (6, 0), (2, 1), (4, 2)]

# white_pieces = [(0, 0)]

# np_board[4, 4] = white | pawn

LegalMoves = []
getRookMoves(np_board, (0, 0), white >> 3, LegalMoves)
print(LegalMoves)

print("Free board moves for white : {}".format(getFreeBoardLegalMoves(np_board, white_pieces, black_pieces, white >> 3)))
print("Free board moves for black : {}".format(getFreeBoardLegalMoves(np_board, white_pieces, black_pieces, black >> 3)))

print('Is white on check?', board_check(np_board, white_pieces, black_pieces, white >> 3))

print("Legal moves for white : {}".format(getLegalMoves(np_board, white_pieces, black_pieces, white >> 3)))

placePiecesOnBoard(np_board)

cv2.imshow('Board', canvas)
    
cv2.waitKey(0)

cv2.destroyAllWindows()
