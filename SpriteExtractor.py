import numpy as np
import cv2

canvas = cv2.imread(r'2000px-Chess_Pieces_Sprite.svg.png', cv2.IMREAD_UNCHANGED)

canvas = canvas[1 : , 1 : -1]

x = 333

names = ['wking.png', 'bking.png', 'wqueen.png', 'bqueen.png', 'wbishop.png', 'bbishop.png', 'wknight.png', 'bknight.png', 'wrook.png', 'brook.png', 'wpawn.png', 'bpawn.png']

for i in range(6) : 
    wimg = canvas[ : x, x * i : x * i + x]
    bimg = canvas[x : , x * i : x * i + x]
    cv2.imwrite(names[2 * i], wimg)
    cv2.imwrite(names[2 * i + 1], bimg)