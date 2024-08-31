BOARD_WIDTH = 10
BOARD_HEIGHT = 24
BORDER = 2
BLOCKSZ = 20

#DAS, ARR, SDF
DAS = 69
ARR = 0
SDR = 0

#Number of visible next pieces

NEXT = 5        

#Seven default pieces
IPiece = [[0, 0], [-1, 0], [1, 0], [2, 0]]
JPiece = [[0, 0], [-1, 0], [1, 0], [-1, 1]]
LPiece = [[0, 0], [-1, 0], [1, 0], [1, 1]]
OPiece = [[0, 0], [1, 0], [0, 1], [1, 1]]
SPiece = [[0, 0], [-1, 0], [0, 1], [1, 1]]
TPiece = [[0, 0], [-1, 0], [1, 0], [0, 1]]
ZPiece = [[0, 0], [1, 0], [0, 1], [-1, 1]]

pieces = [IPiece, JPiece, LPiece, OPiece, SPiece, TPiece, ZPiece]

# SRS Table

_0to1 = [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]]
_1to2 = [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]]
_2to3 = [[0, 0], [1, 0], [1, 1], [0, -2], [-1, -2]]
_3to0 = [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]]

SRSKicks = [_0to1, _1to2, _2to3, _3to0]

i0to1 = [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]]
i1to2 = [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]]
i2to3 = [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]]
i3to0 = [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]]

SRSIKicks = [i0to1, i1to2, i2to3, i3to0]
#Color codes for pieces
COLORS = ["#0f9bd8", "#2142c7", "#e35b03", "#e39e02", "#59b202", "#af298a", "#d70f36", "#a7a7a7", "#333333"]

#Optional starting queue before 7-bag


#Extras
GHOST = True

#Score information
#in order Single, Double, Triple, Tetris
SCORES = [1, 3, 5, 8]

#spawn location
SPAWNLOC = [4, 20]
windowTitle = "Aric's Tetris Client"