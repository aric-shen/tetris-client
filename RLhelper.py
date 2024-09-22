from game import Board
from piece import Piece
from config import SPAWNLOC
import copy
import math
import numpy as np

encoding = []
cnt = 0
for i in range(7):
    encoding.append([])
    if(i == 3):
        next = 1
    elif i in [1, 2, 5]:
        next = 4
    else:
        next = 2
    for j in range(next):
        encoding[i].append([])
        if(i == 0 and j == 0):
            rows = 24
            cols = 7
        elif(i == 0 and j == 1):
            rows = 21
            cols = 10
        elif i in [1, 2, 5]:
            if j in [0, 2]:
                rows = 23
                cols = 8
            else:
                rows = 22
                cols = 9
        elif(i == 3):
            rows  = 23
            cols = 9
        elif i in [4, 6]:
            if(j == 0):
                cols = 8
                rows = 23
            else:
                cols = 9
                rows = 22
        else:
            print("error")
        for k in range(rows):
            encoding[i][j].append([])
            for l in range(cols):
                encoding[i][j][k].append(cnt)
                cnt += 1
print(cnt)

offsets = []
for i in range(7):
    offsets.append([])
    if(i == 3):
        next = 1
    elif i in [1, 2, 5]:
        next = 4
    else:
        next = 2
    for j in range(next):
        offsets[i].append([])
        if(i == 0 and j == 0):
            offsets[i][j] = (-1, 0)
        elif(i == 0 and j == 1):
            offsets[i][j] = (0, -2)
        elif(i in [1, 2, 4, 5, 6] and j == 0):
            offsets[i][j] = (-1, 0)
        elif(i in [1, 2, 4, 5, 6] and j == 1):
            offsets[i][j] = (0, -1)
        elif(i in [1, 2, 5] and j in [2, 3]):
            offsets[i][j] = (-1, -1)
        elif(i == 3 and j == 0):
            offsets[i][j] = (0, 0)
'''
sum = 0
for i in range(len(encoding)):
    print("num: " + str(i))
    for j in range(len(encoding[i])):
        print("state: " + str(j))
        print(np.shape(encoding[i][j]))
        sum += np.shape(encoding[i][j])[0]*np.shape(encoding[i][j])[1]
print(sum)
'''
#print(encoding[6])
    

def getValidActions(board):
    #sets of center, rotationstate tuples that define a piece's position
    explored = set()
    valid = set()
    validNums = []

    piece = copy.deepcopy(board.curPiece)
    GVARecurse(piece, explored, valid)

    for i in valid:
        n = actionsToNum(i, piece.num)
        if n != None:
            validNums.append(n)
    return validNums

def getNaiveChoice(board, validActions):
    global oldFlatness
    global oldMaxHeight
    saveOldF = oldFlatness
    saveOldM = oldMaxHeight

    maxReward = -100000
    chosen = False
    choice = -1
    for i in range(len(validActions)):
        newB = copy.deepcopy(board)
        action = numToAction(validActions[i])
        success, reward = placePiece(newB, action[1], action[2], action[3], action[4])

        oldFlatness = saveOldF
        oldMaxHeight = saveOldM
        if(success):
            if(reward > maxReward):
                maxReward = reward
                choice = validActions[i]
                chosen = True
    return chosen, choice
    


def GVARecurse(piece, explored, valid):
    #check if piece is already explored
    if (piece.center[0], piece.center[1], piece.state) in explored:
        return

    explored.add((piece.center[0], piece.center[1], piece.state))
    save = Piece(None, None, piece)
    if(piece.down()):
        GVARecurse(piece, explored, valid)
        piece = Piece(None, None, save)
    else:
        if piece.num in [0, 4, 6] and piece.state > 1:
            if (piece.num == 0):
                if(piece.state == 2):
                    valid.add((piece.center[0] - 1, piece.center[1], piece.state%2))
                else:
                    valid.add((piece.center[0], piece.center[1] + 1, piece.state%2))
            else:
                if(piece.state == 2):
                    valid.add((piece.center[0], piece.center[1] - 1, piece.state%2))
                else:
                    valid.add((piece.center[0] - 1, piece.center[1], piece.state%2))
        else:
            valid.add((piece.center[0], piece.center[1], piece.state))
    
    if(piece.left()):
        GVARecurse(piece, explored, valid)
        piece = Piece(None, None, save)
    if(piece.right()):
        GVARecurse(piece, explored, valid)
        piece = Piece(None, None, save)
    if(piece.cw()):
        GVARecurse(piece, explored, valid)
        piece = Piece(None, None, save)
    if(piece.ccw()):
        GVARecurse(piece, explored, valid)
        piece = Piece(None, None, save)


def actionsToNum(action, pieceNum):
    x = action[0] + offsets[pieceNum][action[2]][0]
    y = action[1] + offsets[pieceNum][action[2]][1]
    if (y > len(encoding[pieceNum][action[2]])):
        return None
    if (x > len(encoding[pieceNum][action[2]][y])):
        return None
    return encoding[pieceNum][action[2]][y][x]

def numToAction(num):
    pieceNum = 6
    for i in range(len(encoding)):
        if(encoding[i][0][0][0] > num):
            pieceNum = i - 1
            break
    
    rotation = len(encoding[pieceNum]) - 1
    for i in range(len(encoding[pieceNum])):
        if(encoding[pieceNum][i][0][0] > num):
            rotation = i - 1
            break
    
    offset = offsets[pieceNum][rotation]
    
    row = len(encoding[pieceNum][rotation]) - 1
    for i in range(len(encoding[pieceNum][rotation])):
        if(encoding[pieceNum][rotation][i][0] > num):
            row = i - 1
            break
    
    col = len(encoding[pieceNum][rotation][row]) - 1
    for i in range(len(encoding[pieceNum][rotation][row])):
        if(encoding[pieceNum][rotation][row][i] == num):
            col = i
            break
    return (encoding[pieceNum][rotation][row][col], pieceNum, rotation, row - offset[1], col - offset[0])
oldFlatness = 10
oldMaxHeight = 0
def reset():
    global oldFlatness
    global oldMaxHeight
    oldFlatness = 10
    oldMaxHeight = 0
def placePiece(board, pieceNum, rotation, row, col):
    #flatness of a new game with no pieces placed
    global oldFlatness
    global oldMaxHeight
    board.curPiece = Piece(pieceNum, board.boardArr)
    board.curPiece.center = SPAWNLOC
    success = True
    match rotation:
        case 0:
            pass
        case 1:
            success = board.curPiece.cw() and success
        case 2:
            success = board.curPiece.rotate180() and success
        case 3:
            success = board.curPiece.ccw() and success
    board.curPiece.center = [col, row]
    if(board.curPiece.tryMove(board.curPiece) == False):
            success = False
    if(success):
        board.place()
    flatness = getFlatness(board)
    height = getMaxHeight(board)

    #print(board.lastScore)
    #print((flatness - oldFlatness))
    #print((height-oldMaxHeight)/4)
    reward = board.lastScore * 10 + (flatness - oldFlatness) - (height - oldMaxHeight)/4
    oldFlatness = flatness
    oldMaxHeight = height

    return success, reward

def getMaxHeight(board):
    max = 0
    for i in range(len(board.boardArr)):
        for j in range(max, len(board.boardArr[0])):
            if (board.boardArr[i][j]):
                max = j+1
    return max
def getFlatness(board):
    heights = []
    for i in range(len(board.boardArr)):
        height = 0
        for j in range(len(board.boardArr[0])):
            if(board.boardArr[i][j]):
                height = j
        heights.append(height)
    minCol = 0
    minHeight = 24
    for i in range(len(heights)):
        if (heights[i] < minHeight):
            minCol = i
            minHeight = heights[i]
    heights.pop(i)

    diffs = []
    totalDiff = 0
    for i in range(len(heights) - 1):
        diffs.append(abs(heights[i] - heights[i+1]))
        totalDiff += diffs[i]
    
    totalDiff /= 10
    return 10 - totalDiff - getHoles(board)

def getHoles(board):
    numHoles = 0
    for i in range(len(board.boardArr)):
        curState = True
        firstSwitch = False
        for j in range(len(board.boardArr[0])):
            if (curState != bool(board.boardArr[i][j])):
                numHoles += 1
                curState = bool(board.boardArr[i][j])
        numHoles -= 1
    return numHoles

