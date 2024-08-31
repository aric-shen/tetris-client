from config import BOARD_HEIGHT, BOARD_WIDTH, SPAWNLOC, GHOST, SCORES
from piece import Piece
import random

class Board(object):
#initialize empty board, get pieces
    def __init__(self):
        self.boardArr = [[0 for i in range(BOARD_HEIGHT)] for j in range(BOARD_WIDTH)]

        self.queue = []
        self.curPiece = self.generatePiece()
        self.holdPiece = None
        self.score = 0

        self.finished = False
        self.pieceCount = 0

        if GHOST:
            self.ghostPiece = Piece(self.curPiece.num, self.boardArr, self.curPiece)
            self.ghostPiece.goDown()


# Pops and returns a piece number from the queue
    def generatePiece(self):
        while(len(self.queue) < 14):
            bag = [0,1,2,3,4,5,6]
            random.shuffle(bag)
            self.queue += bag
        pieceNum = self.queue[0]
        self.queue.pop(0)
        piece = Piece(pieceNum, self.boardArr)
        piece.center = SPAWNLOC

        #Check for loss condition
        if(piece.tryMove(piece) == False):
            self.finished = True
        return piece
    
    def updateGhost(self):
        self.ghostPiece = Piece(self.curPiece.num, self.boardArr, self.curPiece)
        self.ghostPiece.goDown()
    
#Places the current piece
    def place(self):
        self.curPiece.goDown()
        for i in range(len(self.curPiece.shape)):
            x = self.curPiece.shape[i][0] + self.curPiece.center[0]
            y = self.curPiece.shape[i][1] + self.curPiece.center[1]
            self.boardArr[x][y] = self.curPiece.num + 1
        self.clearLines()
        self.pieceCount += 1
        if(self.pieceCount > 100):
            self.finished = True
        self.curPiece = self.generatePiece()

#Clears lines, if necessary
    def clearLines(self):
        cnt = 0
        for i in reversed(range(BOARD_HEIGHT)):
            filled = True
            for j in range(BOARD_WIDTH):
                if(self.boardArr[j][i] == 0):
                    filled = False
                    break
            if (filled):
                cnt += 1
                for j in range(i, BOARD_HEIGHT - 1):
                    for k in range(BOARD_WIDTH):
                        self.boardArr[k][j] = self.boardArr[k][j+1]
        match cnt:
            case 1:
                self.score += SCORES[0]
            case 2:
                self.score += SCORES[1]
            case 3:
                self.score += SCORES[2]
            case 4:
                self.score += SCORES[3]
                
        return cnt
    
    def hold(self):
        if self.holdPiece is None:
            self.holdPiece = self.curPiece.num
            self.curPiece = self.generatePiece()
        else:
            tmp = self.holdPiece
            self.holdPiece = self.curPiece.num
            self.curPiece = Piece(tmp, self.boardArr)
            self.curPiece.center = SPAWNLOC

            
