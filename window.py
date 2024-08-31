import tkinter as tk
import sched, time
from tkinter import *
from game import Board
from config import windowTitle, BOARD_HEIGHT, BOARD_WIDTH, BLOCKSZ, GHOST, DAS, ARR, SDR, NEXT, COLORS
from piece import Piece

class Window:
    def __init__(self) -> None:
        self.boardState = Board()
        self.root = Tk()

        self.windowHeight = BOARD_HEIGHT * BLOCKSZ
        self.windowWidth = BOARD_WIDTH * BLOCKSZ

        self.board = Canvas(self.root, bg = 'black', height = self.windowHeight, width = self.windowWidth)
        self.board.config(state = NORMAL)
        self.board.grid(column = 1, row = 0)
        self.updateBoard()
        
        self.queue = Canvas(self.root, bg = 'black', height = self.windowHeight, width = self.windowWidth/2)
        self.queue.config(state = NORMAL)
        self.queue.grid(column = 2, row = 0)
        self.updateQueue()
        
        self.hold = Canvas(self.root, bg = 'black', height = self.windowHeight, width = self.windowWidth/2)
        self.hold.config(state = NORMAL)
        self.hold.grid(column = 0, row = 0)
        self.updateHold()


        self.root.bind('<KeyPress>', self.keyManager)
        self.root.bind('<KeyRelease>', self.keyReleaseManager)

        self.root.title(windowTitle)
        a = Label(self.root, text=windowTitle)

        #testing
        self.drawBg()
        self.drawGrid()
        self.drawQueue()
        self.updateBoard

        self.downKey = False
        self.leftKey = False
        self.rightKey = False

        self.lDas = False
        self.rDas = False


        self.root.mainloop()

    def keyManager(self, e):
        match e.keysym:
            case "Down":
                if(not self.downKey):
                    self.downKey = True
                    self.dRepeat()
                    self.retryDas()
            case "Left":
                if(not self.leftKey):
                    self.leftKey = True
                    self.boardState.curPiece.left()
                    self.dRepeat()
                    self.root.after(DAS, self.lRepeat)
            case "Right":
                if(not self.rightKey):
                    self.rightKey = True
                    self.boardState.curPiece.right()
                    self.dRepeat()
                    self.root.after(DAS, self.rRepeat)
            case "space":
                self.boardState.place()
                self.drawQueue()
                self.retryDas()
            case "Up":
                self.boardState.curPiece.cw()
                self.retryDas()
            case "z":
                self.boardState.curPiece.ccw()
                self.retryDas()
            case "x":
                self.boardState.curPiece.rotate180()
                self.retryDas()
            case "r":
                self.boardState = Board()
                self.drawQueue()
                self.drawHold()
            case "c":
                self.boardState.hold()
                self.retryDas()
                self.drawHold()
                self.drawQueue()
        self.dRepeat()
                
        if GHOST:
            self.boardState.updateGhost()
        
        
        self.drawGrid()

    def keyReleaseManager(self, e):
        match e.keysym:
            case "Down":
                self.downKey = False
            case "Left":
                self.leftKey = False
                self.lDas = False
            case "Right":
                self.rightKey = False
                self.rDas = False


    def drawGrid(self):
        self.board.delete("temp")

        # Draw board contents
        i = BOARD_HEIGHT - 1
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                if(self.boardState.boardArr[j][i] != 0):
                    self.drawSquare(j, i, self.board, self.boardState.boardArr[j][i] - 1, "temp")

        # Draw ghost piece
        if GHOST:
            cntr = self.boardState.ghostPiece.center
            for i in range(len(self.boardState.ghostPiece.shape)):
                self.drawSquare(self.boardState.ghostPiece.shape[i][0] + cntr[0], 
                    self.boardState.ghostPiece.shape[i][1] + cntr[1], self.board, 7, "temp")
    
        # Draw current piece
        cntr = self.boardState.curPiece.center
        for i in range(len(self.boardState.curPiece.shape)):
            self.drawSquare(self.boardState.curPiece.shape[i][0] + cntr[0], 
                self.boardState.curPiece.shape[i][1] + cntr[1], self.board, self.boardState.curPiece.num, "temp")

    
    def drawBg(self):
        # Draw the background grid, should always come before drawing squares
        for k in range (BOARD_HEIGHT + 1):
            self.board.create_rectangle(0, k * BLOCKSZ - 1, self.windowWidth + 2, 
                k * BLOCKSZ + 1, fill=COLORS[8], width = 0)
        for k in range (BOARD_WIDTH + 1):
            self.board.create_rectangle(k * BLOCKSZ + 3, 0, k * BLOCKSZ + 1, 
                self.windowHeight, fill=COLORS[8], width = 0)

    def drawSquare(self, i, j, canvas, num, tag):
        color = COLORS[num]
        x = i * BLOCKSZ + 2
        y = BOARD_HEIGHT * BLOCKSZ - j * BLOCKSZ
        canvas.create_rectangle(x, y, x + BLOCKSZ, y - BLOCKSZ,
        fill=color, width = 0, tags=tag)

    def updateBoard(self):
        self.board.update()
    def updateQueue(self):
        self.queue.update()
    def updateHold(self):
        self.hold.update()

    def retryDas(self):
        if(self.lDas):
            self.lRepeat()
        if(self.rDas):
            self.rRepeat()

    def lRepeat(self):
        if(self.leftKey):
            self.lDas = True
            if (self.boardState.curPiece.left()):
                self.dRepeat()
                if GHOST:
                    self.boardState.updateGhost()
                self.drawGrid()
                self.root.after(ARR, self.lRepeat)
    
    def rRepeat(self):
        if(self.rightKey):
            self.rDas = True
            if (self.boardState.curPiece.right()):
                self.dRepeat()
                if GHOST:
                    self.boardState.updateGhost()
                self.drawGrid()
                self.root.after(ARR, self.rRepeat)
    
    def dRepeat(self):
        if(self.downKey):
            if(self.boardState.curPiece.down()):
                self.drawGrid()
                self.root.after(SDR, self.dRepeat)
                
    '''Drawing the queue and hold pieces'''

    def drawQueue(self):
        self.queue.delete("queue")
        self.queueBoard = [[0 for i in range(BOARD_HEIGHT)] for j in range(BOARD_WIDTH)]
        for i in range(NEXT):
            queuePc = Piece(self.boardState.queue[i], self.queueBoard)
            queuePc.center = [2, BOARD_HEIGHT - 4 * (i + 1)]
            cntr = queuePc.center
            for i in range(len(queuePc.shape)):
                self.drawSquare(queuePc.shape[i][0] + cntr[0], 
                queuePc.shape[i][1] + cntr[1], self.queue, queuePc.num, "queue")

        self.queue.create_text(40, 450, text="SCORE: " + str(self.boardState.score), fill='white', tags="queue")
        if(self.boardState.finished):
            self.queue.create_text(40, 460, text="Finished", fill='white', tags="queue")
        self.updateQueue()

    def drawHold(self):
        self.hold.delete("hold")
        self.holdBoard = [[0 for i in range(BOARD_HEIGHT)] for j in range(BOARD_WIDTH)]
        if self.boardState.holdPiece is not None:
            holdPc = Piece(self.boardState.holdPiece, self.holdBoard)
            holdPc.center = [2, BOARD_HEIGHT - 4]
            cntr = holdPc.center
            for i in range(len(holdPc.shape)):
                self.drawSquare(holdPc.shape[i][0] + cntr[0], 
                holdPc.shape[i][1] + cntr[1], self.hold, holdPc.num, "hold")

