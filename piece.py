from config import pieces, BOARD_HEIGHT, BOARD_WIDTH, SRSKicks as KICKS, SRSIKicks as IKICKS
import copy
class Piece:
    def __init__(self, num, boardArr, orig=None) -> None:
        if orig is not None:
            self.shape = copy.deepcopy(orig.shape)
            self.num = orig.num
            self.center = orig.center.copy()
            self.boardArr = orig.boardArr
            self.state = orig.state
        else:
            self.num = num
            self.shape = pieces[num]
            self.center = [0,0]
            self.boardArr = boardArr
            self.state = 0
    
    def tryMove(self, newPc):
        for i in range(len(newPc.shape)):
            x = newPc.shape[i][0] + newPc.center[0]
            y = newPc.shape[i][1] + newPc.center[1]

            if(x < 0 or x > BOARD_WIDTH - 1 or y < 0 or y > BOARD_HEIGHT - 1):
                return False
            if(self.boardArr[x][y]):
                return False
            
        self.shape = newPc.shape
        self.center = newPc.center
        self.state = newPc.state
        return True
    
    def down(self):
        newPiece = Piece(self.num, self.boardArr, self)
        newPiece.center[1] -= 1
        return self.tryMove(newPiece)

    def left(self):
        newPiece = Piece(self.num, self.boardArr, self)
        newPiece.center[0] -= 1
        return self.tryMove(newPiece)
    
    def right(self):
        newPiece = Piece(self.num, self.boardArr, self)
        newPiece.center[0] += 1
        return self.tryMove(newPiece)
    
    def rotate180(self):
        newPiece = Piece(self.num, self.boardArr, self)
        for x in newPiece.shape:
                tempX = x[0]
                tempY = x[1]
                x[0] = tempX * -1
                x[1] = tempY * -1
        if (self.tryMove(newPiece)):
            self.state = (self.state + 2) % 4
            return True
        return False
    
    def cw(self):
        if(self.num == 3):
            return False
        newPiece = Piece(self.num, self.boardArr, self)
        table = KICKS

        if(self.num == 0):
            table = IKICKS
            if(self.num == 0):
                if(self.state == 0):
                    newPiece.center[0] += 1
                if(self.state == 1):
                    newPiece.center[1] -= 1
                if(self.state == 2):                        
                    newPiece.center[0] -= 1
                if(self.state == 3):
                    newPiece.center[1] += 1
            center = copy.copy(newPiece.center)
            for x in newPiece.shape:
                tempX = x[0]
                tempY = x[1]
                x[0] = tempY 
                x[1] = tempX * -1

        else:
            for x in newPiece.shape:
                tempX = x[0]
                tempY = x[1]
                x[0] = tempY
                x[1] = tempX * -1
            center = copy.copy(newPiece.center)

        newPiece.state = (1 + self.state) % 4

        for kick in range(len(table[self.state])):
            newPiece.center[0] = center[0] + table[self.state][kick][0]
            newPiece.center[1] = center[1] + table[self.state][kick][1]
            if(self.tryMove(newPiece)):
                return True 
        return False
    
    def ccw(self):
        if(self.num == 3):
            return False
        newPiece = Piece(self.num, self.boardArr, self)
        table = KICKS

        if(self.num == 0):
            table = IKICKS
            if(self.num == 0):
                if(self.state == 0):
                    newPiece.center[1] -= 1
                if(self.state == 1):
                    newPiece.center[0] -= 1
                if(self.state == 2):                        
                    newPiece.center[1] += 1
                if(self.state == 3):
                    newPiece.center[0] += 1
            for x in newPiece.shape:
                tempX = x[0]
                tempY = x[1]
                x[0] = tempY * -1
                x[1] = tempX
        else:
            for x in newPiece.shape:
                tempX = x[0]
                tempY = x[1]
                x[0] = tempY * -1
                x[1] = tempX
        newPiece.state = (self.state - 1) % 4
        center = copy.copy(newPiece.center)

        for kick in range(len(table[self.state])):
            kickNum = (self.state - 1) % 4
            newPiece.center[0] = center[0] - table[kickNum][kick][0]
            newPiece.center[1] = center[1] - table[kickNum][kick][1]
            if(self.tryMove(newPiece)):
                return True
        return False
    
    def goDown(self):
        while(self.down()):
            pass


        

