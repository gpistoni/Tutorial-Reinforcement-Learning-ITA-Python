import numpy as np
import random
import secrets                      # This module is responsible for providing access to the most secure source of randomness
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

class Tris:
        
     # Inizializza variabili del gioco1
    def __init__(self):
        self.nrows = 3
        self.ncols = 3
        self.nTris = 3

        self.players = ['X', 'O']   # Due giocatori contrassegnano una casella con X oppure O. Questo punto riguarda la parte grafica
        self.run = 0
        self.reset()                 # Chiama il metodo reset per inizializzare lo stato del gioco

    # Questo metodo reimposta la tastiera di gioco, il giocatore corrente, il vincitore e
    # lo stato di fine gioco ai loro valori iniziali.
    def reset(self):
        self.board = np.zeros( (self.ncols, self.nrows))
        self.current_player = random.choice(self.players)       # Sceglie casualmente chi inizia
        self.winner = None
        self.game_over = False
        self.init_board = False

    def GetBoardHash(self):
        state = self.board
        boardHash = str(state.reshape(3 * 3)) # Ricorda : c'è bisogno che stato attuale della griglia sia hashable e non un 'numpy.ndarray'
        return boardHash
    
####################################################################################################################################
    def checkBoard(self, found, val, c, r):
        if (r<0) or (r>=self.nrows) or (c<0) or (c>=self.ncols):
            return False                
        return found and val == self.board[c][r]
    
####################################################################################################################################
    def available_moves(self):
        moves = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.board[c,r] == 0:
                    moves.append((c, r))

        return moves                        # ritorna la lista dei possibli movimenti, cioè dove le caselle sono libere ed hanno valore 0
    
####################################################################################################################################
    def make_move(self, move):              # move è una tupla con le cooridnate della casella in cui vuole effettuata la mossa.
        if self.board[move[0]][move[1]] != 0:
            return False                        # Se la casella è già occupata, restituisce False, indicando che la mossa non è valida

        #Aggiorna il tabellone con il simbolo del giocatore attuale,
        #controlla se la mossa ha portato a una vittoria
        #passa al turno dell'altro giocatore.
        self.board[move[0]][move[1]] = self.players.index(self.current_player) + 1
        # Note:  self.players.index("X") = 0 mentre self.players.index("O") = 1

        self.check_winner()
        self.switch_player()
        
        if self.run % 100 == 1:
            #print("Mossa scelta : " , move)
            self.step_board_image(0.2)            
        return True

####################################################################################################################################
    def switch_player(self):
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]

####################################################################################################################################
    def check_winner(self):
        # Controllo riga
        for r in range(self.nrows):
            for c in range(self.ncols):         
                val = self.board[c][r]
                if (val!=0):
                    found = True # controllo riga                   
                    for tc in range(self.nTris):  
                        found = self.checkBoard(found, val, c + tc , r)
                    if (not found):
                        found = True     # controllo colonna                   
                        for tr in range(self.nTris):  
                           found = self.checkBoard(found, val, c  , r + tr)
                        if (not found):
                            found = True  # controllo diagonale positiva                          
                            for td in range(self.nTris):  
                                found = self.checkBoard(found, val, c + td  , r + td)
                            if (not found):
                                found = True     # controllo diagonale negativa                           
                                for td in range(self.nTris):
                                    found = self.checkBoard(found, val, c + td  , r - td)                                

                    if found:
                        #Se c'è un vincitore, imposta di conseguenza il vincitore e lo stato di game over.
                        self.winner = self.players[int(val - 1)]
                        self.game_over = True 
                        #self.print_board()    
                        return self.winner       

####################################################################################################################################
    #Output grafico della griglia
    #Questo metodo stampa lo stato corrente della griglia
    def print_board(self):
        print("-------------")
        for r in range(self.nrows):
            print("|", end=' ')
            for c in range(self.ncols):
                print(self.players[int(self.board[c][r] - 1)] if self.board[c][r] != 0 else " ", end=' | ')
            print()
            print("-------------")

   
    #####################################################################################################################
    def init_board_image(self, delay, line_width: int = 2,
                        bg_color=(255,255,255), line_color=(0,0,0), text_color=(0,0,0),
                         font_size=20) -> Image.Image:
        
        cell = 20
        sizeX = self.ncols * cell
        sizeY = self.nrows * cell

        self.img  = Image.new("RGB", (sizeX, sizeY), bg_color)
        draw = ImageDraw.Draw(self.img )
        font = ImageFont.truetype("arial.ttf", 20)  # o None per default
        
        for c in range(1, self.ncols):
            x = int(round(c * cell))
            draw.line([(x,0),(x,sizeY)], fill=line_color, width=line_width)
        for r in range(1, self.nrows):
            y = int(round(r * cell))
            draw.line([(0,y),(sizeX,y)], fill=line_color, width=line_width)

        # Mostra a video (apre il viewer di sistema)
        # img.show()
        plt.figure("Board")
        plt.clf()
        plt.imshow(self.img)
        plt.axis('off')
        plt.pause(delay)  # forza aggiornamento

    
    ####################################################################################################################################
    def step_board_image(self, delay, line_width: int = 2,
                     bg_color=(255,255,255), line_color=(0,0,0), text_color=(0,0,0),
                      font_size=20) -> Image.Image:
        
        if not self.init_board:
            self.init_board_image(0.01)
            self.init_board = True

        cell = 20
        sizeX = self.ncols * cell
        sizeY = self.nrows * cell

        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype("arial.ttf", 20)  # o None per default

        for r in range(self.nrows):
            for c in range(self.ncols):
                val = self.board[c, r]                
                text = ""
                if (val > 0):
                    text = self.players[int(val - 1)]  
                w, h = (15,20)
                cx = int((c + 0.5) * cell)
                cy = int((r + 0.5) * cell)
                tx = cx - w/2
                ty = cy - h/2
                draw.text((tx, ty), text, fill=text_color, font=font)

        if self.winner == self.players[0]:
            color = (60, 255, 60)             # light green
            draw.rectangle([(0,0),(sizeX,sizeY)], outline=color, fill=None, width=3)
            delay = 0.3
        elif self.winner == self.players[1]:
            color = (255, 60, 60)             # light blue
            draw.rectangle([(0,0),(sizeX,sizeY)],  outline=color, fill=None, width=3)
            delay = 0.3

        #plt.figure("Board")
        #plt.clf()
        plt.imshow(self.img)
        plt.pause(delay)  # forza aggiornamento
    
    #####################################################################################################################
    def test():
        game = Tris()
        game.print_board()

        while (not game.game_over) and (bool(game.available_moves())) :
            # fin quando non ci sono vincitori : (not game.game_over) == True
            # e
            # fin quando non ci sono azioni possibli : (bool(game.available_moves())) == True
            # Note : bool([]) == False
            # stai nel loop ...
            print("Azioni possibili, espresse in coordinate: " , game.available_moves()  )

            if game.current_player == game.players[0] :
                move = input(f"{game.current_player} è il tuo turno. Inserisci riga e colonna (e.g. 0 0): ")
                move = tuple(map(int, move.split()))
                # come lavora map()
                while move not in game.available_moves():
                    move = input("Mossa non valida, riprova: ")
                    move = tuple(map(int, move.split()))
            else:
                move = random.choice(game.available_moves())

            print("Mossa scelta : " , move)

            game.make_move(move)            


        if game.winner:
            print(f"{game.winner} Vince!")
        else:
            print("Pareggio!")

    #####################################################################################################################
    def test_rand(self):
        self.run += 1
        self.reset()
      
        while (not self.game_over) and (bool(self.available_moves())) :

            #print("Azioni possibili, espresse in coordinate: " , game.available_moves()  )
            move = random.choice(self.available_moves())
            self.make_move(move)
            

        if self.winner:
            print(f"{self.winner} Vince!")
        else:
            print("Pareggio!")

        return self.winner


# Random game test
if  __name__ == '__main__':

    tris = Tris()
    winnerstat = { 'X': 0, 'O': 0, 'Pareggio': 0 }
    
    while tris.run < 1000:
        winner  = tris.test_rand()
        #conteggia le vittorie dei due giocatori
        if (winner == None): winner = 'Pareggio'
        winnerstat[winner] += 1
        print("Statistiche Randomie: ", winnerstat)

    print("Overall ", winnerstat['X'] / tris.run * 100, "% vittorie per X")



