import numpy as np
import random
import secrets                      # This module is responsible for providing access to the most secure source of randomness
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


class Tris:
        
     # Inizializza variabili del gioco
    def __init__(self):
        self.nrows = 3
        self.ncols = 3
        self.nTris = 3

        self.players = ['X', 'O']   # Due giocatori contrassegnano una casella con X oppure O. Questo punto riguarda la parte grafica
        self.current_player = None
        self.reset()                 # Chiama il metodo reset per inizializzare lo stato del gioco

    # Questo metodo reimposta la tastiera di gioco, il giocatore corrente, il vincitore e
    # lo stato di fine gioco ai loro valori iniziali.
    def reset(self):
        self.board = np.zeros( (self.nrows + self.nTris , self.ncols + self.nTris ))
        self.winner = None
        self.game_over = False


    def available_moves(self):
        moves = []
        for i in range(self.ncols):
            for j in range(self.nrows):
                if self.board[i][j] == 0:
                    moves.append((i, j))

        return moves                        # ritorna la lista dei possibli movimenti, cioè dove le caselle sono libere ed hanno valore 0


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
        return True


    def switch_player(self):
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]

####################################################################################################################################
    def check_winner(self):
        # Controllo riga
        for i in range(self.nrows):     
            for j in range(self.ncols):
                val = self.board[i][j]
                if (val!=0):
                    found = True
                    # controllo riga
                    for tr in range(self.nTris):  
                        found = found and val == self.board[i+tr][j]
                    if (not found):
                        found = True
                        # controllo colonna
                        for tc in range(self.nTris):  
                            found = found and val == self.board[i][j+tc]

                    if found:
                            #Se c'è un vincitore, imposta di conseguenza
                            #il vincitore e
                            #lo stato di game over.
                            self.winner = self.players[int(val - 1)]
                            self.game_over = True            

        # Controllo diagonali positive e negative
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.winner = self.players[int(self.board[0][0] - 1)]
            self.game_over = True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.winner = self.players[int(self.board[0][2] - 1)]
            self.game_over = True

####################################################################################################################################
    #Output grafico della griglia
    #Questo metodo stampa lo stato corrente della griglia
    def print_board(self):
        print("-------------")
        for i in range(3):
            print("|", end=' ')
            for j in range(3):
                print(self.players[int(self.board[i][j] - 1)] if self.board[i][j] != 0 else " ", end=' | ')
            print()
            print("-------------")

####################################################################################################################################
    def draw_board_image(self, line_width: int = 2,
                     bg_color=(255,255,255), line_color=(0,0,0), text_color=(0,0,0),
                     font_path=None, font_size=20) -> Image.Image:
        
        cell = 100
        sizeX = self.ncols * cell
        sizeY = self.nrows * cell

        img = Image.new("RGB", (sizeX, sizeY), bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        for c in range(1, self.ncols):
            x = int(round(c * cell))
            draw.line([(x,0),(x,sizeY)], fill=line_color, width=line_width)
        for r in range(1, self.nrows):
            y = int(round(r * cell))
            draw.line([(0,y),(sizeX,y)], fill=line_color, width=line_width)

        for r in range(self.nrows):
            for c in range(self.ncols):
                val = self.board[r, c]
                
                font = ImageFont.truetype("arial.ttf", 50)  # o None per default
                text = ""
                if (val > 0):
                    text = self.players[int(val - 1)]  

                w, h = (40,50)
                cx = int((c + 0.5) * cell)
                cy = int((r + 0.5) * cell)
                tx = cx - w/2
                ty = cy - h/2
                draw.text((tx, ty), text, fill=text_color, font=font)

        # Mostra a video (apre il viewer di sistema)
        # img.show()
        plt.figure("Board")
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        plt.pause(0.001)  # forza aggiornamento

        return img
    
    #####################################################################################################################
    def test():
        game = Tris()
        game.current_player = game.players[0] # Imposta il giocatore corrente su X. Quindi sarà lui a far la prima mossa. Nota : game.players[0] == "X"
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
            game.print_board()
            game.draw_board_image()
            print(" ")

        if game.winner:
            print(f"{game.winner} Vince!")
        else:
            print("Pareggio!")


if __name__ == '__main__':
    Tris.test()
