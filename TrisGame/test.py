import numpy as np
import random
import secrets                      # This module is responsible for providing access to the most secure source of randomness
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt
from tris import Tris
from RLAgent import QLearningAgent

def test_QLearningAgent_rnd(dq):

    num_games = 10_000
    num_wins = 0
    game = Tris()    

    for j in range(num_games):
            game.reset()
            game.current_player = random.choice(game.players)
            stateHash = game.GetBoardHash()

            while (not game.game_over) and (bool(game.available_moves())) :

                if game.current_player == game.players[1]:   # Agente_Random             
                    move = secrets.choice(game.available_moves())
                else:   
                    sub_dq = {}
                    for i in game.available_moves():
                        sub_dq.update({(stateHash,i) : dq[stateHash,i]})

                    newdq = pd.DataFrame.from_dict( sub_dq, orient ='index' )
                    #print("Sub-dict delle azioni disponibili in un certo stato : \n ",newdq)

                    index_max_Qvalue = max(sub_dq, key=sub_dq.get)
                    move =  index_max_Qvalue[1]

                game.make_move(move)
                if j % 1000 == 0:
                    game.draw_board_image(0.1)

            #print("Fine partita nr : ", j )
            #print("WINNER : ",  game.winner)

            if game.winner == game.players[0] :
                num_wins += 1

    print("\nESITO FINALE DEL TEST :")
    print(num_wins / num_games * 100)




# Test del QLearningAgent
if __name__ == '__main__':

    with open('Tris_data_train.pkl', 'rb') as fp:
        dq = pickle.load(fp)

    print(len(dq))

        # salva items in un file csv per ispezione
    items = list(dq.items())
    df = pd.DataFrame(items, columns=['State-Action', 'Q-Value'])
    df.to_csv('Tris_data_train_inspect.csv', index=False)
    
    test_QLearningAgent_rnd(dq);
