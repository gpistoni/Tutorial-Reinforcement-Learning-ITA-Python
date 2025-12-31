import numpy as np
import random
import secrets                      # This module is responsible for providing access to the most secure source of randomness
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from tris import Tris
from RLAgent import QLearningAgent

# Test del QLearningAgent
if __name__ == '__main__':
    QLearningAgent.test_QLearningAgent()

    game = Tris()
    # Ricorda gli stati possibili nel gioco sono 3^9 = 19683
    # 9 caselle: vuoto, marcato X, marcato O.
    num_episodes = 1_000

    alpha=0.5
    epsilon=0.05
    discount_factor=1.0

    game.current_player = game.players[0]                   #Nota : game.players[0] == "X"  è il nostro agente e farà la prima mossa

    agent = QLearningAgent(alpha, epsilon, discount_factor)

    for i in range(1,num_episodes + 1):
        print("\nEpisode nr : ", i)

        game.reset()                                            # pulisce la griglia dalle operazioni precedenti per poter iniziare un nuovo episodio
        state = game.board
        stateHash = str(state.reshape(3 * 3))                   # Ricorda : c'è bisogno che stato attuale della griglia sia hashable e non un 'numpy.ndarray'

        # The exploration-exploitation trade-off
        # Qui un sempllice tunning per bilanciare esplorazione e sfruttamento
        if i > 40000:
            epsilon=0.5
        elif i > 70000:
            epsilon=0.8

        while (not game.game_over) and (bool(game.available_moves())) :

            #STEP 1
            #Seleziona un’azione possibile in un certo stato ed eseguila
            print("\n***Muova mossa****\n")
            print("Azioni possibili : " , game.available_moves()  )

            print("Sta muovendo il giocatore : ",game.current_player)
            if game.current_player == game.players[0] :
                move = agent.choose_action(stateHash, game.available_moves())
                #print("Azione scelta dal Q_learning_Agent=",move)
            else:
                move = secrets.choice(game.available_moves())

            game.make_move(move)

            # Stampa a schermo la situazione di gioco della griglia dopo l'azione 'move'
            # game.print_board()

            # STEP 2
            # agente riceve la ricompensa
            # game.winner puo essere "X" o "O"
            if game.winner == 'X':
                print(f"\t\t{game.winner} Vince!")
                reward = 1.0
            elif game.winner == 'O':
                print(f"\t\t{game.winner} Vince!")
                reward = - 0.05
            else:
                reward = 0.0

            # STEP 3
            #Qui si costruisce ed aggiorna la struttura dati  Q
            next_state = game.board                                 # 'next_state' è il nuovo stato in cui si arriva eseguendo azione: 'action'
            next_boardHash = str(next_state.reshape(3 * 3))

            dq = agent.update_Q_value(stateHash, move, reward, next_boardHash)

    print("\nSTRUTTURA DATI Q_TABLE (in pratica un dict) ottenuta alla fine del training")
    print(dq)
    print("Dimensoni : ",len(dq))

    #
    # Salva la struttura dati ottenuta su un file :
    #
    with open('Tris_data_train.pkl', 'wb') as fp:
        pickle.dump(dq, fp)
        print('dictionary saved successfully to file')