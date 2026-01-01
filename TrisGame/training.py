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
    game = Tris()
    # Ricorda gli stati possibili nel gioco sono 3^9 = 19683 // 9 caselle // vuoto, marcato X, marcato O.

    num_episodes = 10_000
    X_wins = 0

    learnig_rate=0.5
    exploration_rate=0.1
    discount_factor=1.0

    with open('Tris_data_train.pkl', 'rb') as fp:
        dq = pickle.load(fp)

    agent = QLearningAgent(learnig_rate, exploration_rate, discount_factor)
    agent.load_Q_table(dq)  # dq è il dizionario che rappresenta la Q-table
    
     # Ciclo di training per un certo numero di episodi
    for i in range(1,num_episodes + 1):
        print("\nEpisode nr : ", i)

        game.reset() # pulisce la griglia dalle operazioni precedenti per poter iniziare un nuovo episodio
        game.current_player = random.choice(game.players)
        boardHash = game.GetBoardHash()
        reward = 0.0

        # The exploration-exploitation trade-off
        # Qui un sempllice tunning per bilanciare esplorazione e sfruttamento
        if i > 40000:
            exploration_rate=0.5
        elif i > 70000:
            exploration_rate=0.8

        while (not game.game_over) and (bool(game.available_moves())) :

            #STEP 1
            #Seleziona un’azione possibile in un certo stato ed eseguila
            #print("\n***Muova mossa****\n")
            #print("Azioni possibili : " , game.available_moves()  )
            
            QLearningMoved = False
            #print("Sta muovendo il giocatore : ",game.current_player)
            boardHash = game.GetBoardHash() 
            print("boardHash=",boardHash)

            if game.current_player == game.players[0]:
                move = agent.choose_action(boardHash, game.available_moves())
                print("Azione scelta da X=",move)
                QLearningMoved = True
            else:
                move = secrets.choice(game.available_moves())
                print("Azione scelta da O=",move)
            
            game.make_move(move)
            next_boardHash = game.GetBoardHash() 

            # STEP 2
            # agente riceve la ricompensa
            # game.winner puo essere "X" o "O"
            if game.winner == 'X':
                print(f"\t\t{game.winner} Vince!")
                reward = 1.0
                X_wins += 1
            elif game.winner == 'O':
                print(f"\t\t{game.winner} Vince!")
                reward = - 0.05  
            else:
                reward = 0.0


            # STEP 3  aggiorno i valori Q
                agent.update_Q_value(boardHash, move, reward, next_boardHash)


    print("\nSTRUTTURA DATI Q_TABLE (in pratica un dict) ottenuta alla fine del training",len(agent.Q))
    print("\nXWin %:",100 * X_wins / num_episodes )
    
    # Salva la struttura dati ottenuta su un file:    
    with open('Tris_data_train.pkl', 'wb') as fp:
        pickle.dump(agent.Q, fp)
        print('dictionary saved successfully to file')
    
    # salva items in un file csv per ispezione
    items = list(agent.Q.items())
    df = pd.DataFrame(items, columns=['State-Action', 'Q-Value'])
    df.to_csv('Tris_data_train_inspect.csv', index=False)
