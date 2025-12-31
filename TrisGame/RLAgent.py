import numpy as np
import random
import secrets                      # This module is responsible for providing access to the most secure source of randomness
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

from tris import Tris

######################################################################################################################################################################
######################################################################################################################################################################
class QLearningAgent:
######################################################################################################################################################################
    def __init__(self, alpha, epsilon, discount_factor):

        self.Q = {} # la struttura dati per rappresentare la Q-table è un dizionario :
                    #  chiave: (stato,azione) -> valore:  Q-values.

        #paramentri
        self.alpha = alpha      #  learning rate, controlla quanto i valori Q vengono aggiornati ad ogni passaggio.
        self.epsilon = epsilon  #  exploration rate, che controlla la probabilità di scegliere un'azione casuale invece dell'azione ottimale.
        self.discount_factor = discount_factor

######################################################################################################################################################################
    def get_Q_value(self, state, action):
      # Questo metodo restituisce il valore Q per una determinata coppia stato-azione.
      # Chiavi del dizianario Q : stato, azione
      # Ogni stato è una tupla che rappresenta lo stato corrente della griglia
      # Esempio lo stato iniziale : [0. 0. 0. 0. 0. 0. 0. 0. 0.]
      # Ogni azione è una tupla che rappresenta le coordinate del movimento.
      # Esempio posizione al centro della griglia : (1, 1)
      # Valore del dizionario Q : Q-value, ovvero i valori rincompensa
        #print("INPUT state in 'get_Q_value'",state)
        #print("INPUT action in 'get_Q_value'",action)
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0.0   # I valori Q iniziali verranno impostati su zero
        #print("OUTPUT of method 'get_Q_value',the Current dict Q=",self.Q )
        return self.Q[(state, action)]

######################################################################################################################################################################
    def choose_action(self, state, available_moves):
        if  secrets.SystemRandom().uniform(0,1) < self.epsilon: # oppure: random.uniform(0, 1) < self.epsilon:
            return secrets.choice(available_moves)
        else:
            # sceglie l'azione con il valore Q più alto.
            Q_values = []
            for action in available_moves:
                 Q_values.append(self.get_Q_value(state, action))
            #print( "La lista dei valori Q_values per le azioni ancora disponibili (in method 'choose_action') :\n ",Q_values)
            max_Q = max(Q_values)

            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

######################################################################################################################################################################
    def update_Q_value(self, state, action, reward, next_state):
        # Questo metodo aggiorna il valore Q per una determinata coppia stato-azione in base all'algoritmo Q-learning.

        # Sotto la lista di tutti i valori Q che potrebbero essere ottenuti partendo dallo stato 'next_action', applicando le azioni disponibili in questo stato.
        next_Q_values = [self.get_Q_value(next_state, next_action) for next_action in Tris().available_moves()]

        if not next_Q_values : # se la lista è vuota  ritorna  0
            max_next_Q = 0.0
        else:
            max_next_Q =  max(next_Q_values)    # altrimenti ritorno il massimo dei valori futuri ancora possibili

        # Processo iterativo di aggiornamento e correzione basato sulla nuova informazione.
        self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * (reward + self.discount_factor * max_next_Q - self.Q[(state, action)])
        #print(f"Sone nel metodo 'update_Q_value'\n Valore aggiornato di Q in stato {state} per l'azione {action} è : {self.Q[(state, action)]} ")

        return self.Q

######################################################################################################################################################################
    def test_QLearningAgent():
        alpha=0.5
        epsilon=0.1
        discount_factor=1.0

        agent = QLearningAgent(alpha, epsilon, discount_factor)

        state = Tris().board

        available_moves = Tris().available_moves()
        print("available_moves=", available_moves)
        print("state=\n",state)

        # Per lavorare con una struttura dati dizionario
        # c'è bisogno che stato attuale della griglia sia hashable e non un 'numpy.ndarray'
        boardHash = str(state.reshape(3 * 3))
        print("boardHash=",boardHash)

        action = agent.choose_action(boardHash, available_moves)
        print("action=",action)

        state[action[0],action[1]] = 1
        print(" state after action (or next_state) =\n",state)
        next_state = state

        w = True
        if w : # se c'è un vincitore, vedi def check_winner
            reward = 1
        else :
            reward = 0

        next_boardHash = str(next_state.reshape(3 * 3))
        dq = agent.update_Q_value(boardHash, action, reward, next_boardHash)

        sorted_dq = sorted(dq.items(), key=lambda x:x[1], reverse=True)
        dq = dict(sorted_dq)
        print(dq)




