#Author: Kishorekumar Vishwanathan
import numpy as np
import torch
import chess
from model_trainTest import Model
from model_trainTest import * 
from stockfish import Stockfish

saved_model = Model()

#load best model path from your file
# f = open("./savedModels/bestModel.txt", "r")
# bestLoss = float(f.readline())
# model_path = f.readline()
# f.close()

model_path = "savedModels/model_20231230_170618_99"
saved_model.load_state_dict(torch.load(model_path))
saved_model.eval()

numOfGames = 200
winnedGames = 0
lostGames = 0
listOfMovesCount = []
for i in range(numOfGames):

    board = chess.Board()
    allMoves = [] #list of strings for saving moves for setting pos for stockfish

    MAX_NUMBER_OF_MOVES = 150
    for i in range(MAX_NUMBER_OF_MOVES): #set a limit for the game

        #first my ai move
        try:
            move = saved_model.predict(board)
            board.push(move)
            allMoves.append(str(move)) #add so stockfish can see
        except:
            # print("game over. You lost")
            # print("FEN:", board.fen())
            # print(board)
            # print("Move number:", i)
            lostGames += 1
            listOfMovesCount.append(i)
            break

        if board.is_game_over():
            # print("game over. You won")
            # print("FEN:", board.fen())
            # print(board)
            # print("Move number:", i)
            winnedGames += 1
            listOfMovesCount.append(i)
            break    
        # #then random move
        legalMoves =  board.generate_legal_moves()
        # print("Legal moves:", legalMoves)
        randomMove = np.random.choice(list(legalMoves))
        # print("Random move:", randomMove)
        board.push(randomMove)


        # stockfish.set_position(allMoves)
        # stockfishMove = stockfish.get_best_move_time(3)
        # allMoves.append(stockfishMove)
        # stockfishMove = chess.Move.from_uci(stockfishMove)
        # board.push(stockfishMove)


print("Games played:",numOfGames ,"Winned games:", winnedGames, "Lost games:", lostGames)
print("Average number of moves:", round(np.mean(listOfMovesCount),2))

