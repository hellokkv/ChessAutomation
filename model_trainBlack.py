##Author: Kishorekumar Vishwanathan
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import gym
import gym_chess
import os
import chess
from tqdm import tqdm
from gym_chess.alphazero.move_encoding import utils
from pathlib import Path

from stockfish import Stockfish


from typing import Optional

# ****  Helper functiions to decode moves ****

#decoding moves from idx to uci notation

def _decodeKnight(action: int) -> Optional[chess.Move]:
    _NUM_TYPES: int = 8

    _TYPE_OFFSET: int = 56

    _DIRECTIONS = utils.IndexedTuple(
        (+2, +1),
        (+1, +2),
        (-1, +2),
        (-2, +1),
        (-2, -1),
        (-1, -2),
        (+1, -2),
        (+2, -1),
    )

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_knight_move = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_knight_move:
        return None

    knight_move_type = move_type - _TYPE_OFFSET

    delta_rank, delta_file = _DIRECTIONS[knight_move_type]

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move

def _decodeQueen(action: int) -> Optional[chess.Move]:

    _NUM_TYPES: int = 56 # = 8 directions * 7 squares max. distance

    #: Set of possible directions for a queen move, encoded as 
    #: (delta rank, delta square).
    _DIRECTIONS = utils.IndexedTuple(
        (+1,  0),
        (+1, +1),
        ( 0, +1),
        (-1, +1),
        (-1,  0),
        (-1, -1),
        ( 0, -1),
        (+1, -1),
    )
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))
    
    is_queen_move = move_type < _NUM_TYPES

    if not is_queen_move:
        return None

    direction_idx, distance_idx = np.unravel_index(
        indices=move_type,
        shape=(8,7)
    )

    direction = _DIRECTIONS[direction_idx]
    distance = distance_idx + 1

    delta_rank = direction[0] * distance
    delta_file = direction[1] * distance

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move

def _decodeUnderPromotion(action):
    _NUM_TYPES: int = 9

    _TYPE_OFFSET: int = 64

    #: Set of possibel directions for an underpromotion, encoded as file delta.
    _DIRECTIONS = utils.IndexedTuple(
        -1,
        0,
        +1,
    )

    #: Set of possibel piece types for an underpromotion (promoting to a queen
    #: is implicitly encoded by the corresponding queen move).
    _PROMOTIONS = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_underpromotion = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_underpromotion:
        return None

    underpromotion_type = move_type - _TYPE_OFFSET

    direction_idx, promotion_idx = np.unravel_index(
        indices=underpromotion_type,
        shape=(3,3)
    )

    direction = _DIRECTIONS[direction_idx]
    promotion = _PROMOTIONS[promotion_idx]

    to_rank = from_rank + 1
    to_file = from_file + direction

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    return move

#primary decoding function, the ones above are just helper functions
def decodeMove(action: int, board) -> chess.Move:
        move = _decodeQueen(action)
        is_queen_move = move is not None

        if not move:
            move = _decodeKnight(action)

        if not move:
            move = _decodeUnderPromotion(action)

        if not move:
            raise ValueError(f"{action} is not a valid action")

        turn = board.turn
        
        if turn == False: #black to move
            move = utils.rotate(move)

        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == True) or 
                (to_rank == 0 and turn == False)
            )

            piece = board.piece_at(move.from_square)
            if piece is None: #NOTE I added this, not entirely sure if it's correct
                return None
            is_pawn = piece.piece_type == chess.PAWN

            if is_pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        return move

# ****  Helper functions to encode Board ****

def encodeBoard(board: chess.Board) -> np.array:
	"""Converts a board to numpy array representation."""

	array = np.zeros((8, 8, 14), dtype=int)

	for square, piece in board.piece_map().items():
		rank, file = chess.square_rank(square), chess.square_file(square)
		piece_type, color = piece.piece_type, piece.color

		offset = 0 if color == chess.WHITE else 6
		
		# Chess enumerates piece types beginning with one, which we have
		# to account for
		idx = piece_type - 1
	
		array[rank, file, idx + offset] = 1

	# Repetition counters
	array[:, :, 12] = board.is_repetition(2)
	array[:, :, 13] = board.is_repetition(3)

	return array


# **** Model specification ****
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.found_mates = True

        self.INPUT_SIZE = 896 
        self.OUTPUT_SIZE = 4672 # = number of unique moves 
        
        self.activation = torch.nn.Tanh()   

        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 1000)
        self.linear2 = torch.nn.Linear(1000, 1000)
        self.linear3 = torch.nn.Linear(1000, 1000)
        self.linear4 = torch.nn.Linear(1000, 200)
        self.linear5 = torch.nn.Linear(200, self.OUTPUT_SIZE)
        self.softmax = torch.nn.Softmax(1) 
 
    def forward(self, x): #x.shape = (batch size, 896)
        x = x.to(torch.float32)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        return x

    
    def get_best_moves(self, board, num_moves):
        with torch.no_grad():
            encodedBoard = encodeBoard(board)
            encodedBoard = encodedBoard.reshape(1, -1)
            encodedBoard = torch.from_numpy(encodedBoard)
            res = self.forward(encodedBoard)
            probs = self.softmax(res)

            probs = probs.numpy()[0]

            list_of_predicted_moves = []

            while len(probs) > 0: #try max 100 times, if not throw an error
                moveIdx = probs.argmax()
                try:
                    uciMove = decodeMove(moveIdx, board)
                    if (uciMove is None): #could not decode
                        probs = np.delete(probs, moveIdx)
                        continue
                    move = chess.Move.from_uci(str(uciMove))
                    if (move in board.legal_moves): #if legal, return, else: loop continues after deleting the move
                        list_of_predicted_moves.append(move)
                except:
                    pass
                probs = np.delete(probs, moveIdx) #remove the move so its not chosen again next iteration

            # delete duplicates
            list_of_predicted_moves = list(dict.fromkeys(list_of_predicted_moves))
    
            # return list_of_predicted_moves
            if len(list_of_predicted_moves) > 0:
                return list_of_predicted_moves
            else: # return random move if model failed to find move
                moves = board.legal_moves
                if (len(list(moves)) > 0):
                    return moves
                else: #if no legal moves, return None
                    return None

    def predict(self, board : chess.Board):
        
        numOfmoves = 0

        if self.found_mates:
            top_moves = board.legal_moves
            numOfmoves = len(list(top_moves))
        else:
            top_moves = self.get_best_moves(board, 10)
            numOfmoves = len(list(top_moves))

        current_eval = self.evaluateBoard(board, 4)

        top_score = 100000 
        minimax_move = None
        for i in range(1):
            if numOfmoves > 0:
                top_score = 100000
                minimax_move = None
                moveIdx = 0
                for move in top_moves:
                    moveIdx += 1
                    new_board = board.copy()
                    new_board.push(move)
                    score = self.minimax_evaluation(new_board, 4, -100000, 100000, True)
                    print(moveIdx ,"Move:", move, "Score:", score)
                    if score < top_score:
                        top_score = score
                        minimax_move = move
    
                    if score <= -14000:
                        self.found_mates = True
                        print("Found mate in one. Breaking search loop...")
                        break

                if top_score <= -7000:
                    self.found_mates = True
                    print("Found mate")
                else:
                    self.found_mates = False

                if current_eval < top_score:
                    print("Worse move found, checking all possible moves")
                    top_moves = board.legal_moves
                    numOfmoves = len(list(top_moves)) 
                else:
                    break
            else:
                return None
        print("Top score:", top_score, "Move:", minimax_move, "Number of moves:", numOfmoves, "Found mate:", self.found_mates)   

        return minimax_move    

    def minimax_evaluation(self, board, depth, alpha, beta, maximizingPlayer):
        
        moves = board.legal_moves
        
        if depth == 0 or board.is_checkmate() or len(list(moves)) == 0:
            return self.evaluateBoard(board, depth)
        
        if len(list(moves)) == 0:
            print("No legal moves found")
        if maximizingPlayer:
            maxEval = -100000
            for move in moves:
                board.push(move)
                eval = self.minimax_evaluation(board, depth-1, alpha, beta, False)
                board.pop()
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = 100000
            for move in moves:
                board.push(move)
                eval = self.minimax_evaluation(board, depth-1, alpha, beta, True)
                board.pop()
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            if minEval == 100000:
                print("Something went wrong")
            return minEval

    def evaluateBoard(self, board, depth):
        total_score = 0
        pieces = board.piece_map()
        list_of_pieces = []
        
        for i in range(8):
            for j in range(8):
                index = i*8 + j
                if index in pieces:
                    piece_type = pieces.get(index)
                    piece_type = str(piece_type)
                    total_score += self.evaluate_figure(piece_type,i,j)

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                total_score -= 10000 + ( 1000 * depth)
                # print("Found checkmate for black")
            else:
                total_score += 10000 + ( 1000 * depth)    
                # print("Found checkmate for white at depth:", depth, "Score:", total_score)

        if total_score >= 100000:
            print("Wrooong")
        return total_score
    
        

    def evaluate_figure(self,figure,i,j):
        material_scores = {
            chess.PAWN: 10,
            chess.KNIGHT: 30,
            chess.BISHOP: 30,
            chess.ROOK: 50,
            chess.QUEEN: 90,
            chess.KING: 900
        }

        whitePawnEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] 
        ]
        # only zero field
        whitePawnEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]   
        ]

        blackPawnEval = list(reversed(whitePawnEval))

        knightEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] 
        ]

        whiteBishopEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] 
        ]    

        blackBishopEval = list(reversed(whiteBishopEval))

        whiteRookEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] 
        ]

        blackRookEval = list(reversed(whiteRookEval))

        queenEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] 
        ]  

        whiteKingEval = [
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] 
        ]
        blackKingEval = list(reversed(whiteKingEval))

        if figure == 'P':
            return material_scores[chess.PAWN] + whitePawnEval[i][j]
        elif figure == 'N':
            return material_scores[chess.KNIGHT] + knightEval[i][j]
        elif figure == 'B':
            return material_scores[chess.BISHOP] + whiteBishopEval[i][j]
        elif figure == 'R':
            return material_scores[chess.ROOK] + whiteRookEval[i][j]
        elif figure == 'Q':
            return material_scores[chess.QUEEN] + queenEval[i][j]
        elif figure == 'K':
            return material_scores[chess.KING] + whiteKingEval[i][j]
        elif figure == 'p':
            return -(material_scores[chess.PAWN] + blackPawnEval[i][j])
        elif figure == 'n':
            return -(material_scores[chess.KNIGHT] + knightEval[i][j])
        elif figure == 'b':
            return -(material_scores[chess.BISHOP] + blackBishopEval[i][j])
        elif figure == 'r':
            return -(material_scores[chess.ROOK] + blackRookEval[i][j])
        elif figure == 'q':
            return -(material_scores[chess.QUEEN] + queenEval[i][j])
        elif figure == 'k':
            return -(material_scores[chess.KING] + blackKingEval[i][j])
        else:
            print("ERROR: figure not found")
            return 0    

# **** training helper functions ****
def train_one_epoch(model,optimizer,loss_fn,epoch_idx, tb_writer, trainLoader):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(trainLoader):

        # split data into inputs and labels
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_idx * len(trainLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def createBestModelFile():
    #first find best model if it exists:
    path = Path('./savedModels/blackModels/bestModel.txt')

    if not (path.is_file()):
        #create the files
        print("Creating bestModel.txt file")
        f = open(path, "w")
        f.write("10000000") #set to high number so it is overwritten with better loss
        f.write("\ntestPath")
        f.close()

def saveBestModel(vloss, pathToBestModel):
    f = open("./savedModels/blackModels/bestModel.txt", "w")
    f.write(str(vloss))
    f.write("\n")
    f.write(pathToBestModel)
    print("NEW BEST MODEL FOUND WITH LOSS:", vloss)

def retrieveBestModelInfo():
    f = open('./savedModels/blackModels/bestModel.txt', "r")
    bestLoss = float(f.readline())
    bestModelPath = f.readline()
    f.close()
    return bestLoss, bestModelPath


# **** main function ****
def main():
    #check if cuda available
    # torch.cuda.is_available()
    print("Is cuda available:" ,torch.cuda.is_available())

    # **** Hyperparameters ****
    FRACTION_OF_DATA = 1 # 1 = all data, 0.5 = half of data, 0.1 = 10% of data
    BATCH_SIZE = 32 

    # **** Load data ****
    loaded_Positions = np.load('data/preparedData/encoded_positions_black.npy')
    print("Loaded positions")
    loaded_Moves = np.load('data/preparedData/encoded_moves_black.npy')
    print("Loaded moves")

    all_Positions = loaded_Positions[:int(len(loaded_Positions)*FRACTION_OF_DATA)]  #take only a fraction of data
    all_Moves = loaded_Moves[:int(len(loaded_Moves)*FRACTION_OF_DATA)] #take only a fraction of data
    assert len(all_Positions) == len(all_Moves)     #check if same length
    print("Number of positions: ", len(all_Positions))

    train_koef = 0.9
    print("Train koef: ", train_koef)
    train_size = int(train_koef * len(all_Positions))
    print("Train size: ", train_size)

    # transfer data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    all_Positions = torch.from_numpy(np.asanyarray(all_Positions)).to(device)
    all_Moves = torch.from_numpy(np.asanyarray(all_Moves)).to(device)
    print("Data transfered to GPU")

    trainingData = torch.utils.data.TensorDataset(all_Positions[:train_size], all_Moves[:train_size])
    testData = torch.utils.data.TensorDataset(all_Positions[train_size:], all_Moves[train_size:])

    trainLoader = torch.utils.data.DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)

    #hyperparams
    EPOCHS = 70
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9


    # ****          Training cycle          ****

    createBestModelFile()
    bestLoss, bestModelPath = retrieveBestModelInfo()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    model = Model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_vloss = 1_000_000.


    for epoch in tqdm(range(EPOCHS)):
        if epoch_number % 5 == 0:
            print('EPOCH {}:'.format(epoch_number + 1))

        # Train
        model.train()
        avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch_number, writer, trainLoader)

        running_vloss = 0.

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testLoader):
                vinputs, vlabels = data
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
        
        avg_vloss = running_vloss / ( i + 1 )

        if epoch_number % 5 == 0:
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalar('Training Loss', avg_loss, epoch_number + 1)
        writer.add_scalar('Validation Loss', avg_vloss, epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss

        #     if (bestLoss > best_vloss): #if better than previous best loss from all models created, save it
        #         model_path = 'savedModels/model_{}_{}'.format(timestamp, epoch_number)
        #         torch.save(model.state_dict(), model_path)
        #         saveBestModel(best_vloss, model_path)\

        # track every model
        model_path = 'savedModels/blackModels/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
        saveBestModel(avg_vloss, model_path)



        epoch_number += 1


    print("\n\nBEST VALIDATION LOSS FOR ALL MODELS: ", bestLoss)

if __name__ == "__main__":
    main()