#Author: Kishorekumar Vishwanathan
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_trainWhite import Model as whiteModel
from model_trainBlack import Model as blackModel
from model_trainTest import Model as combinedModel
import torch
import chess

app = Flask(__name__)
CORS(app)

white_chess_model = whiteModel()
black_chess_model = blackModel()

#load model from path
white_model_path = "savedModels/whiteModels/model_20240118_100613_66"
black_model_path = "savedModels/blackModels/model_20240117_181529_69"

white_chess_model.load_state_dict(torch.load(white_model_path))
white_chess_model.eval()

black_chess_model.load_state_dict(torch.load(black_model_path))
black_chess_model.eval()


@app.route('/predict_move', methods=['POST'])
def predict_move():
    print(request.json)
    if request.method == 'POST':
        board = request.json['board_state']
        board = chess.Board(board)
        move = None
        if board.turn == chess.WHITE:
            move = white_chess_model.predict(board)
            print("White move:", move)
        else:
            move = black_chess_model.predict(board)
            print("Black move:", move)

        return jsonify({'move': str(move)})
    
if __name__ == '__main__':
    app.run(port=5000)