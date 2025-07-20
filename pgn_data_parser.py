##Author: Kishorekumar Vishwanathan
import chess.pgn
import pandas as pd

def flip_board(fen):
    # Split the FEN string into its components
    pieces, turn, castling, en_passant, halfmove, fullmove = fen.split()

    flipped_pieces = ''
    for char in pieces[::-1]:
        if char.isalpha():
            if char.islower():
                flipped_pieces += char
            else:
                flipped_pieces += char  
        else:
            flipped_pieces += char
    
    # Combine the components to form the new FEN string
    flipped_fen = f'{flipped_pieces} {turn} {castling} {en_passant} {halfmove} {fullmove}'

    return flipped_fen


def extract_data_from_pgn(pgn_file):
    print("Processing PGN file:", pgn_file)
    data = []
    move_count = 0
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # No more games in the file
            board = chess.Board()  # Initialize a new board for each game
            
            for move in game.mainline_moves():
                data.append((board.fen(), move.uci()))
                board.push(move)  # Update the board with the move
                move_count += 1
                if move_count % 100000 == 0:
                    print("Processed", move_count, "moves")
    
    print("*** Final count of processed moves is:", move_count, "***")           
    return data

def extract_winner_data_from_pgn(pgn_file):
    print("Processing PGN file:", pgn_file)
    data = []
    move_count = 0
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # No more games in the file

            # Check if the game has a result and a clear winner
            result = game.headers.get("Result")
            if result is not None and result != "1/2-1/2":
                # Determine the winner
                winner = "white" if result == "1-0" else "black"
                
                board = chess.Board()  # Initialize a new board for each game
                
                for move in game.mainline_moves():

                    if (board.turn == chess.WHITE and winner == "white"):
                        data.append((board.fen(), move.uci(), winner))
                        move_count += 1
                    elif (board.turn == chess.BLACK and winner == "black"):
                        data.append((flip_board(board.fen()), move.uci(), winner))
                        move_count += 1
                    board.push(move)  # Update the board with the move
                    # data.append((board.fen(), move.uci(), winner))
                    if move_count % 100000 == 0:
                        print("Processed", move_count, "moves")
    
    print("*** Final count of processed moves is:", move_count, "***")           
    return data


def extract_last_10_moves(pgn_file):
    print("Processing PGN file:", pgn_file)
    data = []
    move_count = 0
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # No more games in the file

            # Check if the game has a result and a clear winner
            result = game.headers.get("Result")
            if result is not None and result != "1/2-1/2":
                # Determine the winner
                winner = "white" if result == "1-0" else "black"
                
                board = chess.Board()  # Initialize a new board for each game
                last_10_moves = []

                for move in game.mainline_moves():

                    if (board.turn == chess.WHITE and winner == "white"):
                        if len(last_10_moves) < 10:
                            last_10_moves.append((board.fen(), move.uci(), winner))
                        else:
                            last_10_moves.pop(0)
                            last_10_moves.append((board.fen(), move.uci(), winner))
                    elif (board.turn == chess.BLACK and winner == "black"):
                        # data.append((flip_board(board.fen()), move.uci(), winner))
                        if len(last_10_moves) < 10:
                            last_10_moves.append((flip_board(board.fen()), move.uci(), winner))
                        else:
                            last_10_moves.pop(0)
                            last_10_moves.append((flip_board(board.fen()), move.uci(), winner))                        
                    board.push(move)  # Update the board with the move

                move_count += 10
                data.extend(last_10_moves)
                if move_count % 100000 == 0:
                        print("Processed", move_count, "moves")
                if move_count % 1000000 == 0 and move_count != 0:
                    print("*** Final count of processed moves is:", move_count, "***")           
                    return data
                # Append the last 10 moves to the data list
                
    
    print("*** Final count of processed moves is:", move_count, "***")           
    return data

def extract_data_for_white(pgn_file):
    print("Processing PGN file:", pgn_file)
    moves_wanted = 400000
    data = []
    move_count = 0

    with open(pgn_file) as f:
        while move_count < moves_wanted:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # No more games in the file
                
            # Check if the game has a result and a clear winner
            result = game.headers.get("Result")
            if result == "1-0": #white won  
                winner = "white"              
                board = chess.Board()  # Initialize a new board for each game
                
                for move in game.mainline_moves():

                    if (board.turn == chess.WHITE and winner == "white"):
                        data.append((board.fen(), move.uci(), winner))
                        move_count += 1
   
                    board.push(move)  # Update the board with the move
                    # data.append((board.fen(), move.uci(), winner))
                    if move_count % 100000 == 0:
                        print("Processed", move_count, "moves")
    
    print("*** Final count of processed moves for WHITE is:", move_count, "***")           


    return data

def extract_data_for_black(pgn_file):
    print("Processing PGN file:", pgn_file)
    moves_wanted = 400000
    data = []
    move_count = 0
    with open(pgn_file) as f:
        while move_count < moves_wanted:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # No more games in the file
            result = game.headers.get("Result")
            if result == "0-1": #black won  
                winner = "black"              
                board = chess.Board()  # Initialize a new board for each game
                for move in game.mainline_moves():
                    if (board.turn == chess.BLACK and winner == "black"):
                        data.append((board.fen(), move.uci(), winner))
                        move_count += 1
                    board.push(move)  # Update the board with the move
    return data

# # Define the path to your PGN file
# pgn_file_path = "LichessEliteDatabase\lichess_elite_2018-07.pgn"
# # Extract data from PGN file
# data = extract_data_from_pgn(pgn_file_path)

# # Create a DataFrame to store the data
# df = pd.DataFrame(data, columns=["FEN Position", "UCI Move"])

# # Save the data to a CSV file
# df.to_csv("data/rawData/chess_data4.csv", index=False)

# Define the path to your PGN file
pgn_file_path = "LichessEliteDatabase\lichess_elite_2017-11.pgn"
# Extract data from PGN file, considering only games with a clear winner
# data = extract_winner_data_from_pgn(pgn_file_path)
# data = extract_last_10_moves(pgn_file_path)
dataWhite = extract_data_for_white(pgn_file_path)
dataBlack = extract_data_for_black(pgn_file_path)

# Create a DataFrame to store the data
dfWhite = pd.DataFrame(dataWhite, columns=["FEN Position", "UCI Move", "Winner"])
dfBlack = pd.DataFrame(dataBlack, columns=["FEN Position", "UCI Move", "Winner"])

# Save the data to a CSV file
dfWhite.to_csv("data/rawData/white/chess_white_data6.csv", index=False)
dfBlack.to_csv("data/rawData/black/chess_black_data6.csv", index=False)
