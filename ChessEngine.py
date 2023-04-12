import chess
#import chess.pgn
from train_model import import_and_process, convert_fen, train_from_processed
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np


#1st dimension order is K, Q, R, B, N, P, k, q, r, b, n, p, en_passant,
#special(Active player(a1), White Can Castle Kingside(e1), White can Castle Queenside(d1), Black Can Castle Kingside(e8), Black Can Castle Queenside(d8)) 
layer_list = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
              #, 'en_passant', 'special']

def choose_move(fen, model):
    board = chess.Board(fen=fen)
    converted_boards = []
    for move in board.legal_moves:
        board.push(move)
        move_and_board = (move, convert_fen(board.fen()))
        converted_board.append(move_and_board)
        board.pop()
    best_move = ''
    best_score = 0
    turn_mod = 1
    if board.turn == chess.BLACK:
        turn_mod = -1
    for board in converted_boards:
        pass


def build_model(path):
    model = models.Sequential()
    #model.add(layers.Conv2D(1024, (1, 1), activation='relu', input_shape=(8, 8, 14)))
    
    model.add(layers.Dense(2048, activation='relu', input_shape = [768], batch_size=1))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1))

    model.load_weights(path)

    model.summary()

    return model

def evaluate(fen, model):
    board, turn_mod = convert_fen(fen)
    board = np.reshape(board, (1, 768))
    prediction = model.predict(board, verbose=0)
    #max_pre = np.argmax(prediction)
    return prediction * (1 - (2*turn_mod))


#import_and_process("GameData.pgn", "labeled_games_continuous.csv", 2000, skip_games=8000)

#train_from_processed("labeled_games_continuous.csv", num_epochs=10)
model = build_model("SamePerspectiveMore.h5")

print(evaluate("3Q4/8/k7/8/8/1R1K4/8/8 b - - 0 1", model))

