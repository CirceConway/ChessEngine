import chess
#import chess.pgn
from train_model import import_and_process, convert_fen, train_from_processed
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from operator import itemgetter
import time

#1st dimension order is K, Q, R, B, N, P, k, q, r, b, n, p, en_passant,
#special(Active player(a1), White Can Castle Kingside(e1), White can Castle Queenside(d1), Black Can Castle Kingside(e8), Black Can Castle Queenside(d8)) 
layer_list = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
              #, 'en_passant', 'special']

positions = 0

def alpha_beta(board, move_list, model, depth, prev_best=1):
    #if depth == 0:
    #    for move in board.legal_moves:
    #        converted, turn_mod = convert_fen(fen)
    #        full_eval = evaluate(converted, model)
    #        eval = full_eval[2] - full_eval[0]
    #        #print(f"Evaluating position {fen}")
    #        print(eval)
    #        return ('', eval)
    #board = chess.Board(fen=fen)
    evals = dict()
    current_best = -1
    for row in move_list:
        move = row[0]
        board.push(move)
        if board.is_checkmate():
            eval = 2
        elif depth == 0:
            converted, turn_mod = convert_fen(board.fen())
            
            full_eval = evaluate(converted, model)
            eval = full_eval[2] - full_eval[0]
            
            global positions
            positions += 1
        else:
            move_list = []
            for temp_move in board.legal_moves:
                move_list.append([temp_move, 0])
            
            eval = alpha_beta(board, move_list, model, depth - 1, prev_best = (current_best * -1))[0][1] * -1
        if eval > current_best:
            current_best = -eval

        evals[move] = eval
        board.pop()

        if eval > prev_best:
            break

    sorted_evals = sorted(evals.items(), key=itemgetter(1), reverse=False)
    return sorted_evals

#Maybe try getting a batch prediciton of every move of the first leaf node
#Then doing batch predictions in a breadth-first search made of 1 move from each leaf node


def choose_move(board, model, depth, iterative=True):
    #board = chess.Board(fen=fen)
    move_list = []
    for move in board.legal_moves:
        move_list.append([move, 0])
    global positions
    positions = 0
    if not iterative:
        move_list = alpha_beta(board, move_list, model, depth)
        print(f"Searched {positions} positions")
        return move_list
    for i in range(depth+1):
        move_list = alpha_beta(board, move_list, model, i)
    print(f"Searched {positions} positions")
    return move_list


def build_model(path):
    input_shape = (8, 8, 12)
    input_layer = tf.keras.layers.Input(shape=input_shape, batch_size = 1)

    #First convolutional layer for detecting features across the whole board
    conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(8, 8), activation='relu')(input_layer)
    flatten1 = tf.keras.layers.Flatten()(conv1)
    #Second layer for detecting features on smaller subsections of the board
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(input_layer)
    flatten2 = tf.keras.layers.Flatten()(conv2)

    merged_layer = tf.keras.layers.Concatenate()([flatten1, flatten2])
    #flatten_layer = tf.keras.layers.Flatten()(merged_layer)
    dropout_layer = tf.keras.layers.Dropout(.1)(merged_layer)
    dense_layer = tf.keras.layers.Dense(units=128, activation='relu')(dropout_layer)
    dense_output = tf.keras.layers.Dense(units=3)(dense_layer)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense_output)

    #model = models.Sequential()
    #
    #model.add(layers.Dense(2048, activation='relu', input_shape = [768], batch_size=1))
    #model.add(layers.Dense(1024, activation='relu'))
    #model.add(layers.Dense(512, activation='relu'))
    #model.add(layers.Dense(3, activation='softmax'))
    #
    #model.load_weights(path)

    #model.summary()

    return model

def evaluate(board, model):
    #converted_board, turn_mod = convert_fen(fen)
    board = np.reshape(board, (1, 8, 8, 12))
    #Prediction at this point is [Opponent, Draw, Player]
    prediction = model.predict(board, verbose=0)[0]
    #max_pre = np.argmax(prediction)
    return prediction
    #* (1 - (2*turn_mod))

def play_self_game(model):
    print("Starting Game")
    board = chess.Board()

    while not board.is_game_over():
        move_list = choose_move(board, model, 1)
        board.push(move_list[0][0])
        print(move_list[0])
    print(board.fen())
    

#import_and_process("GameData.pgn", "labeled_games.csv", 30000, skip_games=20000)

#train_from_processed("labeled_games.csv", "Simultaneous_Convolutions.h5", num_epochs=5)
model = build_model("Simultaneous_Convolutions.h5")

#play_self_game(model)

test_fen="r1b1k2r/pppp1ppp/2n5/3B2B1/3pP3/8/PPP2PPP/R2QK2R w KQkq - 0 1"


board = chess.Board(fen=test_fen)
#start=time.time()
#move_list = choose_move(board, model, 2, iterative=False)
#end = time.time()
#print(move_list[0])
#total_time = end - start
#print("\n Not iterative: "+ str(total_time))
#
start=time.time()
move_list = choose_move(board, model, 2, iterative=True)
end = time.time()
print(move_list[0])
total_time = end - start
print("\n Iterative: "+ str(total_time))
#board.push(move_list[0][0])
#prediction = evaluate(convert_fen("r1b2k1r/pppp1Bpp/2n5/6B1/3pP3/8/PPP2PPP/R2QK2R w KQkq - 0 1")[0], model)
#print(prediction[2] - prediction[0])

#test, test_mod = convert_fen("8/8/2k2Q2/8/3K4/8/8/8 b - - 0 1")

#print(test)

