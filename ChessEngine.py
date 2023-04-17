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
        if depth == 0:
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
    model = models.Sequential()
    #model.add(layers.Conv2D(1024, (1, 1), activation='relu', input_shape=(8, 8, 14)))
    
    model.add(layers.Dense(2048, activation='relu', input_shape = [768], batch_size=1))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.load_weights(path)

    #model.summary()

    return model

def evaluate(board, model):
    #converted_board, turn_mod = convert_fen(fen)
    board = np.reshape(board, (1, 768))
    #Prediction at this point is [Opponent, Draw, Player]
    prediction = model.predict(board, verbose=0)[0]
    #max_pre = np.argmax(prediction)
    return prediction
    #* (1 - (2*turn_mod))


#import_and_process("GameData.pgn", "labeled_games.csv", 30000, skip_games=20000)

#train_from_processed("labeled_games.csv", "BackToCategorical.h5", num_epochs=20)
model = build_model("BackToCategorical.h5")

test_fen="r1b1k2r/pppp1ppp/2n5/3B2B1/3pP3/8/PPP2PPP/R2QK2R w KQkq - 0 1"


board = chess.Board(fen=test_fen)
start=time.time()
move_list = choose_move(board, model, 2, iterative=False)
end = time.time()
print(move_list[0])
total_time = end - start
print("\n Not iterative: "+ str(total_time))

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

