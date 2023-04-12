import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import chess.pgn
import chess
import csv
import numpy as np
from sklearn.model_selection import train_test_split

#model.summary()
layer_list = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
              #, 'en_passant', 'special']

#Creates a board fill with "My pieces" and "Their pieces" instead of white and black
def convert_fen(fen):
    converted_board = np.zeros((8,8,12))
    piece_split = fen.split()
    board = piece_split[0].split("/")
    turn_mod = 0
    if piece_split[1] == 'b':
        turn_mod = 1
    for rank in range(len(board)):
        board_file = 0
        for file in range(len(board[rank])):
            piece = board[rank][file]
            if piece.isnumeric():
                board_file += int(piece)
            else:
                layer_ind = layer_list.index(piece)
                #This line makes sure that first 6 boards always belong to the current turn player
                layer_ind = (layer_ind + (6*turn_mod)) % 12
                converted_board[rank][board_file][layer_ind] = 1
                board_file += 1

    #Used for player turn and castling ability
    #if piece_split[1] == 'b':
    #    converted_board[0][0][13] = 1
    #if 'K' in piece_split[2]:
    #    converted_board[0][4][13] = 1
    #if 'Q' in piece_split[2]:
    #    converted_board[0][3][13] = 1
    #if 'k' in piece_split[2]:
    #    converted_board[7][4][13] = 1
    #if 'q' in piece_split[2]:
    #    converted_board[7][3][13] = 1

    #Implement En passant eventually in layer index 12
    #if piece_split[2] != '-':


    return converted_board.flatten(), turn_mod

#Extracts midgame and endgame positions from each game and stores them in out_path.
#OUTPATH IS WIPED AT THE BEGINNING OF THIS FUNCTION
def import_and_process(in_path, out_path, num_games, skip_games=0):
    gamelist = open(in_path)
    for i in range(skip_games):
        chess.pgn.skip_game(gamelist)
    with open(out_path, 'a') as file:
        print("Beginning Processing")
        csv_writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in range(num_games):
            if i % 100 == 0:
                print(f"Progress: {100 * (i/num_games)}%")
            game = chess.pgn.read_game(gamelist)
            label = game.headers["Result"].split('-')[0]
            if label == '1/2':
                label = 0.5
            label = float(label)*2 - 1
            #temp_board = chess.Board()
            game = chess.pgn.read_game(gamelist)
            move = 1
            while game.next() is not None:
                if move == 13 or move == 14:
                    csv_writer.writerow([label, game.board().fen()])
                if move % 29 == 0:
                    csv_writer.writerow([label, game.board().fen()])
                move += 1
                game = game.next()
            csv_writer.writerow([label, game.board().fen()])
    #print(game)
    #print(game.next().board().fen())
                
    #print(temp_game.variations)


def train_from_processed(path, num_epochs=5):
    print("Building Dataset...")
    data = np.genfromtxt(path, delimiter = ',', dtype=str)

    labels = []
    boards = []
    
    for row in data:
        temp_board, turn_mod = convert_fen(row[1])
        temp_label = float(row[0]) * (1 - (2*turn_mod))

        labels.append(temp_label)
        boards.append(temp_board)
       
    train_data, test_data, train_labels, test_labels = train_test_split(np.array(boards), np.array(labels), test_size=0.20, shuffle=True)

    print("Beginning Training")
    model = models.Sequential()
    #model.add(layers.Conv2D(1024, (8, 8), activation='relu', input_shape=(8, 8, 14)))
    
    #model.add(layers.Flatten())

    model.add(layers.Dense(2048, activation='relu', input_shape = [768]))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1))

    model.load_weights("SamePerspective.h5")

    #model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.mae,
              metrics=['mae'])

    history = model.fit(train_data, train_labels, epochs=num_epochs, 
                    validation_data=(test_data, test_labels))

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

    print(test_acc)

    model.save_weights("SamePerspectiveMore.h5")

