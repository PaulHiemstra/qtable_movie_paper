import numpy as np
import copy
import dill # Alternative to pickle
from treelib import Node, Tree

class Tictoe:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros(size*size)
        self.letters_to_move = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'][:size*size]
    def get_board(self):
        return self.board.reshape([self.size, self.size])
    def make_move(self, who, where, verbose=False):
        self.board[self.letters_to_move.index(where)] = who
    def get_sums_of_board(self):
        local_board = self.get_board()
        return np.concatenate([local_board.sum(axis=0),             # columns
                               local_board.sum(axis=1),             # rows
                               np.trace(local_board),               # diagonal
                               np.trace(np.fliplr(local_board))], axis=None)   # other diagonal
    def is_endstate(self):
        someone_won = len(np.intersect1d((self.size, -self.size), self.get_sums_of_board())) > 0
        draw = np.count_nonzero(self.board) == self.size * self.size
        return someone_won or draw
    def get_value(self):
        sums = self.get_sums_of_board()
        if self.size in sums:
            return 10 - np.count_nonzero(self.board)
        elif -self.size in sums:
            return -10 + np.count_nonzero(self.board)
        else:
            return 0

def remove_value_list(l, val):
    return [el for el in l if el != val]

flip_player = {1: -1, -1: 1}

possible_options = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

def add_options_to_node(tree, node, tt_data, player, remaining_options):
    for option in remaining_options:
        local_tt_data = copy.deepcopy(tt_data)           # To prevent changing these values in other branches of the tree
        local_tt_data.make_move(player, option, False)
        if node.identifier != 'root':
            new_identifier = node.identifier + option
        else:
            new_identifier = option
        tree.create_node(option, new_identifier, node.identifier, data = local_tt_data)
        if len(remaining_options) > 1 and not local_tt_data.is_endstate():
            add_options_to_node(tree, tree[new_identifier], local_tt_data, 
                                flip_player[player], remove_value_list(remaining_options, option))
    return None

TicToe_state = Tictoe(3)
TicToe_3x3 = Tree()
TicToe_3x3.create_node("root", "root")
add_options_to_node(TicToe_3x3, TicToe_3x3["root"], 
                    TicToe_state, 1, possible_options)

with open('tree_tactoe_3x3.pkl', 'wb') as f:
    dill.dump(TicToe_3x3, f)