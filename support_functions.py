import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
import hashlib
import dill
import itertools
from plotnine import *

class Tictoe:
    def __init__(self, size):
        self.size = size
        self.board_size = size*size
        self.board = np.zeros(self.board_size)
        self.letters_to_move = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'][:self.board_size]
        self.possible_next_moves = copy.deepcopy(self.letters_to_move)
        self.moves_made = ''
    def reset_board(self):
        self.board = np.zeros(self.board_size)
        self.letters_to_move = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'][:self.board_size]
        self.possible_next_moves = copy.deepcopy(self.letters_to_move)
        self.moves_made = ''
    def get_board(self):
        return np.copy(self.board.reshape([self.size, self.size]))
    def make_move(self, who, where, verbose=False):
        self.board[self.letters_to_move.index(where)] = who
        self.moves_made += where
        self.possible_next_moves.remove(where)
        if verbose:
            print(self.get_board())
            print('Is game done?: ', self.is_endstate())
        return [self.get_current_state(), self.get_reward(who), where]
    def get_sums_of_board(self):
        local_board = self.get_board()
        return np.concatenate([local_board.sum(axis=0),             # columns
                               local_board.sum(axis=1),             # rows
                               np.trace(local_board),               # diagonal
                               np.trace(np.fliplr(local_board))], axis=None)   # other diagonal
    def is_endstate(self):
        someone_won = len(np.intersect1d((self.size, -self.size), self.get_sums_of_board())) > 0
        draw = np.count_nonzero(self.board) == (self.size * self.size) - 1
        return someone_won or draw
    def get_reward(self, who):
        sums = self.get_sums_of_board()
        if self.size in sums:       # The 1 player won
            #print('1.', end='')
            return who * 10
        elif -self.size in sums:    # The -1 player won
            #print('-1.', end='')
            return who * -10
        elif np.count_nonzero(self.board) == (self.size * self.size) - 1:  # Draw
            #print('0.', end='')
            return 5
        else:
            return 0  
    def get_moves_made(self):
        return self.moves_made
    def get_current_state(self):
        return hashlib.sha1(self.get_board()).hexdigest()
    def get_possible_next_states(self):
        return [self.moves_made + next_move for next_move in self.possible_next_moves]
    def get_possible_next_moves(self):
        return self.possible_next_moves.copy()  # Make a copy to ensures things work out in the game loop when we make the next move

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

class Memoize_tree:
    '''
    From https://www.python-course.eu/python3_memoization.php
    '''
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        function_call_hash = args[1:]  # Note we skip the first argument, this is the tree that is always the same. Adding this would slow down the hashing procedure
        if function_call_hash not in self.memo:
            self.memo[function_call_hash] = self.fn(*args)
        return self.memo[function_call_hash]

@Memoize_tree
def minmax_tt(tree, current_id, is_max):
    #print('Dealing with id: ', current_id)
    current_node = tree[current_id] 
    if current_node.data.is_endstate():
        return current_node.data.get_value()
    children_of_current_id = tree.children(current_id)
    scores = [minmax_tt(tree, child.identifier, not is_max) for child in children_of_current_id]
    if is_max:
        return max(scores)
    else:
        return min(scores)

def determine_move(tree, current_id, is_max):
    potential_moves = tree.children(current_id)
    moves = [child.identifier[-1] for child in potential_moves]
    raw_scores = [minmax_tt(tree, child.identifier, not is_max) for child in potential_moves]
    #print(dict(zip(moves, raw_scores)))
    if is_max:
        return moves[raw_scores.index(max(raw_scores))]
    else:
        return moves[raw_scores.index(min(raw_scores))]   

class Player_vs_itself:
    def __init__(self, id, alpha = 0.5, gamma = 0.6, epsilon = 0.1):
        self.qtable = {}
        self.board_to_state = {}
        self.id = id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def get_qtable(self):
        return self.qtable
    def get_id(self):
        return self.id
    def set_params(self, 
                   alpha = 0.5,       # How fast do we learn from new info
                   gamma = 0.6,       # How much are we focused on the short or the long term. 1 = max long term, 0 is max short term
                   epsilon = 0.1):    # exploration vs exploitation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def make_move(self, game, opponent_qtable, verbose=False):
        if game.is_endstate():
            # If the game is done by the time we get to make a move, simply skip this step
            return game
        # Make a choice what move to take next
        possible_moves = game.get_possible_next_moves()
        current_state = game.get_current_state()
        
        # If the current_state does not exist in the qtable, insert it
        if current_state not in self.qtable:
            # New entry in the qtable, init to zero. 
            #self.qtable[current_state] = dict(zip([current_state + move for move in possible_moves], 
            #                                      np.random.uniform(0, 0.1, len(possible_moves))))  # initialize on a small amount of random noise. Promotes varyiety
            self.qtable[current_state] = dict(zip(possible_moves, 
                                                  np.zeros(len(possible_moves))))  
            
        # Insert epsilon choice here, exploit or explore
        if random.uniform(0, 1) < self.epsilon:
            new_state, reward, action = game.make_move(self.id, random.choice(possible_moves))   # Random choice
        else:  # Exploit our qtable
            new_state, reward, action = game.make_move(self.id, keywithmaxval(self.qtable[current_state]))   # Optimal choice
        if self.epsilon == 0:    # If we set epsilon to 0, we only want to play. No updating needed. 
            return game
        
        # Update the qtable
        old_value = self.qtable[current_state][action]
        try:
            next_max = max(-np.array(list(opponent_qtable[new_state].values())))
        except KeyError:  # In case the tree for next state has not been made yet, simply return 0
            next_max = 0
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        if verbose:
            print('old: ', old_value, 'new_value: ', new_value, 'alpha: ', self.alpha, 'reward: ', reward, 'gamma: ', self.gamma, 'next_max: ', next_max)
        self.qtable[current_state][action] = new_value

        if verbose:
            print(self.id, current_state, self.qtable[current_state], new_state, action, old_value, new_value)
        
        return game

class memory_dict(dict):
    def __init__(self, *args):
        dict.__init__(self, args)
        self.memory = {}

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        if key in self.keys():
            try:
                self.memory[key].append(val)
            except KeyError:
                self.memory[key] = [self[key], val]
        dict.__setitem__(self, key, val)
    
    def get_memory(self, key):
        return self.memory[key]
    
class Player_vs_tree:
    def __init__(self, id, tree, alpha = 0.5, gamma = 0.6, epsilon = 0.1):
        self.qtable = {}
        self.board_to_state = {}
        self.state_list = []
        self.id = id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.our_reward_lut = {10: -10, 5:5, 0:0, -10:10}
        self.tree = tree
    def get_qtable(self):
        return self.qtable
    def get_board_to_state(self):
        return self.board_to_state
    def get_id(self):
        return self.id
    def set_params(self, 
                   alpha = 0.5,       # How fast do we learn from new info
                   gamma = 0.6,       # How much are we focused on the short or the long term. 1 = max long term, 0 is max short term
                   epsilon = 0.1):    # exploration vs exploitation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def make_move(self, game):
        if game.is_endstate():
            # If the game is done by the time we get to make a move, simply skip this step
            return game
        # Make a choice what move to take next
        possible_moves = game.get_possible_next_moves()
        self.current_state = game.get_current_state()
        
        # If the current_state does not exist in the qtable, insert it
        if self.current_state not in self.qtable:
            # New entry in the qtable, init to zero. 
            self.board_to_state[self.current_state] = game.get_board()
            #action_vs_qvalue = memory_dict()    # We use a memory dict here so that we store the intermediate values
            action_vs_qvalue = dict()
            for action in possible_moves:
                action_vs_qvalue[action] = 0
            self.state_list.append(self.current_state)  # For plotting later
            self.qtable[self.current_state] = action_vs_qvalue
            
        # Insert epsilon choice here, exploit or explore
        if random.uniform(0, 1) < self.epsilon:
            self.new_state, self.reward, self.action = game.make_move(self.id, random.choice(possible_moves))   # Random choice
        else:  # Exploit our qtable
            self.new_state, self.reward, self.action = game.make_move(self.id, keywithmaxval(self.qtable[self.current_state]))   # Optimal choice
        
        return game  
    def make_computer_move(self, game):
        # When we play against the tree, we make the tree give us the next move it makes
        # so in essence we treat the tree as a black box that also changes the worldstate. 
        # And we also only learn from our own moves, and no longer access the alternative
        # qtable. 
        if self.id == 1:
            self.new_state_after_tree, self.reward_after_tree, self.action_after_tree = game.make_move(-1, determine_move(self.tree, game.get_moves_made(), False))
        if self.id == -1:
            self.new_state_after_tree, self.reward_after_tree, self.action_after_tree = game.make_move(1, determine_move(self.tree, game.get_moves_made(), True))
        return game
    def update_qtable(self):
        # Update the qtable
        old_value = self.qtable[self.current_state][self.action]
        try:
            next_max = max(self.qtable[self.new_state_after_tree].values())
        except KeyError:  # In case the tree for next state has not been made yet, simply return 0
            next_max = 0
        # Note that the reward we actually get is the reward after the tree has made its move. We then reverse that reward vs the lut to get our own. 
        new_value = (1 - self.alpha) * old_value + self.alpha * (self.our_reward_lut[self.reward_after_tree] + self.gamma * next_max)
        self.qtable[self.current_state][self.action] = new_value
    def plot_qtable(self):
        def get_for_sha1(sha1_state):
            def try_na(qt, action):
                # See if an action exists for a given state, return NA if not
                try:
                    return qt[action]
                except KeyError:
                    return np.nan

            text_lut = {0: np.nan, 1: 'X', -1: 'O'}
            return pd.DataFrame({'x': np.tile([1, 2, 3], 3), 'y': np.repeat([1, 2, 3], 3), 
                          'board_state': [text_lut[val] for val in self.board_to_state[sha1_state].flatten()], 
                          'q_values': [try_na(self.qtable[sha1_state], action) for action in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']], 
                          'sha1_state': sha1_state})
        plot_data = pd.concat([get_for_sha1(sha1) for sha1 in self.board_to_state.keys()])
        
        # Sort states from the beginning to later in the game
        #plot_data['board_na'] = plot_data['board_state'].isna()
        #board_state_sorted = plot_data.groupby(['sha1_state'])['board_na'].sum().sort_values(ascending=False).index
        plot_data['sha1_state'] = pd.Categorical(plot_data['sha1_state'], categories=self.state_list)
        
        return (
            ggplot(plot_data, aes(x = 'x', y = 'y')) + 
                geom_tile(aes(fill = 'q_values')) + 
                geom_text(aes(label = 'board_state'), color = 'white') + 
                scale_fill_gradient2() + 
                scale_y_reverse() + 
                facet_wrap('~ sha1_state') + 
                theme(figure_size=(22,22), axis_text=element_blank(), axis_ticks=element_blank(), 
                      strip_text_x = element_blank(), axis_title=element_blank())
        )

class Player_tree:
    def __init__(self, tree, id):
        self.tree = tree
        self.id = id
    def make_move(self, moves_made):
        if self.id == -1:
            return determine_move(self.tree, moves_made, False)
        else:
            return determine_move(self.tree, moves_made, True)

def play_tictactoe_vs_itself(no_episodes, player1, player2, verbose=False):
    tactoe = Tictoe(3)       

    p1_rewards = np.zeros(no_episodes)
    p2_rewards = np.zeros(no_episodes)
    paths = np.chararray(no_episodes, itemsize=9)
    for ep_idx in tqdm(range(no_episodes)):
        stop = False
        while not stop:
            tactoe = player1.make_move(tactoe, player2.get_qtable(), verbose=verbose)
            tactoe = player2.make_move(tactoe, player1.get_qtable(), verbose=verbose)
            if tactoe.is_endstate():
                p1_rewards[ep_idx] = tactoe.get_reward(player1.get_id())
                p2_rewards[ep_idx] = tactoe.get_reward(player2.get_id())
                paths[ep_idx] = tactoe.get_moves_made()
                tactoe.reset_board()
                stop = True
    return [player1, player2, p1_rewards, p2_rewards, paths]

import random
from tqdm import tqdm

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value
         
         
     Based on https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary"""  
     k=list(d.keys())
     # boltzmann
     v = np.array(list(d.values()))
     return k[int(random.choice(np.argwhere(v == np.amax(v))))]  # If there are multiple max values, choose randomly

class Player:
    def __init__(self, id, alpha = 0.5, gamma = 0.6, epsilon = 0.1):
        self.qtable = {}
        self.id = id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def get_qtable(self):
        return self.qtable
    def get_id(self):
        return self.id
    def set_params(self, 
                   alpha = 0.5,       # How fast do we learn from new info
                   gamma = 0.6,       # How much are we focused on the short or the long term. 1 = max long term, 0 is max short term
                   epsilon = 0.1):    # exploration vs exploitation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def make_move(self, game, opponent_qtable, verbose=False):
        if game.is_endstate():
            # If the game is done by the time we get to make a move, simply skip this step
            return game
        # Make a choice what move to take next
        possible_moves = game.get_possible_next_moves()
        current_state = game.get_current_state()
        
        # If the current_state does not exist in the qtable, insert it
        if current_state not in self.qtable:
            # New entry in the qtable, init to zero. 
            #self.qtable[current_state] = dict(zip([current_state + move for move in possible_moves], 
            #                                      np.random.uniform(0, 0.1, len(possible_moves))))  # initialize on a small amount of random noise. Promotes varyiety
            self.qtable[current_state] = dict(zip(possible_moves, 
                                                  np.zeros(len(possible_moves))))  
            
        # Insert epsilon choice here, exploit or explore
        if random.uniform(0, 1) < self.epsilon:
            new_state, reward, action = game.make_move(self.id, random.choice(possible_moves))   # Random choice
        else:  # Exploit our qtable
            new_state, reward, action = game.make_move(self.id, keywithmaxval(self.qtable[current_state]))   # Optimal choice
        if self.epsilon == 0:    # If we set epsilon to 0, we only want to play. No updating needed. 
            return game
        
        # Update the qtable
        # Example qtable entry
        #  
        #  qtable['abd'] = {'abde': 2.1, 'abdf': 1.3, etc}
        # 
        # where 'abde': 2.1 is the q-value of taking action 'e' in state 'abd'. 
        # Note that the qtable is ragged. Not all moves are possible from each state
        old_value = self.qtable[current_state][action]
        try:
            next_max = max(-np.array(list(opponent_qtable[new_state].values())))
        except KeyError:  # In case the tree for next state has not been made yet, simply return 0
            next_max = 0
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        if verbose:
            print('old: ', old_value, 'new_value: ', new_value, 'alpha: ', self.alpha, 'reward: ', reward, 'gamma: ', self.gamma, 'next_max: ', next_max)
        self.qtable[current_state][action] = new_value

        if verbose:
            print(self.id, current_state, self.qtable[current_state], new_state, action, old_value, new_value)
        
        return game

def play_tictactoe(no_episodes, player1, player2, verbose=False):
    tactoe = Tictoe(3)       

    p1_rewards = np.zeros(no_episodes)
    p2_rewards = np.zeros(no_episodes)
    paths = np.chararray(no_episodes, itemsize=9)
    for ep_idx in tqdm(range(no_episodes)):
        stop = False
        while not stop:
            tactoe = player1.make_move(tactoe, player2.get_qtable(), verbose=verbose)
            tactoe = player2.make_move(tactoe, player1.get_qtable(), verbose=verbose)
            if tactoe.is_endstate():
                p1_rewards[ep_idx] = tactoe.get_reward(player1.get_id())
                p2_rewards[ep_idx] = tactoe.get_reward(player2.get_id())
                paths[ep_idx] = tactoe.get_moves_made()
                tactoe.reset_board()
                stop = True
    return [player1, player2, p1_rewards, p2_rewards, paths]