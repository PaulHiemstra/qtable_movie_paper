from support_functions import *
import numpy as np
import dill
import itertools
import sys
import warnings
warnings.filterwarnings('ignore')

print('Loading tree...')
with open('tree_tctoe_3x3.pkl', 'rb') as f:
    tree = dill.load(f)

print('Precomputing best moves...')
all_states = []
for length in range(1,9):
    tree_states = [''.join(state) for state in list(itertools.permutations(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], r=length))]
    all_states.extend(tree_states)

for state in tqdm(all_states):
    try:
        move = determine_move(tree, state, False) 
    except:
        pass 

tictactoe = Tictoe(3)
player_tree = Player_vs_tree(1,
                            tree, 
                            alpha = 0.01,
                            gamma = 0.8,
                            epsilon = 0.1)

print('Starting the training loop...')
no_episodes = int(sys.argv[1])
rewards = np.zeros(no_episodes)
frame_counter = 0
plots = {}
for ep_idx in tqdm(range(no_episodes)):
    while not tictactoe.is_endstate():
        tictactoe = player_tree.make_move(tictactoe)
        tictactoe = player_tree.make_computer_move(tictactoe)
        player_tree.update_qtable()
        
    rewards[ep_idx] = tictactoe.get_reward(1)
    
    # Specifically meant for plotting the frames in the YT movie
    if ep_idx % 2000 == 0:
        #ggsave(player_tree.plot_qtable(), filename='plots/qtable_ep%06d.png' % frame_counter)
        plots[frame_counter] = player_tree.plot_qtable()
        frame_counter += 1
    tictactoe.reset_board()
print('Training finished...')
print('Generating frames...')
from tqdm.contrib.concurrent import process_map  # or thread_map
#import matplotlib.pyplot as plt
#plt.ioff()

def f(frame_id):
    ggsave(plots[frame_id], filename='plots/qtable_ep%06d.png' % int(frame_id))
    
frame_list = list(plots.keys())
process_map(f, frame_list, max_workers=5)
print('done...')