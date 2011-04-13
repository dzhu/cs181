#
# Darts playing model for CS181.
#

import sys
import time
import random
import throw
import mdp
import modelbased
import modelfree

GAMMA = .5
EPOCH_SIZE = 10


# <CODE HERE>: Complete this function, which should return a
# list of all possible states.
def get_states():
  # should return a **list** of states. Each state should be an integer.
  return []

# Returns a list of all possible actions, or targets, which include both a
# wedge number and a ring.
def get_actions():

  actions = []
  
  for wedge in throw.wedges:
    actions = actions + [throw.location(throw.CENTER, wedge)]
    actions = actions + [throw.location(throw.INNER_RING, wedge)]
    actions = actions + [throw.location(throw.FIRST_PATCH, wedge)]
    actions = actions + [throw.location(throw.MIDDLE_RING, wedge)]
    actions = actions + [throw.location(throw.SECOND_PATCH, wedge)]
    actions = actions + [throw.location(throw.OUTER_RING, wedge)]
    
  return actions

# <CODE HERE>: Define the reward function
def R(s,a):
  # takes a state s and action a
  # returns the reward for completing action a in state s
  return 0.0;


# Play a single game 
def play(method):
    score = throw.START_SCORE
    turns = 0
    
    if method == "mdp":
        target = mdp.start_game(GAMMA)
    else:
        target = modelfree.start_game()
        
    targets = []
    results = []
    while(True):
        turns = turns + 1
        result = throw.throw(target)
        targets.append(target)
        results.append(result)
        raw_score = throw.location_to_score(result)
        if raw_score <= score:
            score = int(score - raw_score)
        else:
            cc=1
        if score == 0:
            break

        if method == "mdp":
            target = mdp.get_target(score)
        else:
            target = modelfree.get_target(score)
            
   # print "WOOHOO!  It only took", turns, " turns"
    #end_game(turns)
    return turns

# Play n games and return the average score. 
def test(n, method):
    score = 0
    for i in range(n):
        score += play(method)
        
    return float(score)/float(n)
  




if __name__ =="__main__":
    main()




