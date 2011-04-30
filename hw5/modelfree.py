from random import *
import throw
import darts
import sys

# The default player aims for the maximum score, unless the
# current score is less than the number of wedges, in which
# case it aims for the exact score it needs.
#
# You may use the following functions as a basis for
# implementing the Q learning algorithm or define your own
# functions.

ACTIVE_STRATEGY = 1

ALPHA = .1

last_score = None
last_action = None
actions = None

Q = {} # keys are states; values are dictionaries with actions as keys

def start_game():
  global last_score, last_action, actions

  if last_score is None:
    actions = darts.get_actions()
    for s in darts.get_states():
      Q[s] =  {}
      for a in actions:
        Q[s][a] = 0.

  last_score = throw.START_SCORE

  print >>sys.stderr, 'start'

  last_action = throw.location(throw.INNER_RING, throw.NUM_WEDGES)
  print >>sys.stderr, last_action
  return last_action

def get_target(score, iterations=[0]):
  global last_score, last_action
  Q_learning(last_score, score, last_action)

  print >>sys.stderr, 'get_target(%d)' % score
  last_score = score

  if ACTIVE_STRATEGY == 1:
    to_explore = ex_strategy_one()
  else:
    iterations[0] += 1
    to_explore = ex_strategy_two(iterations[0])

  if to_explore:
    last_action = choice(actions)
  else:
    qs = [(v, k) for k, v in Q[score].iteritems()]
    last_action = max(qs)[1]

  print >>sys.stderr, last_action
  return last_action

def ex_strategy_one():
  return int(random() < .1)

def ex_strategy_two(num_iterations):
  return int(random() < 100. / (500 + num_iterations))


# The Q-learning algorithm:
def Q_learning(s, s2, a):
  q_new = (0. if s2 else 1.) + darts.GAMMA * max(Q[s2][a2] for a2 in Q[s2])

  Q[s][a] = ALPHA * q_new + (1 - ALPHA) * Q[s][a]


