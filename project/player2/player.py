import common
import nn
import game_interface
import math
class MoveGenerator():
  '''You can keep track of state by updating variables in the MoveGenerator
  class.'''
  def __init__(self):
    self.calls = 0

  def get_move(self, view):
    self.calls += 1
#    return common.get_move(view)
    get_move(view)

  def init_point_settings(self, plant_bonus, plant_penalty, observation_cost,
                          starting_life, life_per_turn):
    self.plant_bonus = plant_bonus
    self.plant_penalty = plant_penalty
    self.observation_cost = observation_cost
    self.starting_life = starting_life
    self.life_per_turn = life_per_turn

move_generator = MoveGenerator()
NN_ACCURACY = 0.68
NN_POISONOUS_ACCURACY = 0.68
NN_NUTRITIOUS_ACCURACY = 0.57
PROB_POISONOUS = 0.15

def expected_utility_eat(n,p,x,y):
  B = prob_obs_given_state(n,p,True)*prior_nutritious(x,y) + prob_obs_given_state(n,p,False)*(1.0-prior_nutritious(x,y))
  return prob_obs_given_state(n,p, True) * prior_nutritious(x,y) / B * move_generator.plant_bonus \
       - prob_obs_given_state(n,p, False) * (1.0 - prior_nutritious(x,y)) / B * move_generator.plant_penalty
def decide_eat(n,p,x,y):
  expected_utility = expected_utility_eat(n,p,x,y)  
  eat = (expected_utility > 0) 
  return eat

# info is a pair of readings (# nutritious, # poisonous)
def prob_obs_given_state(n,p, is_nutritious):
  # P(o|s), i.e probability that observations return info given state is or is 
  if is_nutritious:
    return NN_NUTRITIOUS_ACCURACY**(n)*(1-NN_NUTRITIOUS_ACCURACY)**(p)
  return NN_POISONOUS_ACCURACY**(p)*(1-NN_POISONOUS_ACCURACY)**(n)
  

def prior_nutritious(x,y):
  # return (# nutritious)/(total plants observed), probably from running offline...
  # probability that given a location, it's poisonous is always 0.15
  # probability that given a location, it's nutritious is approx
# f2(x) = (f)/(h+exp(g*x+j))+a*x**2+d+c*x
  
  f = 0.547293
  g = 0.553155
  j = -6.93274
  h = 9.40699
  a = 3.30966e-05
  d = 0.0861578
  c = -0.00327286

  r = (x**2+y**2)**(0.5)
  f2 = f/(h+math.exp(g*r+j))+a*r**2+d+c*r

  return f2/(f2+PROB_POISONOUS)


########### specific implementations: finite state controller ##############

def init_observation_info__FSC(view):
  return (0,0) # (nutritions observations, poisonous observations)

def decide_observe__FSC(n,p,x,y,view):
  # if the total cost incurred from observing exceeds benefit from eating, stop.
  # times 2, just because. arbitrary?
#  if (move_generator.observation_cost * (info[0]+info[1]+1) * 2 > min(move_generator.plant_bonus,view.GetLife()) ):
#    return (False, info)
  # don't waste too much energy
  if (n+p>=5):
    return False
  # if you're about to die -- TODO: consider other bounds? would we pursue a different
  # strategy given different quantities of energy?
  if (move_generator.observation_cost * 2 >= view.GetLife() ):
    return False
  # stopping condition
  if (abs(n-p)>1): # maybe 2 should be 3 or something 
    return False
  return True
########### specific implementations: MDP, value iteration  #################

# run a finite state value iteration
# horizon H: the max number of observations you will make before doing dumb things

# TODO: bound H or use the other strategy, since states are O(H^2)
# Store the k-step-to-go value and policy for each state and each k
# states are: energy left x number nutritious observations x number poisonous observations
# actions: observe or not observe. if you observe you move into a new state.

def init_observation_info__VI(view):
  return (0,0)

def decide_observe__VI(n,p,x,y,view):

  H = int(move_generator.plant_bonus / move_generator.observation_cost)
  V = [ [ 0 for p in range(H) ] for n in range(H) ] 

  Q_not_obs = [ [ 0 for p in range(H) ] for n in range(H) ] 
  for n in range(H):
    for p in range(H):
      Q_not_obs[n][p]=expected_reward_obs(n,p,x,y) # just the expected reward from the given (n,p)
  # for k=1 to H (approximately)

  pistar = [ [ False for p in range(H) ] for n in range(H) ] 
  for k in range(H): 
    V_old = V
    V = [ [ 0 for p in range(H) ] for n in range(H) ] 
    Q_obs  = [ [ 0 for p in range(H) ] for n in range(H) ] 
    # for every state
    for n in range(H):
      for p in range(H):
        Q_obs[n][p] = -move_generator.observation_cost #expected_reward_obs(n, p, x,y) # TODO: this guy
        # add \sum_{s'} P(s'|s,a)V_{k-1}(s')
        # so for every neighboring state, i.e. n+1 or p+1
        # and for each action. but the only action that doesn't terminate is observing.
        if n+p+1<H:
          Q_obs[n][p] += T( n,p, True,  x,y ) * V_old[n+1][p]
          Q_obs[n][p] += T( n,p, False, x,y ) * V_old[n][p+1]
          #print " ----- %f %f" % (T(n,p,True,x,y),T(n,p,False,x,y))
        # optimal policy
        if (Q_obs[n][p] > Q_not_obs[n][p]):
          if k==H-1:
            pistar[n][p] = True
          V[n][p] = Q_obs[n][p]
        else: 
          V[n][p] = Q_not_obs[n][p]
        # new V
#  print "============================================================="
#  print [[  int(V[a][b]) for b in range(H) ] for a in range(H)]
#  print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
#  print [[ int(Q_obs[a][b]) for b in range(H) ] for a in range(H)]
  return pistar[n][p]

def T( n,p, observe_nutritious, x, y ): #TODO: learn this offline. 
  # TODO: implement
  # this is the probability that the next observation will be poisonous (P) or nutritious (N) (depending on observe_nutritious)
  # given that we've already had (n,p) nutritious/poisonous observations.
  # P( next obsv is N | (n,p) ) = P( next is N | (n,p), actually P ) P( actually P | (n,p) )
  #                             + P( next is N | (n,p), actually N ) P( actually N | (n,p) )
  #                             = P( next is N | actually P ) P( actually P | (n,p) )
  #                             + P( next is N | actually N ) P( actually N | (n,p) )
  # Note that P( actually N | (n,p) ) = P( (n,p) | actually N ) P( actually N ) / P( (n,p) ), and we'll call that denominator B...
  
  # P( (n,p) | actually N )
  P_np_N = prob_obs_given_state(n,p, True)
  P_np_P = prob_obs_given_state(n,p, False)
  P_N = prior_nutritious(x,y)
  P_P = 1-P_N
  B = P_np_N*P_N + P_np_P*P_P
  P_N_np = P_np_N*P_N / B
  P_P_np = P_np_P*P_P / B

  if observe_nutritious:
    P_nextN_np = (1-NN_POISONOUS_ACCURACY) * P_P_np + NN_NUTRITIOUS_ACCURACY * P_N_np
    return P_nextN_np
  else:
    P_nextP_np = NN_POISONOUS_ACCURACY * P_P_np + (1-NN_NUTRITIOUS_ACCURACY) * P_N_np
    return P_nextP_np

def expected_reward_obs(n, p, x,y):
  # figure out whether we'd eat or not, using our policy.
  # then figure out expected reward given the fixed policy of whether we eat or not
  eat = decide_eat(n,p,x,y)
  if eat:
    return expected_utility_eat(n,p,x,y)
  else:
    return 0

#def expected_reward_eat(n, p, view):
#  prior_n = prior_nutritious(view)
#  prior_p = 1 - prior_n
#
#  prior_n *= prob_obs_given_state(view, (n, p), 1)
#  prior_p *= prob_obs_given_state(view, (n, p), 0)

#  prob_n = prior_n / (prior_n + prior_p)
#  prob_p = prior_p / (prior_n + prior_p)

#  return move_generator.plant_bonus * prob_n - move_generator.plant_penalty * prob_p

net = nn.read_from_file('net.pic')

def is_nutritious_by_NN(plant_image):
#  print plant_image
  res = nn.feed_forward(net, plant_image)
  return res[0] > .5

def init_point_settings(plant_bonus, plant_penalty, observation_cost,
                        starting_life, life_per_turn):
  '''Called before any moves are made.  Allows you to make customizations based
  on the specific scoring parameters in the game.'''
  move_generator.init_point_settings(plant_bonus, plant_penalty, observation_cost, starting_life, life_per_turn)


if False:
  decide_observe = decide_observe__FSC
  init_observation_info = init_observation_info__FSC
else:
  decide_observe = decide_observe__VI
  init_observation_info = init_observation_info__VI

def try_observe(n,p,should_observe):
  if not should_observe:
    return (n,p)
  is_nutritious = is_nutritious_by_NN(view.GetImage())
  if (is_nutritious):
    n+=1
  else:
    p+=1
  return (n,p)
def get_move(view):
  '''Returns a (move, bool) pair which specifies which move to take and whether
  or not the agent should try and eat the plant in the current square.  view is
  an object whose interface is defined in python_game.h.  In particular, you can
  ask the view for observations of the image at the current location.  Each
  observation comes with an observation cost.
  '''

  hasPlant = view.GetPlantInfo() == game_interface.STATUS_UNKNOWN_PLANT


  # TODO: Decide on a direction, and whether or not to eat
  dir = common.game_interface.UP
  eat = False
  n=0
  p=0
  x=view.GetXPos()
  y=view.GetYPos()
  if (hasPlant):
    info = init_observation_info(view)
    should_observe_again  = decide_observe(n,p,x,y,view)
    (n,p)= try_observe(n,p,should_observe_again) 

    while should_observe_again:


      (should_observe_again, n,p) = decide_observe(n,p,x,y,view)

      (n,p)= try_observe(n,p,should_observe_again) 
    # Decide whether to eat
    # max_a \sum_s P(o|s)P(s)R(s,a); states are poisonous or nutritious
    # P(s) is the prior on how likely a plant is to be poisonous. etc. 
#    print "                                   %d %d ::: %d" % (info[0], info[1],expected_utility_eat)
#    print eat 
#    print "EATING: "
  return (dir, decide_eat(n,p,view.GetXPos(),view.GetYPos()))

