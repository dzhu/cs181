import common

class MoveGenerator():
  '''You can keep track of state by updating variables in the MoveGenerator
  class.'''
  def __init__(self):
    self.calls = 0

  def get_move(self, view):
    self.calls += 1
    return common.get_move(view)

  def init_point_settings(self, plant_bonus, plant_penalty, observation_cost,
                          starting_life, life_per_turn):
    self.plant_bonus = plant_bonus
    self.plant_penalty = plant_penalty
    self.observation_cost = observation_cost
    self.starting_life = starting_life
    self.life_per_turn = life_per_turn

move_generator = MoveGenerator()

def get_move(view):
  '''Returns a (move, bool) pair which specifies which move to take and whether
  or not the agent should try and eat the plant in the current square.  view is
  an object whose interface is defined in python_game.h.  In particular, you can
  ask the view for observations of the image at the current location.  Each
  observation comes with an observation cost.
  '''

  hasPlant = view.GetPlantInfo()

  # TODO: Decide on a direction, and whether or not to eat
  dir = common.game_interface.UP
  eat = False

  if (hasPlant):
    info = init_observation_info(view)
    (should_observe_again, info) = decide_observe(view, info)
    while (should_observe_again):
      (should_observe_again, info) = decide_observe(view, info)

    # Decide whether to eat
    # max_a \sum_s P(o|s)P(s)R(s,a); states are poisonous or nutritious
    # P(s) is the prior on how likely a plant is to be poisonous. etc. 
    expected_utility_eat =    prob_obs_given_state(view, info, 1) * prior_nutritious(view) * move_generator.plant_bonus \
                            + prob_obs_given_state(view, info, 0) * (1.0 - prior_nutritious(view)) * move_generator.plant_penalty
    eat = (expected_utility_eat > 0) # maybe this shouldn't be >0 
  
  return (dir, eat)

def prob_obs_given_state(view, info, is_nutritious):
  # TODO: implement
  # P(o|s), i.e probability that observations return info given state is or is 
  # not nutritious
  return 1.0

def prior_nutritious(view):
  # return (# nutritious)/(total plants observed), probably from running offline...
  # TODO: implement
  return 0.5

def init_observation_info(view):
  #TODO: implement. reinitializes things for first observation of a given plant
  return 0

def decide_observe(view, info):
  #TODO: implement
  if (info == 5):
    return (False, 0)
  return (True, info+1)

########### specific implementations: finite state controller ##############

def init_observation_info__FSC(view):
  return (0,0) # (nutritions observations, poisonous observations)

def decide_observe___FSC(view, info, is_nutritious):
  # if the total cost incurred from observing exceeds benefit from eating, stop.
  if (move_generator.observation_cost * (info[0]+info[1]+1) > move_generator.plant_bonus):
    return (False, info)
  # if you're about to die -- TODO: consider other bounds? would we pursue a different
  # strategy given different quantities of energy?
  if (move_generator.observation_cost >= view.GetLife()):
    return (False, info)
  # stopping condition
  if (abs(info[0]-info[1])>2): # maybe 2 should be 3 or something 
    return (False, info)
  is_nutritious = is_nutritious_by_NN(view.getImage())
  if (is_nutritious):
    return (True, (info[0]+1,info[1]))
  return (True, (info[0],info[1]+1))

########### specific implementations: MDP, value iteration  #################

def init_observation_info__VI(view):
  return 1

def decide_observe___VI(view, info, is_nutritious):
  # run a finite state value iteration
  # horizon H: the max number of observations you will make before doing dumb things
  H = (int)(move_generator.plant_bonus / move_generator.observation_cost)
  # TODO: bound H or use the other strategy, since states are O(H^2)
  # Store the k-step-to-go value and policy for each state and each k
  # states are: energy left x number nutritious observations x number poisonous observations
  
  V = [ [ 0 for p in range(H) ] for n in range(H) ] 
  # for k=1 to H (approximately)
  for k in range(H): 
    V_old = V
    V = [ [ 0 for p in range(H) ] for n in range(H) ] 
    Q_eat     = [ [ 0 for p in range(H) ] for n in range(H) ] 
    Q_not_eat = [ [ 0 for p in range(H) ] for n in range(H) ] 
    pistar = [ [ 0 for p in range(H) ] for n in range(H) ] # 0 for don't eat, 1 for eat TODO change to enums
    # for every state
    for n in range(H):
      for p in range(H):
        # for each action:
        # EAT
        Q_eat[n][p] = expected_reward_eat( (n,p) ) # TODO: define this function. 
        # NOT EAT 
        Q_not_eat[n][p] = 0
        # add \sum_{s'} P(s'|s,a)V_{k-1}(s')
        # so for every neighboring state, i.e. n+1 or p+1
        Q_eat[n][p] +=     T( (n,p), True,  True )
        Q_eat[n][p] +=     T( (n,p), False, True )
        Q_not_eat[n][p] += T( (n,p), True,  False)
        Q_not_eat[n][p] += T( (n,p), False, False)

        # optimal policy
        if (Q_eat[n][p] > Q_not_eat[n][p]):
          pistar[n][p] = 1
          V[n][p] = Q_eat[n][p]
        else: 
          V[n][p] = Q_not_eat[n][p]
        # new V

def T( s, observe_nutritious, eat ): #TODO: learn this offline. 
  # TODO: implement
  return 0.0

def expected_reward_eat(info):
  # TODO: implement
  return 0.5

def is_nutritious_by_NN(plant_image):
  #TODO: implement
  return True

def init_point_settings(plant_bonus, plant_penalty, observation_cost,
                        starting_life, life_per_turn):
  '''Called before any moves are made.  Allows you to make customizations based
  on the specific scoring parameters in the game.'''
  move_generator.init_point_settings(plant_bonus, plant_penalty, observation_cost, starting_life, life_per_turn)
