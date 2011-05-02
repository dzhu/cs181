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

def init_point_settings(plant_bonus, plant_penalty, observation_cost,
                        starting_life, life_per_turn):
  '''Called before any moves are made.  Allows you to make customizations based
  on the specific scoring parameters in the game.'''
  move_generator.init_point_settings(plant_bonus, plant_penalty, observation_cost, starting_life, life_per_turn)
