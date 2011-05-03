import common
import game_interface as game
import nn
import math
import sys

outfile = 'log'

class MoveGenerator():
  '''You can keep track of state by updating variables in the MoveGenerator
  class.'''
  def __init__(self):
    self.log = open(outfile, 'w')
    self.last_life = None

  def get_move(self, view):
    life = view.GetLife()

    print >>self.log, view.GetXPos(), view.GetYPos()
    # print >>self.log, view.GetImage()
    # print >>self.log, view.GetPlantInfo()
    # self.log.flush()
    return common.get_move(view)

  def init_point_settings(self, plant_bonus, plant_penalty, observation_cost,
                          starting_life, life_per_turn):
    self.plant_bonus = plant_bonus
    self.plant_penalty = plant_penalty
    self.observation_cost = observation_cost
    self.starting_life = starting_life
    self.life_per_turn = life_per_turn

    print >>self.log, plant_bonus, plant_penalty, observation_cost, starting_life, life_per_turn
    self.log.flush()


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
  r = math.sqrt(x**2 + y**2)
  f2 = f/(h+math.exp(g*r+j))+a*r**2+d+c*r

  return f2/(f2+0.15)

class ExploreMoveGenerator():
  def __init__(self):
    self.log = open(outfile, 'w')

    self.plant_net = nn.read_from_file('net.pic')

    self.last_life = None
    self.last_image = None
    self.is_plant = False

    self.lastx = self.lasty = self.targetx = self.targety = 0

    self.seen_pos = set()

  def next_target(self, x, y):
    if x >= -y and x <= y:
      return x+1, y
    elif x > y and x <= -y:
      return x-1, y
    elif y <= x and y > -x:
      return x, y-1
    else:
      return x, y+1



  def get_move(self, view):

    x, y = view.GetXPos(), view.GetYPos()
    #print x, y, self.targetx, self.targety

    if x == self.targetx and y == self.targety:
      while (self.targetx, self.targety) in self.seen_pos:
        self.targetx, self.targety = self.next_target(self.targetx, self.targety)

    dx, dy = self.targetx - x, self.targety - y
    if abs(dx) < abs(dy): #(ternary operators would be nice, but are not in 2.5)
      if dy < 0:
        move = game.DOWN
      else:
        move = game.UP
    else:
      if dx < 0:
        move = game.LEFT
      else:
        move =  game.RIGHT

    eat = (prior_nutritious(x,y)>=0.5)

    return move, eat

  def init_point_settings(self, plant_bonus, plant_penalty, observation_cost,
                          starting_life, life_per_turn):
    self.plant_bonus = plant_bonus
    self.plant_penalty = plant_penalty
    self.observation_cost = observation_cost
    self.starting_life = starting_life
    self.life_per_turn = life_per_turn

    #print >>self.log, plant_bonus, plant_penalty, observation_cost, starting_life, life_per_turn
    self.log.flush()

#move_generator = MoveGenerator()
move_generator = ExploreMoveGenerator()

def get_move(view):
  '''Returns a (move, bool) pair which specifies which move to take and whether
  or not the agent should try and eat the plant in the current square.  view is
  an object whose interface is defined in python_game.h.  In particular, you can
  ask the view for observations of the image at the current location.  Each
  observation comes with an observation cost.
  '''
  return move_generator.get_move(view)

def init_point_settings(plant_bonus, plant_penalty, observation_cost,
                        starting_life, life_per_turn):
  '''Called before any moves are made.  Allows you to make customizations based
  on the specific scoring parameters in the game.'''
  move_generator.init_point_settings(plant_bonus, plant_penalty, observation_cost, starting_life, life_per_turn)
