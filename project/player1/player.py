import common
import game_interface as game
import nn

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
    sys.stdout = self.log

    x, y = view.GetXPos(), view.GetYPos()
    #print x, y, self.targetx, self.targety

    life = view.GetLife()

    if self.last_image:
      dlife = life - self.last_life
      print self.lastx, self.lasty,
      if dlife == self.plant_bonus - self.life_per_turn:
        print 1,
      elif dlife == -self.plant_penalty - self.life_per_turn:
        print 0,
      else:
        print 2,#raise Exception

      print ''.join(map(str, self.last_image))

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

    if (x, y) not in self.seen_pos and view.GetPlantInfo() == game.STATUS_UNKNOWN_PLANT:
      self.last_image = view.GetImage()
    else:
      self.last_image = None

    eat = self.last_image and nn.feed_forward(self.plant_net, self.last_image)[0] > .5
    #if self.last_image: print nn.feed_forward(self.plant_net, self.last_image)[0]

    self.seen_pos.add((x, y))
    self.last_life = view.GetLife()
    self.lastx = x
    self.lasty = y

    self.log.flush()
    sys.stdout = sys.__stdout__

    if self.targetx > 25: raise Exception
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

move_generator = MoveGenerator()
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
