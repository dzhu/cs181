import sys
import time
import random
import throw
import darts
import mdp
import modelbased
import modelfree
import math

import unittest

class MDPTest(unittest.TestCase):
    # test for learn_from_labeled_data()
    def test_get_states(self):
        x = throw.START_SCORE;
        self.assertTrue(x/2 < len(darts.get_states() ));
        self.assertTrue(0.75* x < len(darts.get_states()));

    def test_R(self):
        self.assertEqual(darts.R(10,0), 0)
        self.assertTrue (darts.R(0,0) > 0)
    
    def test_T(self) :
        def act(r,w):
                return throw.location(r,w)
        
        self.assertEqual(mdp.T( act(throw.CENTER, 1), 100, 110), 0.0) 
        self.assertEqual(mdp.T( act(throw.CENTER, 1), 100, 80), mdp.T( act(throw.CENTER,1), 90, 70));
        bullseye = throw.location_to_score(throw.location(throw.CENTER, 1));
        self.assertEqual( mdp.T(act(throw.FIRST_PATCH, 1), 100, 100-bullseye), 0.1);  
        self.assertEqual( mdp.T(act(throw.INNER_RING, 1), 100, 95), 0.5);  
        
