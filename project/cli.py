#!/usr/bin/python
import pickle, socket, sys, time, run_game, player1, player2

while True:
    s = socket.socket()

    try:
        s.connect(('htns.xvm.mit.edu', 1810))
    except socket.error:
        print "couldn't connect, trying again in 15 sec..."
        time.sleep(15)
        continue

    f = s.makefile()
    outfn = pickle.load(f)
    args = pickle.load(f)

    print 'args:', args
    print 'writing to file', outfn

    player1.player.outfile = outfn
    player1.player.move_generator = None
    run_game.main(args)

    f.close()
    s.close()
