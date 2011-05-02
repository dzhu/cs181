#!/usr/bin/python
import pickle, socket, time

serv = socket.socket()
serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
while True:
    try:
        serv.bind(('', 1810))
        break
    except socket.error:
        print 'bind error, trying again...'
        time.sleep(5)
serv.listen(16)
print 'waiting...'

hosts = set()

try:
    t0 = time.time()
    ns = range(500)
    for n, i in enumerate(ns):
        sock, addr = serv.accept()
        addr = addr[0][9:]
        hosts.add(addr)

        print 'sending %d (%d/%d) t=%.2f to %s, %d hosts' % (i, n, len(ns), time.time() - t0, addr, len(hosts))

        f = sock.makefile()

        pickle.dump('data/%04d' % i, f) #outfile
        pickle.dump(['cli.py', '--single_player_mode', '--starting_life', '10000000', '-d', '0'], f)

        f.close()
        sock.close()
except KeyboardInterrupt:
    pass

print 'closing'
serv.close()
