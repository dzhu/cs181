#!/usr/bin/python
import nn
n = nn.read_from_file('net.pic')
d = nn.load_data('project/test_img')

f0 = [nn.feed_forward(n, q.listDblFeatures)[0] for q in d if q.iLabel == 0]
f1 = [nn.feed_forward(n, q.listDblFeatures)[0] for q in d if q.iLabel == 1]

print >>open('d0', 'w'), '\n'.join(map(str, sorted(f0)))
print >>open('d1', 'w'), '\n'.join(map(str, sorted(f1)))
