import numpy as np
import remap

v1 = np.random.uniform(-10, 10, 5).astype(np.int64)
v2 = np.random.uniform(-10, 10, 5).astype(np.int64)
c = remap.Remapper()

print "start"
for p in zip(v1.ravel(), v2.ravel()):
    print p
print ""

c.add_merges(v1, v2)
f, t = c.fetch()
f = f.astype(np.int64)
t = t.astype(np.int64)
for p in zip(f, t):
    print p
print ""

c.pack()
f, t = c.fetch()
f = f.astype(np.int64)
t = t.astype(np.int64)
for p in zip(f, t):
    print p
print ""

b = np.random.uniform(-10, 10, 10).astype(np.int64)
print "BEF", b
c.remap(b)
print "AFT", b
