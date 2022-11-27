# Convertendo tempo real em tempo simulado

import numpy as np
import matplotlib.pyplot as plt

lvl3 = np.load('level3.npy')[0]
lvl4 = np.load('level4.npy')[0]
pump1pc = np.load('pump1pc.npy')[0]

d_pump1pc = np.gradient(pump1pc)
idx = [0]

for i in range(len(d_pump1pc)):
    if d_pump1pc[i] != 0 and i%2 == 0:
        idx.append(i)

idx.append(len(d_pump1pc))

Pi = np.array([0, 1, 2, 3, 4]) * 2500
Pf = np.array([1, 2, 3, 4, 5]) * 2500

t_c = []

for i in range(0, len(Pi)):
    t_c.append(np.arange(Pi[i], Pf[i]-(2500/(idx[i+1]-idx[i])), 2500/(idx[i+1]-idx[i])))


t_converted = np.hstack((t_c[0],t_c[1],t_c[2],t_c[3],t_c[4],12499.5,12499.75,12500))

np.save('time.npy', t_converted)

print(len(t_converted), len(lvl3), len(lvl4))
