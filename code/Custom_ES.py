#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
The Evolution Strategy can be summarized as the following term:
{mu/rho +, lambda}-ES

Here we use following term to find a maximum point.
{n_pop/n_pop + n_kid}-ES

Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt
import reuters
import evaluate_classification 
import pdb
import math
import reuters

# pdb.set_trace()


DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [1,1000]       # solution upper and lower bounds
N_GENERATIONS = 2
POP_SIZE = 2           # population size
N_KID = 2               # n kids per generation
SIGMA_EPSILON = 10      #Minimum sigma value
TAU = 1 / math.sqrt(2 * math.sqrt(N_KID))


#def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function

def F(x): return evaluate_classification.evaluate(reuters, x)

# find non-zero fitness for selection
def get_fitness(eval_pop): 
    
    # print(eval_pop.shape)
    # print(eval_pop)
    pred = [0]*len(eval_pop)
    for i in range(len(eval_pop)):
        pred[i] = F(eval_pop[i][0])
        
    # pred = [F(x[0]) for x in eval_pop]
    # print("$"*30)
    # print(pred)
    # print("$"*30)
    pred = np.asarray(pred)
    pred.reshape(-1, 1)
    # pdb.set_trace()
    return pred


def make_kid(pop, n_kid):
    
    # generate empty kid holder

    kids = {'DNA': np.zeros([POP_SIZE * N_KID, DNA_SIZE], dtype = int)}
    kids['mut_strength'] = np.zeros([POP_SIZE * N_KID, DNA_SIZE])
    
    for p in range(len(pop['DNA'])):
        for k in range(n_kid):
            kids['DNA'][p * N_KID + k] = pop['DNA'][p] #All children of parent p should be initialized to parent p
            kids['mut_strength'][p * N_KID + k] = pop['mut_strength'][p]
        
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # pdb.set_trace()
        # crossover (roughly half p1 and half p2)
        # p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
        # cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points

        # mutate (change DNA based on normal distribution)
        ks[:] = ks[:] * math.exp(TAU * np.random.normal()) * 10
        if(ks[0] < SIGMA_EPSILON):
            ks[0] = SIGMA_EPSILON
        # ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        # print("ks: {}".format(ks))
        mutated_value = kv + ks * np.random.randn(*kv.shape) 
      
        # kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(int(mutated_value), *DNA_BOUND)   # clip the mutated value

        
    
    # pdb.set_trace()    
    # kids['DNA'] = np.array(kids['DNA'].flatten().tolist())
    
    # kids['mut_strength'] = np.array(kids['mut_strength'].flatten().tolist())
    for k, v in kids.items():
        print("kids[" + str(k) + "]: " + str(v))
        
    # pdb.set_trace()

    return kids


def kill_bad(pop, kids):
    selected_fitness = []
    # pdb.set_trace()    

    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))
        
    # print("#"*30)    
    # print(pop)
    # print("#"*30)

    fitness = get_fitness(pop['DNA'])            # calculate global fitness
    print("fitness for {} is {}".format(pop['DNA'], fitness))
    # pdb.set_trace()
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    best_fitnesses = fitness[good_idx]

    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
        
    return pop, best_fitnesses


def execute_es():
    pop = dict(DNA= np.random.choice(1001, POP_SIZE, replace=False),   # initialize the pop DNA values
           mut_strength= np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values
    
    pop['DNA'] = pop['DNA'].reshape(-1,1)
    pop['mut_strength'] = 10 * pop['mut_strength']
    
    
    
    for k, v in pop.items():
        print("pop[" + str(k) + "]: " + str(v))
    # plt.ion()       # something about plotting
    # x = np.linspace(*DNA_BOUND, 200)
    # plt.plot(x, F(x))

    for _ in range(N_GENERATIONS):
        # something about plotting
        # if 'sca' in globals(): sca.remove()
        # sca = plt.scatter(pop['DNA'], F(pop['DNA']), s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

        # ES part
        kids = make_kid(pop, N_KID)
        best_pop, best_fitness = kill_bad(pop, kids)   # keep some good parent for elitism

    # plt.ioff(); plt.show()
        
    best_index = np.argmax(best_fitness) 
    return best_pop['DNA'][best_index], best_fitness[best_index]


best, best_fit = execute_es()
print("#"*40)
print(best, best_fit)
print("#"*40)