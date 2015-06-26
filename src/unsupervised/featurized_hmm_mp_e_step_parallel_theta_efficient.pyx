 #!/usr/bin/env python
# -*- coding: utf-8 -*-

from docutils.nodes import thead



__author__ = 'amansour'

# Painless Unsupervised Learning with Features
# Berg-Kirkpatrick et al.

from numpy import random, exp, zeros, ones, log, array_split
import time
from collections import defaultdict
from scipy.optimize import minimize
import nltk
from nltk.stem import *
import multiprocessing
from multiprocessing import Pool, RLock
global data_likelihood, grad, theta
from multiprocessing import Process, Value, Array
import ctypes as ct
import numpy as np

def initialize_w(feature_index, t_table):
    w = random.uniform(-1,1,len(feature_index))
    w = ones(len(feature_index))
    for (d, c) in t_table:
        features = get_features_fired(d, c)
        for f in features:
            w[feature_index[f]] = log(t_table[(d,c)])
    return w

#This function is called in the M-step after w is computed
#To update theta (parameters) in this case t_table
def get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map): #w is the weight vector, d is the decision and c is the context
    #Compute features fired
    #t_table = load_t_table();
    #print 'get_decision_context'
#    print 'd,c', d, c


    fired_features = get_features_fired(d, c)

    #w_dot_features = t_table[(d,c)]*w[feature_index[(d,c)]]
    w_dot_features = sum([ w[feature_index[f]] for f in fired_features])
#    print 'w_dot_features',w_dot_features
    all_decisions = normalizing_decision_map[c]

    sum_exp_normalizing_term = 0
#    print len(all_decisions)
    for d_prime in all_decisions:
        d_prime_features_fired = get_features_fired(d_prime,c)
        #w_dot_d_prime_features = t_table[(d_prime,c)]*w[feature_index[(d_prime,c)]]
        '''sum_exp = 0
        for f in d_prime_features_fired:
            sum_exp += w[feature_index[f]]
            if sum_exp > 1000:
                print 'w[featureindex] ', sum_exp
        '''
        sum_exp_normalizing_term += exp(sum([ w[feature_index[f]] for f in d_prime_features_fired]))
#        print sum_exp_normalizing_term
#    print 'sum_exp_normalizing_term',sum_exp_normalizing_term
#    print 'theta in thetat ',exp(w_dot_features)/sum_exp_normalizing_term

    return exp(w_dot_features)/sum_exp_normalizing_term

#computes theta for all d, c pairs. This should work more efficiently.

def get_theta(w, e, feature_index, normalizing_decision_map): #w is the weight vector, d is the decision and c is the context
    global theta
    theta = defaultdict(int)
    print 'w[emission,0,0]= ', w[feature_index[("EMISSION", ".", ".")]]
    N_c = {} #denominator computing for each c

    for (d, c) in e:
        fired_features = get_features_fired(d, c)
        w_dot_features = sum([ w[feature_index[f]] for f in fired_features])
        all_decisions = normalizing_decision_map[c]
        if c not in N_c:
            sum_exp_normalizing_term = 0
            for d_prime in all_decisions:
                d_prime_features_fired = get_features_fired(d_prime,c)
                sum_exp_normalizing_term += exp(sum([ w[feature_index[f]] for f in d_prime_features_fired]))
            N_c[c] = sum_exp_normalizing_term
        if d == '.' and c == '.':
            print '.*******.', exp(w_dot_features), ' ', N_c[c]
        theta[(d, c)] = exp(w_dot_features)/N_c[c]


def Expectation2(lock, t_table, N, Y, sd_size, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood):
    #print 'Expectation2'
    for y, x in Y[start:end]: #y is the source sentence and x is the target sentence
        T = len(y)
        N = len(x)
        c = defaultdict(int)

        if iterations == 0:
            a, pi = initializeBasedOnC(N)
        alpha_hat, c_scaled = forward_with_t_scaled(a, pi, y, N, T, x, t_table) #BE careful on indexing on x
        beta_hat = backward_with_t_scaled(a, pi, y, N, T, x, t_table, c_scaled) #else:
        #    alpha = forward(a, b, pi, y, N, T)
        #    beta = backward(a, b, pi, y, N, T)
        gamma = np.zeros((N + 1, T + 1))
        xi = zeros((N + 1, N + 1, T + 1))

        for t in range(1, T):
            #print 'c_scaled ',c_scaled[t]
            with lock:
                logLikelihood.value += -(log(c_scaled[t]))
            for i in range(1, N + 1):
                gamma[(i,t)] = (alpha_hat[(i,t)]*beta_hat[(i,t)])/c_scaled[t]
                with lock:
                    totalGammaOverAllObservations[i] += gamma[i, t]
                address = (i * sd_size) + indexMap[(y[t - 1], x[i - 1])]
                with lock:
                    totalGammaDeltaOverAllObservations_t[address] += gamma[i, t]
        t = T
        with lock:
            logLikelihood.value += -(log(c_scaled[t]))
        for i in range(1, N + 1): #
            gamma[(i,t)] = (alpha_hat[(i,t)]*beta_hat[(i,t)])/c_scaled[t]
            with lock:
                totalGammaOverAllObservations[i] += gamma[i, t] #               totalGammaDeltaOverAllObservations[(i,y[t-1])] += gamma[(i,t)]
            #totalGammaDeltaOverAllObservations_t[(i,y[t-1],x[i-1])] += gamma[(i,t)]
            address = (i * sd_size) + indexMap[(y[t - 1], x[i - 1])]
            with lock:
                totalGammaDeltaOverAllObservations_t[address] += gamma[i, t]
        for t in range(1, T):
            for i in range(1, N + 1):
                for j in range(1, N + 1):
#                        if iterations == 0:
                    xi[i, j, t] = alpha_hat[(i, t)] * a[(i, j)] * t_table[(y[t], x[j - 1])] * beta_hat[(j, t + 1)] #                        else:

#                            xi[(i,j,t)] = (alpha[(i,t)]*a[(i,j)]*b[(j,y[t])]*beta[(j,t+1)])/total[t]
#                        totalXiOverAllObservations[(i,j)] += xi[(i,j,t)]
        for i in range(1, N + 1):

            totalGamma1OverAllObservations[i] += gamma[i, 1]

        for d in range(-N - 1, N+1):
            #c[d] = 0
            for t in range(1, T):
                for i in range(1, N + 1):
                    if i + d <= N and i + d >= 1:
                        c[d] += xi[i, i + d, t]

        #print 'c',c
        #Liang et al. suggestion
#             for d in c:
#                 if d < -7:
#                     c[-7] += c[d]
#                     c[d] = 0
#                 if d > 7:
#                     c[7] += c[d]
#                     c[d] = 0
	#print 'totalC_j_Minus_iOverAllObservations',totalC_j_Minus_iOverAllObservations
        #print 'N',N
        for i in range(1, N + 1):
                for j in range(1, N + 1): #                     if j-i >= 7:
    #                         totalC_j_Minus_iOverAllObservations[(i,j)] += c[7]
    #                     elif j-i <= -7:
    #                         totalC_j_Minus_iOverAllObservations[(i,j)] += c[-7]
    #                     else:
                    with lock:
                        totalC_j_Minus_iOverAllObservations[(i,j)] += c[j - i]
                for l in range(1, N + 1): #                     if l-i >= 7:
#                         totalC_l_Minus_iOverAllObservations[i] += c[7]
#                     elif l-i <= -7:
#                         totalC_l_Minus_iOverAllObservations[i] += c[-7]
#                     else:
                    with lock:
                        totalC_l_Minus_iOverAllObservations[i] += c[l - i]


    #return N, i, a, pi, j

def em_with_features(bitext_sd, s_count,t_table,sd_count, feature_index, normalizing_decision_map,kappa, f_vector, event_index):#L is the number of observations

    w = initialize_w(feature_index, t_table)
    N = maxTargetSentenceLength(bitext_sd)
    print 'N',N
    N_max = N
    Y = bitext_sd

    yValues = s_count.keys()#computes all possible english words
    (indexMap,biword) = map_bitext_to_int(sd_count)
    sd_size = len(indexMap)
    lastLikelihood = 0


    #a = zeros((N+1,N+1))
    a_array = Array(ct.c_double,((N+1)*(N+1)))
    a_array2 = np.frombuffer(a_array.get_obj()) # mp_arr and arr share the same memory
    a = a_array2.reshape((N+1,N+1)) # b and arr share the same memory

    #pi = zeros(N+1)
    pi = Array('d', N+1)

    logLikelihood = Value('d',0.0)

    lastLogLikelihood = Value('d',0.0)

    L = len(Y)
    #N = len(Y[0][1]) #first sentence x length
    #(a,pi) = initializeUniformly(N)

    for iterations in range(0,10):
        #E step
        #c = defaultdict(int)
        startTime = time.time()
        print 'iteration',iterations
        logLikelihood.value = 0.0
        #totalGammaOverAllObservations = zeros(N+1)
        totalGammaOverAllObservations = Array('d', [0]*(N+1))

        #totalGammaDeltaOverAllObservations_t = zeros((N+1,sd_size))
        totalGammaDeltaOverAllObservations_t = Array('d',[0]*((N+1)*(sd_size)))

        #totalGammaDeltaOverAllObservations_t_overall_states = zeros(sd_size)
        totalGammaDeltaOverAllObservations_t_overall_states = Array('d',[0]*sd_size)

        totalGammaDeltaOverAllObservations_t_overall_states_over_dest = defaultdict(int)


        #totalGamma1OverAllObservations = zeros(N+1)
        totalGamma1OverAllObservations = Array('d',[0]*(N+1))

        #totalC_j_Minus_iOverAllObservations = zeros((N+1,N+1))
        totalC_j_Minus_iOverAllObservations_array = Array(ct.c_double,(N+1)*(N+1))
        totalC_j_Minus_iOverAllObservations_array2 = np.frombuffer(totalC_j_Minus_iOverAllObservations_array.get_obj())
        totalC_j_Minus_iOverAllObservations = totalC_j_Minus_iOverAllObservations_array2.reshape((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                totalC_j_Minus_iOverAllObservations[i,j] = 0.0

        #totalC_l_Minus_iOverAllObservations = zeros(N+1)
        totalC_l_Minus_iOverAllObservations = Array('d',[0]*(N+1))
        get_theta(w, indexMap, feature_index, normalizing_decision_map)

        if iterations == 0:
            for k in range(sd_size):
                (f, e) = biword[k]
                t_table[(f,e)] = theta[(f, e)]
        '''if iterations == 0:
            for k in range(sd_size):
                (f,e) = biword[k]
                t_table[(f, e)] = get_theta_decision_given_context(w, f, e, feature_index, normalizing_decision_map)
        '''

        intervals = 16
        jobs = []
        lock = RLock()
        length_of_interval = L/intervals
        for i in range(0,intervals-1):
            start = i*length_of_interval
            end = (i+1)*length_of_interval
            #print start
            #print end
            p = Process(target=Expectation2, args = (lock, t_table, N, Y, sd_size, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood))
            p.start()
            jobs.append(p)

        start = (intervals-1)*length_of_interval
        end = L
        p = Process(target=Expectation2, args = (lock, t_table, N, Y, sd_size, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood))
        p.start()
        jobs.append(p)
        for p in jobs:
            p.join()


        endTime = time.time()

        #print N
        print "%.2gs" % (endTime - startTime)
        #N = len(totalGamma1OverAllObservations)-1
        #print N
        print 'last , new ', lastLogLikelihood.value, logLikelihood.value
        #print 'likelihood difference ', (logLikelihood.value - lastLogLikelihood.value)
        lastLogLikelihood.value = logLikelihood.value

        totalGammaOverAllObservationsOverAllStates = 0.0
        sartTime = time.time()
        for i in range(1,N+1):
            totalGammaOverAllObservationsOverAllStates += totalGammaOverAllObservations[i]




        print 'likelihood ', logLikelihood
        #lastLikelihood = liklihood
        N = len(totalGamma1OverAllObservations)-1
        #print N

        # Create expected_counts(d,c) to be consistent with the Berg-Kirkpatrick et al.
        # To make it more memory efficient just keep either totalGammaDeltaOverAllObservations_t_overall_states or expected_counts
        expected_counts = defaultdict(int)

        for i in range(1,N+1):
            for k in range(sd_size):

                address = i*sd_size + k
                totalGammaDeltaOverAllObservations_t_overall_states[k] += totalGammaDeltaOverAllObservations_t[address]
                (f,e) = biword[k]
                totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e] += totalGammaDeltaOverAllObservations_t[address]


        for k in range(sd_size):
            (f, e) = biword[k]
            expected_counts[(f, e)] = totalGammaDeltaOverAllObservations_t_overall_states[k]

        totalGammaOverAllObservationsOverAllStates = 0.0
        for i in range(1,N+1):
            totalGammaOverAllObservationsOverAllStates += totalGammaOverAllObservations[i]

        # M Step
        # We can clear a, b and pi here and then set the values for them
        a = zeros((N+1,N+1))
        pi = zeros(N+1)
        t_table = defaultdict(int)

        for i in range(1,N+1):
            pi[i] = totalGamma1OverAllObservations[i]*(1.0/L)

            for j in range(1,N+1):
                a[(i,j)] = totalC_j_Minus_iOverAllObservations[(i,j)]/totalC_l_Minus_iOverAllObservations[i]

        # Climbing with LBFGS method
        #for step in range(0,5):
        #    print 'step ', step
        #result = minimize(get_likelihood_with_expected_counts_mp,w,args=(expected_counts,kappa,feature_index,normalizing_decision_map, f_vector, event_index), method='L-BFGS-B', jac=get_gradient_with_counts_serial ,options={'maxfun' : 5, 'ftol' : 0.1})
        result = minimize(get_likelihood_with_expected_counts_serial,w,args=(expected_counts,kappa,feature_index,normalizing_decision_map, f_vector, event_index), method='L-BFGS-B', jac=get_gradient_with_counts_parallel_first ,options={'maxfun' : 5, 'ftol' : 0.1})
        w = result.x

        global theta
        get_theta(w, expected_counts, feature_index, normalizing_decision_map)
        for k in range(sd_size):
            (f,e) = biword[k]
            t_table[(f,e)] = theta[(f, e)]
            #t_table[(f,e)] = get_theta_decision_given_context(w, f, e, feature_index, normalizing_decision_map)

        print iterations

    return (a,t_table,pi)


def batch_likelihood_with_expected_counts(w, e, batch, event_index, feature_index, normalizing_decision_map):
    global theta
    batch_sum_likelihood = 0.0
    for i in batch:
        (d, c) = event_index[i]
        #theta_d_c = get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map)
        batch_sum_likelihood += e[(d, c)]*log(theta[(d, c)])
    return batch_sum_likelihood

def batch_accumilate_likelihood_with_expected_counts(results):
    global data_likelihood
    data_likelihood += results

def get_likelihood_with_expected_counts(w, e, kappa,feature_index,normalizing_decision_map, f_vector):
    print 'get_likelihood_with_expected_counts'
    startTime = time.time()
    likelihood = 0
    for (d,c) in e :
        theta_d_c = get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map)
        likelihood += e[(d,c)]*log(theta_d_c)
    likelihood -=  kappa * sum(w**2)
    print 'likelihood_with_expected_count ', likelihood
    endTime = time.time()
    print "run time for likelihood_with_expected_count %.2gs" % (endTime - startTime)
    return -likelihood

def get_likelihood_with_expected_counts_mp(w, e, kappa,feature_index,normalizing_decision_map, f_vector, event_index):
    print 'get_likelihood_with_expected_counts'
    #print 'w', w
    startTime = time.time()

    global data_likelihood, theta
    get_theta(w, e, feature_index, normalizing_decision_map)
    #theta is computed here; so, we can use it from now on in the likelihood and gradient step

    data_likelihood = 0.0

    cpu_count = 10#multiprocessing.cpu_count()

    pool = Pool(processes=cpu_count) # uses all available CPUs

    batches_fractional_counts = array_split(range(len(event_index)), cpu_count)

    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_likelihood_with_expected_counts, args=(w, e, batch_of_fc, event_index, feature_index, normalizing_decision_map), callback=batch_accumilate_likelihood_with_expected_counts)
    pool.close()
    pool.join()

    data_likelihood -=  kappa * sum(w**2)
    print 'likelihood_with_expected_count ', data_likelihood
    endTime = time.time()
    print "run time for likelihood_with_expected_count %.2gs" % (endTime - startTime)
    
    return -data_likelihood

def get_likelihood_with_expected_counts_serial(w, e, kappa,feature_index,normalizing_decision_map, f_vector, event_index):
    print 'get_likelihood_with_expected_counts'
    #print 'w', w
    startTime = time.time()

    global data_likelihood, theta
    get_theta(w, e, feature_index, normalizing_decision_map)
    #theta is computed here; so, we can use it from now on in the likelihood and gradient step

    data_likelihood = 0.0
    for (d,c) in e :
        data_likelihood += e[(d,c)]*log(theta[(d,c)])

    data_likelihood -=  kappa * sum(w**2)
    print 'likelihood_with_expected_count ', data_likelihood
    endTime = time.time()
    print "run time for likelihood_with_expected_count %.2gs" % (endTime - startTime)

    return -data_likelihood

def get_features_fired(decision, context):
    fired_features = []

    #Lexical feature
    fired_features = [("EMISSION", decision, context)]

    #if decision == context:
    #    fired_features += [("IS_SAME", decision, context)]

   # english_stemmer = SnowballStemmer("english")
   # french_stemmer = SnowballStemmer("french")
    #print decision, context
   # fired_features += [("STEM", french_stemmer.stem(decision.decode("utf8")), english_stemmer.stem(context.decode("utf8")))]

   # if decision[0:3] == context[0:3]:
   #     fired_features += [("PREFIX", decision[0:3], context[0:3])]
    return fired_features


def get_gradient(w, e, kappa, feature_index, normalizing_decision_map, f_vector):
    print 'get_gradient'
    grad = -2 * kappa * w

    event_grad = {}

    for (d_prime, c) in e:

        theta_d_prime_c = get_theta_decision_given_context(w,d_prime,c,feature_index,normalizing_decision_map)

        norm_events = [(d, c) for d in normalizing_decision_map[c]]

        sum_c = 0
        for (d,c) in norm_events:
            if d_prime == d:
                f = 1.0 #assuming that feature values are 1 not t_table values
                #sum_c += e[(d,c)]
            else:
                f = 0.0
                #sum_c += -1*e[(d,c)]*theta_d_prime_c
            sum_c += e[(d,c)]*(f - theta_d_prime_c)
        event_grad[(d_prime,c)] = sum_c

    #events_to_features = get_event_to_features()

    for (d_event,c_event) in event_grad:
        #feats = events_to_features[event]
        feats = get_features_fired(d_event, c_event)
        for f in feats:
            grad[feature_index[f]] += event_grad[(d_event, c_event)]

    return -grad

def print_dictionary(dictionary):
    for k in dictionary:
        print k,': ',dictionary[k],'\n'

def get_gradient_with_counts(w, e, kappa, feature_index, normalizing_decision_map, f_vector):
    print 'get_gradient_with_counts'
    startTime = time.time()
    grad = -2 * kappa * w
    #grad = zeros(len(w))
    #precompute thetas
    theta = defaultdict(int)
    for (d, c) in e:
        theta[(d, c)] = get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map)
    print 'theta'
    print_dictionary(theta)
    for (d, c) in e:
        features_fired_list = get_features_fired(d, c)
        for feature in features_fired_list:
            grad[feature_index[feature]] += f_vector[feature_index[feature]]*e[(d, c)]
        norm_events = [(d_prime, c) for d_prime in normalizing_decision_map[c]]

        for (d_prime, c) in norm_events:
            features_fired_list_d_prime_c = get_features_fired(d_prime, c)
            for feature in features_fired_list_d_prime_c:
                grad[feature_index[feature]] -= theta[(d_prime, c)]*f_vector[feature_index[feature]]*e[(d, c)]
    endTime = time.time()
    print "run time for get_gradient_with_count %.2gs" % (endTime - startTime)
    print 'grad',-grad
    return -grad
def get_gradient_with_counts_serial(w, e, kappa, feature_index, normalizing_decision_map, f_vector, event_index):
    print 'get_gradient_with_counts'
    startTime = time.time()
    grad = -2 * kappa * w
    #grad = zeros(len(w))
    global theta
    #precompute thetas
    #theta = defaultdict(int)
    #for (d, c) in e:
    #    theta[(d, c)] = get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map)
    print 'theta'
    #print_dictionary(theta)
    for (d, c) in e:
        features_fired_list = get_features_fired(d, c)
        for feature in features_fired_list:
            grad[feature_index[feature]] += f_vector[feature_index[feature]]*e[(d, c)]
        norm_events = [(d_prime, c) for d_prime in normalizing_decision_map[c]]

        for (d_prime, c) in norm_events:
            features_fired_list_d_prime_c = get_features_fired(d_prime, c)
            for feature in features_fired_list_d_prime_c:
                grad[feature_index[feature]] -= theta[(d_prime, c)]*f_vector[feature_index[feature]]*e[(d, c)]
    endTime = time.time()
    print "run time for get_gradient_with_count %.2gs" % (endTime - startTime)
    #print 'grad',-grad
    return -grad

def batch_accumulate_gradient_with_counts(result):
    global theta
    for (d, c) in result:
        theta[(d, c)] = result[(d, c)]
def batch_accumulate_gradient_with_counts_grad(result):
    global grad
    grad += result


def get_gradient_with_counts_parallel_first(w, e, kappa, feature_index, normalizing_decision_map, f_vector, event_index):
    print 'get_gradient_with_counts'
    startTime = time.time()
    global grad
    grad = -2 * kappa * w
    #grad = zeros(len(w))
    #precompute thetas
    global theta
    '''
    theta = defaultdict(int)

    cpu_count = multiprocessing.cpu_count()

    pool = Pool(processes=cpu_count) # uses all available CPUs
    batches_fractional_counts = array_split(range(len(event_index)), cpu_count)

    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_theta, args=(w, e, batch_of_fc, feature_index, normalizing_decision_map, f_vector, event_index), callback=batch_accumulate_gradient_with_counts)
    pool.close()
    pool.join()
    endTime = time.time()
    print "run time for theta %.2gs" % (endTime - startTime)
    '''

    cpu_count = 16#multiprocessing.cpu_count()

    pool = Pool(processes=cpu_count) # uses all available CPUs
    batches_fractional_counts = array_split(range(len(event_index)), cpu_count)


    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_gradient, args=(w, e, batch_of_fc, feature_index, normalizing_decision_map, f_vector, event_index), callback=batch_accumulate_gradient_with_counts_grad)
    pool.close()
    pool.join()

    '''
    for (d, c) in e:
        features_fired_list = get_features_fired(d, c)
        for feature in features_fired_list:
            grad[feature_index[feature]] += f_vector[feature_index[feature]]*e[(d, c)]
        norm_events = [(d_prime, c) for d_prime in normalizing_decision_map[c]]

        for (d_prime, c) in norm_events:
            features_fired_list_d_prime_c = get_features_fired(d_prime, c)
            for feature in features_fired_list_d_prime_c:
                grad[feature_index[feature]] -= theta[(d_prime, c)]*f_vector[feature_index[feature]]*e[(d, c)]
    '''
    endTime = time.time()
    print "run time for get_gradient_with_count %.2gs" % (endTime - startTime)
    #print 'grad',-grad
    print "grad norm2 ",sum(grad**2) 
    return -grad


def batch_gradient(w, e, batch, feature_index, normalizing_decision_map, f_vector, event_index):
    global theta
    grad_batch = zeros(len(w))
    for i in batch:
        (d, c) = event_index[i]
        features_fired_list = get_features_fired(d, c)
        for feature in features_fired_list:
            grad_batch[feature_index[feature]] += f_vector[feature_index[feature]]*e[(d, c)]
        norm_events = [(d_prime, c) for d_prime in normalizing_decision_map[c]]

        for (d_prime, c) in norm_events:
            features_fired_list_d_prime_c = get_features_fired(d_prime, c)
            for feature in features_fired_list_d_prime_c:
                grad_batch[feature_index[feature]] -= theta[(d_prime, c)]*f_vector[feature_index[feature]]*e[(d, c)]
    return grad_batch

def batch_theta(w, e, batch, feature_index, normalizing_decision_map, f_vector, event_index):
    theta_batch = {}
    for i in batch:
        (d, c) = event_index[i]
        theta_batch[(d, c)] = get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map)
    return theta_batch

def get_gradient_with_counts_mp(w, e, kappa, feature_index, normalizing_decision_map, f_vector, event_index):
    print 'get_gradient_with_counts'
    startTime = time.time()
    global grad, theta
    grad = -2 * kappa * w

    theta = defaultdict(int)
    #precompute thetas
    #for (d, c) in e:
    #    theta[(d, c)] = get_theta_decision_given_context(w, d, c, feature_index, normalizing_decision_map)

    cpu_count = multiprocessing.cpu_count()

    pool = Pool(processes=cpu_count) # uses all available CPUs
    batches_fractional_counts = array_split(range(len(event_index)), cpu_count)

    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_theta, args=(w, e, batch_of_fc, feature_index, normalizing_decision_map, f_vector, event_index))
    pool.close()
    pool.join()

    endTime = time.time()
    print "run time for computing theta %.2gs" % (endTime - startTime)

    pool = Pool(processes=cpu_count) # uses all available CPUs
    batches_fractional_counts = array_split(range(len(event_index)), cpu_count)


    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_gradient, args=(w, e, batch_of_fc, feature_index, normalizing_decision_map, f_vector, event_index))
    pool.close()
    pool.join()


    endTime = time.time()
    print "run time for get_gradient_with_count %.2gs" % (endTime - startTime)
    #print 'grad',-grad
    return -grad


def forward_with_t_scaled(a,pi,y,N,T,d, t_table): #N is the number of states and d is the target sentence
    # y is the source sentence
    # d is the dest sentence
    c_scaled = ones(T+1)
    alpha_hat = zeros((N+1,T+1))
    total_alpha_double_dot = 0
    for i in range(1,N+1):
        alpha_hat[(i,1)] = pi[i]*t_table[(y[0],d[i-1])]
        total_alpha_double_dot += alpha_hat[(i,1)]
    c_scaled[1] = 1.0/total_alpha_double_dot
    #alpha_hat[:,1] = c_scaled[1]*alpha_hat[:,1]
    for i in range(1,N+1):
        alpha_hat[(i,1)] = c_scaled[1]*alpha_hat[(i,1)]
    for t in range(1,T):
        total_alpha_double_dot = 0
        for j in range(1,N+1):
            total = 0
            for i in range(1,N+1):
                total += alpha_hat[(i,t)]*a[(i,j)]
            alpha_hat[(j,t+1)] = t_table[(y[t],d[j-1])]*total
            total_alpha_double_dot += alpha_hat[(j,t+1)]
        c_scaled[t+1] = 1.0/total_alpha_double_dot
        for i in range(1,N+1):
            alpha_hat[(i,t+1)] = c_scaled[t+1]*alpha_hat[(i,t+1)]
        #alpha_hat[:,t+1] = c_scaled[t+1]*alpha_hat[:,t+1]

    return (alpha_hat,c_scaled)
def backward_with_t_scaled(a,pi,y,N,T,d, t_table,c_scaled):
    beta_hat = zeros((N+1,T+1))
    for i in range(1,N+1):
        beta_hat[(i,T)] = c_scaled[T]
    #beta_hat[:,T] = c_scaled[T]
    for t in range(T-1,0,-1):
        for i in range(1,N+1):
            total = 0
            for j in range(1,N+1):
                total += beta_hat[(j,t+1)]*a[(i,j)]*t_table[(y[t],d[j-1])]
            beta_hat[(i,t)] = c_scaled[t]*total
    return beta_hat
def compute_c(): #Indirect HMM-based hypothesis alignment  by He et al.
    k = 2
    c = defaultdict(int)
    for d in range(-5,8):
        c[d] = 1.0/pow(1+abs(d-1),k)
    return c

def initializeBasedOnC(N):
    a = zeros((N+1,N+1))
    pi = zeros(N+1)

    c = compute_c()

    total = defaultdict(int)
    for i in range(1,N+1):
        pi[i] = 1.0/N
    for l in range(1,N+1):
        for i in range(1,N+1):
            total[i] += c[l-i]
    for i in range(1,N+1):
        for j in range(1,N+1):
            a[(i,j)] = c[j-i]/total[i]
#     for j in range(1,N+1):
#         for y_t in yValues:
#             b[(j,y_t)] = t[(y_t,)]
    return (a,pi)

def maxTargetSentenceLength(bitext):
    maxLength = 0
    for (s,d) in bitext:
        if len(d) > maxLength:
            maxLength = len(d)
    return maxLength

def map_bitext_to_int(sd_count):
    #index = zeros(len(sd_count)) #index vector
    index = defaultdict(int)
    biword = defaultdict(int)
    for (i,(s,d)) in enumerate(sd_count):
        index[(s,d)] = i
        biword[i] = (s,d)
    return (index,biword)























