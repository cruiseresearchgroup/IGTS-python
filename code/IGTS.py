#!/usr/bin/env python
# coding: utf-8
# author : Shohreh Deldari
# https://github.com/cruiseresearchgroup/IGTS-python
# Matlab Implementation : https://github.com/cruiseresearchgroup/IGTS-matlab
# Reference to paper : 
#### Sadri, Amin, Yongli Ren, and Flora D. Salim. "Information gain-based metric for recognizing transitions in human activities." Pervasive and Mobile Computing 38 (2017): 92-109. 
import numpy as np

# # TOPDOWN IGTS Approach
def TopDown(Integ_TS,k,step,double):
    # pre processing
    Integ_TS = Clean_TS(Integ_TS,double);
    # get size of the input
    if Integ_TS.ndim == 1:
        Num_TS = 1
        Len_TS = Integ_TS.shape[0]
    else:
        Num_TS = Integ_TS.shape[0]
        Len_TS = Integ_TS.shape[1]
    expTT=0
    maxIG = np.zeros(k)
    pos_TT = np.zeros(k+1).astype(int)
    for i in range(k):
        for j in list(range(0,Len_TS,step)):
            pos_TT[i+1] = Len_TS-1
            pos_TT[i] = j
            IG = IG_Cal(Integ_TS,pos_TT,i+1);
            if IG > maxIG[i]:  
                #rint('set expTT i : ',i)
                expTT = pos_TT.copy()
                maxIG[i] = IG
         pos_TT=expTT.copy(); 
    return sorted(expTT[0:k]), maxIG


# Dynamic Programing IGTS
def DP_IGTS(Integ_TS,k,step,double):
    # pre processing
    Integ_TS = Clean_TS(Integ_TS,double);
    # get size of the input
    if Integ_TS.ndim == 1:
        Num_TS = 1
        Len_TS = Integ_TS.shape[0]
    else:
        Num_TS = Integ_TS.shape[0]
        Len_TS = Integ_TS.shape[1]
    #
    cost = np.zeros((Len_TS,Len_TS,k+1)).astype(float)
    TS_dist = np.zeros(Num_TS).astype(float)
    pos = np.zeros((Len_TS, k+1)).astype(int)
    expTT = np.zeros(k).astype(int)
    
    for i in range(0,Len_TS,step):
        for j in range(i+1,Len_TS,step):
            for l in range(0,Num_TS):
                TS_dist[l] = Integ_TS[l,j]-Integ_TS[l,i]
            cost[i:i+step:1 , j:j+step:1 , 0]  =((j-i)/Len_TS) * SH_Entropy(TS_dist)
    
    for b in range(1,k+1):
        for i in range(1,Len_TS):
            cost[0,i,b] = cost[0,i,b-1].copy()
            pos[i,b] = 1
            for j in range(step,i-1,step):
                if cost[0,j,b-1] + cost[j+1,i,0] <= cost[0,i,b]:
                    cost[0,i,b] = cost[0,j,b-1] + cost[j+1,i,0]
                    pos[i,b] = j
             
    maxVAR = cost[0,Len_TS-1,k].copy()
 
    idx=Len_TS-1;
    for b in range(k,0,-1) :
        expTT[b-1]=pos[idx,b].copy();
        idx=expTT[b-1].copy();

    return expTT, maxVAR


# # SHANON ENTROPY Function
def SH_Entropy(x):
#     remove zeors
    x = x[(x != 0)]
    p = np.true_divide(x,np.sum(x))
    return -1 * sum(p * np.log(p))
    


# #  Information Gain Calculation
def IG_Cal(Integ_TS,pos,k):
    Num_TS = Integ_TS.shape[0]
    Len_TS = Integ_TS.shape[1]
    pos = sorted(pos[0:k+1])
    i=0
    TS_dist = np.zeros(Num_TS).astype(float)

    while i < Num_TS:
        TS_dist[i] = Integ_TS[i, Len_TS-1];
        i=i+1
    IG=SH_Entropy(TS_dist)
    last_id=0;
    
    for i in range(k+1):
        for j in range(Num_TS):
            TS_dist[j] = Integ_TS[j,pos[i]] - Integ_TS[j,last_id];
        IG=IG-((pos[i]-last_id)/Len_TS)*SH_Entropy(TS_dist);
        last_id=pos[i];
    return IG


# # CLEAN Time Series
def Clean_TS(O_Integ_TS,double):
    Integ_TS=O_Integ_TS
    if Integ_TS.ndim == 1:
        Num_TS = 1
        Len_TS = Integ_TS.shape[0]
    else:
        Num_TS = Integ_TS.shape[0]
        Len_TS = Integ_TS.shape[1]
    for i in range(Num_TS):
        minVal = min(Integ_TS[i,:])
        if double == 2:
            minVal=0
        Integ_TS[i,:] = Integ_TS[i,:]-minVal;
        if double != 2:
            sumVal=sum(Integ_TS[i,:])/1000;
            Integ_TS[i,:]=Integ_TS[i,:]/sumVal;
        if double == 1:
            maxVal=max(Integ_TS[i,:]);
            to_append = maxVal-Integ_TS[i,:]
            sumVal=sum(to_append)/1000;
            Integ_TS = np.vstack((Integ_TS,np.array(to_append/sumVal)))
    return np.cumsum(Integ_TS,axis=1)
