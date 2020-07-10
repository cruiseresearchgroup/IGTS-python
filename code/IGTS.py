"""

Authors: Shohreh Deldari and Sam Nolan
GitHub: https://github.com/cruiseresearchgroup/IGTS-python
Matlab Implementation: https://github.com/cruiseresearchgroup/IGTS-matlab
Reference to paper : Sadri, Amin, Yongli Ren, and Flora D. Salim. "Information gain-based metric for recognizing transitions in human activities." Pervasive and Mobile Computing 38 (2017): 92-109.

Information Gain Temporal Segmntations (IGTS) is a method for segmenting
multivariate time series based off reducing the entopy in each segment.

The amount of entropy lost by the segmentations made is called the Information
Gain (IG). The aim is to find the segmentations that have the maximum information
gain for any number of segmentations.

Definitions:
 - A channel is one of the variables in a multivariate time series

Things to note:
 - The splits returned in the time series are taken to be after the kth element, not before


"""
import numpy as np


def TopDown(Multivar_TS, k, step, double=1):
    """
    Top down IGTS. This method of IGTS tries to greedily find the next segment
    location that creates the maximum information gain. Once this is found, it
    repeats the process until we have k splits in the time series.

    :param Multivar_TS: a numpy array of dimensions (amount of channels, time).
    :param k: The amount of splits to find in the time series. Which makes the
              amount of segments equal to k + 1(int)
    :param step: The size of the steps to make when searching through the time
                 series to find the heighest value. For instance, a step of 5
                 will find the max IG out of 0...5...10 etc
    :param double: paramater passed to Clean_TS for the purpose of removing
                   correlation from the time series. Leaving it at 1 is reccomended

    :return: (splits,InformationGain,knee)
             splits is a numpy array of integers of size <= k
             The amount of splits (that I will call n) can be smaller than or
             equal to k. It is smaller than k when creating any more segments
             does not increase information gain. This usually (by experience)
             does not occur when the amount of splits is larger than 50% of
             the time series.

             splits represents the positions of splits that are found to be optimal in
             the time series. These splits are after the position they index,
             for instance, if there is a 2 in the array, then {0,1,2} is one
             segment and {3,4,5...} is another. The order of this array is
             important, and is not sorted. The first element is the split that
             was found to have the highest information gain, and the second has
             the second heighest, etc.

             Information Gain is an numpy array of floats of size n. The ith element of the
             arary represents the information gain found for the first i segments.

             knee is a number <= n that is the knee point of the time series.
             Choosing a balance between
             number of segments (usually creating a larger amount of information
             gain) vs the size of the segments (too many segments can make them
             very small). The knee point of the information gain vs segments
             curve is returned to knee.
    """

    # Pre Process the time series
    Integ_TS = Clean_TS(Multivar_TS, double)

    # get size of the input
    if Integ_TS.ndim == 1:
        Len_TS = Integ_TS.shape[0]
    else:
        Len_TS = Integ_TS.shape[1]

    # maxTT is the segments found for the maximum IG found so far
    maxTT = np.zeros(k + 1).astype(int)

    # tryTT is the working segments, that we will be trying
    tryTT = np.zeros(k + 1).astype(int)

    # IG_arr is the information gain found for k
    IG_arr = np.zeros(k + 1)
    IG_arr[0] = 0

    # p values are the second derivative of the curve at all points. Used to
    # determine the mean
    p_arr = np.zeros(k + 1)
    maxIG = 0

    # Segments k times
    for i in range(k):

        # Try for a segment in j
        for j in list(range(0, Len_TS, step)):

            # Add a new segment at point j
            tryTT[i+1] = Len_TS-1
            tryTT[i] = j

            # Does an incremental IG calculation. The incremental version of
            # this function performs much better for larger k
            IG = IG_Cal_Incremental(IG_arr[i], Integ_TS, tryTT[0:i], tryTT[i])
            if IG > maxIG:
                # Record
                maxTT = tryTT.copy()
                maxIG = IG

        # If we did not make any progress from the information gain we already had
        if maxIG == IG_arr[i]: 
            # We didn't get any information gain from this, so we should not continue
            break

        tryTT=maxTT.copy()

        IG_arr[i + 1] = maxIG
        
        # If it's possible to calculate the curvature, do so and add it to p
        if i >= 1:
            p_arr[i] = (IG_arr[i] - IG_arr[i-1]) / (IG_arr[i + 1] - IG_arr[i])
    knee = np.argmax(p_arr)
    return tryTT,IG_arr,knee


# Dynamic Programing IGTS
def DP_IGTS(Multivar_TS,k,step,double):
    """
    Dynamic Programming IGTS. This method of IGTS tries find the segment boundary
    locations that create the maximum information gain using Dynamic Programming.

    :param Multivar_TS: a numpy array of dimensions (amount of channels, time).
    :param k: The amount of splits to find in the time series. Which makes the
              amount of segments equal to k + 1(int)
    :param step: The size of the steps to make when searching through the time
                 series to find the heighest value. For instance, a step of 5
                 will find the max IG out of 0...5...10 etc
    :param double: paramater passed to Clean_TS for the purpose of removing
                   correlation from the time series. Leaving it at 1 is reccomended

    :return: (expTT,InformationGain)
             expTT is a numpy array of integers of size = k
             
             expTT represents the positions of splits that are found to be optimal in
             the time series. These splits are after the position they index,
             for instance, if there is a 2 in the array, then {0,1,2} is one
             segment and {3,4,5...} is another. The order of this array is
             important, and is not sorted. The first element is the split that
             was found to have the highest information gain, and the second has
             the second heighest, etc.

             Information Gain is floats represents the information gain of the whole time series
             regards to estimated segment boundaries.
             
    """ 
    # pre processing
    Integ_TS = Clean_TS(Multivar_TS,double)
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
 
    idx=Len_TS-1
    for b in range(k,0,-1):
        expTT[b-1]=pos[idx,b].copy()
        idx=expTT[b-1].copy()

    return expTT, maxVAR


# # SHANON ENTROPY Function
def SH_Entropy(x):
    x = x[(x != 0)]
    p = np.true_divide(x,np.sum(x))
    return -1 * sum(p * np.log(p))
    
def IG_Cal_Incremental(IG_old,Integ_TS,pos_old,new_pos):
  """
   Calculates the information gain incrementally, based off the information
   gain previously recorded.

   This function works by finding the segment for which the new_pos splits
   in two. It then subtracts the information gain for the whole segment and
   adds the IG for the split segments. This operation has complexity O(

   :param IG_old: The information Gain of Integ_TS with positions pos_old
   :param Integ_TS: The integral (cumulative sum) of the time series to 
                    calculate the information gain on. 
                    Representes as a numpy array of shape (number of series, time)
   :param pos_old: The positions on Integ_TS that have IG_old IG
                   Represented as a numpy aray of integer positions
   :param new_pos: The new position to add to the Integ_TS, integer

   :returns: float representing the information gain of Integ_TS over pos_old 
             and new_pos splits
  """
  Num_TS = Integ_TS.shape[0]
  Len_TS = Integ_TS.shape[1]

  # Working array to keep the new values in
  TS_dist = np.zeros(Num_TS).astype(float)
   
  # The positions of the segment boundaries higher and lower than the new_pos
  lower = max([-1]+ [x for x in pos_old if x <= new_pos])
  higher = min([Len_TS-1] + [x for x in pos_old if x >= new_pos])

  # Calculate the entropy of the whole old segment
  for j in range(Num_TS):
      # This operation here is meant to get the sum of all the elements between
      # lower and higher, not inclusive of higher. If lower is the very 
      # beginning of the time series (represented as lower = -1), then the 
      # cumulative sum at higher is equal to the sum between the start and higher
      if lower == -1:
          TS_dist[j] = Integ_TS[j,higher]
      else:
          TS_dist[j] = Integ_TS[j,higher] - Integ_TS[j,lower]
  old_entroy = SH_Entropy(TS_dist)

  # Use the same method to calculate the entropy of the left segment and then
  # the right
  for j in range(Num_TS):
      if lower == -1:
          TS_dist[j] = Integ_TS[j,new_pos]
      else:
          TS_dist[j] = Integ_TS[j,new_pos] - Integ_TS[j,lower]
  new_entropy_left = SH_Entropy(TS_dist)

  for j in range(Num_TS):
      TS_dist[j] = Integ_TS[j,higher] - Integ_TS[j,new_pos]

  new_entropy_right = SH_Entropy(TS_dist)

  # Then we calculate the change in weighted entropy
  weighted_right = (higher - new_pos) * new_entropy_right / Len_TS
  weighted_old = (higher - lower) * old_entroy / Len_TS
  weighted_left = (new_pos - lower) * new_entropy_left / Len_TS

  entropy_change = weighted_old - weighted_left - weighted_right

  return IG_old + entropy_change


# #  Information Gain Calculation
def IG_Cal(Integ_TS,pos,k):
    Num_TS = Integ_TS.shape[0]
    Len_TS = Integ_TS.shape[1]
    pos = sorted(pos[0:k+1])
    i=0
    TS_dist = np.zeros(Num_TS).astype(float)

    while i < Num_TS:
        TS_dist[i] = Integ_TS[i, Len_TS-1]
        i=i+1
    IG=SH_Entropy(TS_dist)
    last_id=0
    
    for i in range(k+1):
        for j in range(Num_TS):
            TS_dist[j] = Integ_TS[j,pos[i]] - Integ_TS[j,last_id]
        IG=IG-((pos[i]-last_id)/Len_TS)*SH_Entropy(TS_dist)
        last_id=pos[i]
    return IG


# # CLEAN Time Series
def Clean_TS(O_Integ_TS,double):
    """
      Clean_TS does three different types of cleaning based off the values
      passed from double

      double == 0, normalise all values in the series to be between 0 and 1000 (where 1000 is the max in the channel and 0 is the min)
      double == 1, Appends the reverse of the time series to remove positive correlation
      double == 2 normalises so that 0 is still 0 but 1000 is the max of the time
      series

      We reccomend to use 1 as double under most circumstances
    """
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
        Integ_TS[i,:] = Integ_TS[i,:]-minVal
        if double != 2:
            sumVal=sum(Integ_TS[i,:])/1000
            print(sumVal)
            Integ_TS[i,:]=Integ_TS[i,:]/sumVal
        if double == 1:
            maxVal=max(Integ_TS[i,:])
            to_append = maxVal-Integ_TS[i,:]
            sumVal=sum(to_append)/1000
            Integ_TS = np.vstack((Integ_TS,np.array(to_append/sumVal)))
    return np.cumsum(Integ_TS,axis=1)
