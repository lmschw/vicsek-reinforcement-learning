
import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from pettingzoo import ParallelEnv

from enum_neighbour_selection_mechanism import NeighbourSelectionMechanism

class VicsekEnvironment(ParallelEnv):
    metadata = {
        "name": "vicsek-environment_v0"
    }

    def __init__(self, render_mode=None, domain_size=(25, 2550), radius=50, speed=1, noise=0.1, base_number_agents=5, number_foodsources=1, start="ordered"):
        self.domain_size = np.array(domain_size)
        self.radius = radius
        self.speed = speed
        self.noise = noise
        self.nsm = None
        self.foodsources = None
        self.number_foodsources = number_foodsources
        self.agent_definitions = None
        self.positions = None
        self.orientations = None
        self.start = start
        self.timestep = None
        self.base_number_agents = base_number_agents
        self.number_agents = None
        self.minReplacementValue = None
        self.maxReplacementValue = None
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.minReplacementValue = -5
        self.maxReplacementValue = -4
        #self.nsms = np.full(self.base_number_agents, NeighbourSelectionMechanism.NEAREST)
        self.neighbour_selection_mechanism = NeighbourSelectionMechanism.NEAREST
        agents = list(range(self.base_number_agents))
        agents.append("P")
        self.agents = agents
        self.number_agents = self.base_number_agents + 1
        self.agent_definitions = [(0.1, 100, 1, -1, 1, 5) for agent in self.agents]
        self.positions = self.domain_size*np.random.rand(self.number_agents,len(self.domain_size))

        #self.positions = np.array([[random.random() * self.domain_size[0], random.random() * self.domain_size[1]] for agent in self.agents])
        if self.start == "ordered":
            orientation = [(random.random() * 2) -1, (random.random() * 2) -1]
            self.orientations = np.array([orientation for agent in self.agents])
        else:
            self.orientations = np.array([[random.random() * self.domain_size[0], random.random() * self.domain_size[1]] for agent in self.agents])
        
        self.foodsources = np.array([[random.random() * self.domain_size[0], random.random() * self.domain_size[1]] for foodsource in range(self.number_foodsources)])
        self.neighbours = self.getNeighbours(self.positions, self.domain_size, self.radius)

        self.localOrders = self.computeLocalOrders(orientations=self.orientations, neighbours=self.neighbours)
        observations = {
            a: (
                self.localOrders[a]
            )
            for a in self.agents[:-1]
        }
        observations['P'] = self.localOrders[-1]

        infos = {a:{} for a in self.agents}

        return observations, infos

    def step(self, actions):
        self.neighbours = self.getNeighbours(self.positions, self.domain_size, self.radius)
        
        #new_positions = copy(self.positions) 
        #new_orientations = copy(self.orientations)
        self.orientations = self.computeNewOrientations(self.neighbours, self.positions, self.orientations, None, ks=actions)

        self.positions += (self.orientations * self.speed)
        self.positions += -self.domain_size*np.floor(self.positions/self.domain_size)

        # Termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        for agent in self.agents:
            if agent == 'P':
                continue
            if self.positions[agent][0] == self.positions[-1][0] and self.positions[agent][0] == self.positions[-1][0]:
                rewards[agent] = -1
                rewards[0] = 1
                terminations[agent] = True
            
        
        # Truncation conditions
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}

        self.timestep += 1

        # get observations 
        self.localOrders = self.computeLocalOrders(orientations=self.orientations, neighbours=self.neighbours)
        observations = {
            a: (
                self.localOrders[a]
            )
            for a in self.agents[:-1]
        }
        observations['P'] = self.localOrders[-1]

        infos = {a:{} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.full(self.domain_size, " ")
        for agent in self.agents:
            grid[int(self.positions[agent][1]), int(self.positions[agent][0])] = agent
        for foodsource in self.foodsources:
            grid[int(foodsource[1]), int(foodsource[0])] = "F"
        print(f"{grid} \n")


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # the position and orientation of every agent (including predator)
        maxDim = max(self.domain_size[0], self.domain_size[1])
        #return Box(low=np.array([0, -1, False, 0]), high=np.array([maxDim, 1, True, maxDim]), shape=(3, self.number_agents, 2))
        #return Dict({"localOrder": Box(low=0, high=1, shape=(1,))})
        return Box(low=0, high=1, shape=(1,))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #return MultiDiscrete([6]*2)
        return Discrete(5)

    def computeLocalOrders(self, orientations, neighbours):
        """
        Computes the local order for every individual.

        Params: 
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

        Returns:
            An array of floats representing the local order for every individual at the current time step (values between 0 and 1)
        """
        sumOrientation = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        return np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), np.count_nonzero(neighbours, axis=1))

    def getDifferences(self, array, domain_size):
        """
        Computes the differences between all individuals for the values provided by the array.

        Params:
            - array (array of floats): the values to be compared

        Returns:
            An array of arrays of floats containing the difference between each pair of values.
        """
        rij=array[:,np.newaxis,:]-array   
        rij = rij - domain_size*np.rint(rij/domain_size) #minimum image convention
        return np.sum(rij**2,axis=2)

    def getNeighbours(self, positions, domain_size, radius):
        """
        Determines all the neighbours for each individual.

        Params:
            - positions (array of floats): the position of every individual at the current timestep

        Returns:
            An array of arrays of booleans representing whether or not any two individuals are neighbours
        """
        rij2 = self.getDifferences(positions, domain_size)
        return (rij2 <= radius**2)
    
    def computeNewOrientations(self, neighbours, positions, orientations, nsms, ks):
        """
        Computes the new orientation of every individual based on the neighbour selection mechanisms, ks, time delays and Vicsek-like 
        averaging.
        Also sets the colours for ColourType.EXAMPLE.

        Params:
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - nsms (array of NeighbourSelectionMechanism): the neighbour selection mechanism used by every individual at the current timestep
            - ks (array of ints): the number of neighbours k used by every individual at the current timestep
            - activationTimeDelays (array of ints): at what rate updates are possible for every individual at the current timestep

        Returns:
            An array of floats representing the orientations of all individuals after the current timestep
        """
        ks = np.array(list(ks.values()))
        pickedNeighbours = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=self.neighbour_selection_mechanism,
                                                                                      positions=positions, 
                                                                                      orientations=orientations, 
                                                                                      neighbours=neighbours,
                                                                                      ks=ks)

        np.fill_diagonal(pickedNeighbours, True)

        oldOrientations = np.copy(orientations)

        orientations = self.calculateMeanOrientations(orientations, pickedNeighbours)
        orientations = self.normalizeOrientations(orientations+self.generateNoise())

        return orientations
    
    def generateNoise(self):
        """
        Generates some noise based on the noise amplitude set at creation.

        Params:
            None

        Returns:
            An array with the noise to be added to each individual
        """
        return np.random.normal(scale=self.noise, size=(self.number_agents, len(self.domain_size)))

     
    def calculateMeanOrientations(self, orientations, neighbours):
        """
        Computes the average of the orientations of all selected neighbours for every individual.

        Params:
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

        Returns:
            An array of floats containing the new, normalised orientations of every individual
        """
        summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        return self.normalizeOrientations(summedOrientations)

    def normalizeOrientations(self, orientations):
        """
        Normalises the orientations of all particles for the current time step

        Parameters:
            - orientations (array): The current orientations of all particles

        Returns:
            The normalised orientations of all particles as an array.
        """
        return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])

    def __checkPickedForNeighbourhood(self, posDiff, candidates, kMaxPresent):
        """
        Verifies that all the selected neighbours are within the perception radius.

        Params:
            - posDiff (array of arrays of float): the position difference between every pair of individuals
            - candidates (array of int): the indices of the selected neighbours
            - kMaxPresent (int): waht is the highest value of k present in the current values of k

        Returns:
            An array of int indices of the selected neighbours that are actually within the neighbourhood.
        """
        if len(candidates) == 0 or len(candidates[0]) == 0:
            return candidates
        # exclude any individuals that are not neighbours
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        minusOnes = np.full((self.number_agents,kMaxPresent), -1)
        picked = np.where(((candidates == -1) | (pickedDistances > self.radius**2)), minusOnes, candidates)
        return picked
    
    def __createBooleanMaskFromPickedNeighbourIndices(self, picked, kMax):
        """
        Creates a boolean mask from the indices of the selected neighbours.

        Params:
            - picked (array of array of int): the selected indices for each individual

        Returns:
            An array of arrays of booleans representing which neighbours have been selected by each individual.
        """
        if len(picked) == 0 or len(picked[0]) == 0:
            return np.full((self.number_agents, self.number_agents), False)
        # create the boolean mask
        ns = np.full((self.number_agents,self.number_agents+1), False) # add extra dimension to catch indices that are not applicable
        pickedValues = np.full((self.number_agents, kMax), True)
        np.put_along_axis(ns, picked, pickedValues, axis=1)
        ns = ns[:, :-1] # remove extra dimension to catch indices that are not applicable
        return ns
    
    def __getPickedNeighbours(self, posDiff, candidates, ks, isMin):
        """
        Determines which neighbours the individuals should consider.

        Params:
            - posDiff (array of arrays of floats): the distance from every individual to all other individuals
            - candidates (array of arrays of floats): represents either the position distance between each pair of individuals or a fillValue if they are not neighbours  
            - ks (array of ints): which value of k every individual observes
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        
        kMax = np.max(ks)

        sortedIndices = candidates.argsort(axis=1)
        if isMin == False:
            sortedIndices = np.flip(sortedIndices, axis=1)
        
        picked = self.__getPickedNeighbourIndices(sortedIndices=sortedIndices, kMaxPresent=kMax, ks=ks)
        picked = self.__checkPickedForNeighbourhood(posDiff=posDiff, candidates=picked, kMaxPresent=kMax)
        mask = self.__createBooleanMaskFromPickedNeighbourIndices(picked, kMax)
        return mask        

    def __getPickedNeighbourIndices(self, sortedIndices, kMaxPresent, ks):
        """
        Chooses the indices of the neighbours that will be considered for updates.

        Params:
            - sortedIndices (arrays of ints): the sorted indices of all neighbours
            - kMaxPresent (int): what is the highest value of k present in the current values of k
            - ks (array of int): the current values of k for every individual

        Returns:
            Array containing the selected indices for each individual.
        """
        #kValues = np.array(list(ks.values()))
        kValues = ks
        kMin = np.min(kValues)
        kMax = np.max(kValues)

        uniqueKs = np.unique(kValues)
        candidates = np.full((self.number_agents, kMax), -1)
        
        for k in uniqueKs:
            candidatesK = sortedIndices[:, :k]
            if k < kMax:
                candidatesK = self.padArray(a=candidatesK, n=self.number_agents, kMin=k, kMax=kMax)

            candidates = np.where(((kValues == k)[:, None]), candidatesK, candidates)

        return candidates
           
    def pickPositionNeighbours(self, positions, neighbours, ks, isMin=True):
        """
        Determines which neighbours the individuals should considered based on the neighbour selection mechanism and k with regard to position.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every 
            - ks (array of ints): which value of k every individual observes
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        posDiff = self.getDifferences(self.positions, self.domain_size)
        if isMin == True:
            fillValue = self.maxReplacementValue
        else:
            fillValue = self.minReplacementValue

        fillVals = np.full((self.number_agents,self.number_agents), fillValue)
        candidates = np.where((neighbours), posDiff, fillVals)

        # select the best candidates
        return self.__getPickedNeighbours(posDiff=posDiff, candidates=candidates, ks=ks, isMin=isMin)
    
    def pickOrientationNeighbours(self, positions, orientations, neighbours, ks, isMin=True):
        """
        Determines which neighbours the individuals should consider based on the neighbour selection mechanism and k with regard to orientation.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes
            - isMin (boolean) [optional, default=True]: whether to take the least orientionally different or most orientationally different neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        posDiff = self.getDifferences(positions, self.domain_size)
        orientDiff = self.getDifferences(orientations, self.domain_size)

        if isMin == True:
            fillValue = self.maxReplacementValue
        else:
            fillValue = self.minReplacementValue

        fillVals = np.full((self.number_agents,self.number_agents), fillValue)
        candidates = np.where((neighbours), orientDiff, fillVals)

        # select the best candidates
        return self.__getPickedNeighbours(posDiff=posDiff, candidates=candidates, ks=ks, isMin=isMin)
    
    def pickRandomNeighbours(self, positions, neighbours, ks):
        """
        Determines which neighbours the individuals should consider based on random selection.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        np.fill_diagonal(neighbours, False)
        posDiff = self.getPositionDifferences(positions, self.domain_size)
        kMax = np.max(ks)
        
        candidateIndices = self.getIndicesForTrueValues(neighbours, paddingType='repetition')
        rng = np.random.default_rng()
        rng.shuffle(candidateIndices, axis=1)
        
        picked = self.__getPickedNeighbourIndices(sortedIndices=candidateIndices, kMaxPresent=kMax, ks=ks)
        picked = self.__checkPickedForNeighbourhood(posDiff=posDiff, candidates=picked, kMaxPresent=kMax)
        selection = self.__createBooleanMaskFromPickedNeighbourIndices(picked, kMax)
        np.fill_diagonal(selection, True)
        return selection

    def getPickedNeighboursForNeighbourSelectionMechanism(self, neighbourSelectionMechanism, positions, orientations, neighbours, ks):
        """
        Determines which neighbours should be considered by each individual.

        Params:
            - neighbourSelectionMechanism (NeighbourSelectionMechanism): how the neighbours should be selected
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes
        """
        match neighbourSelectionMechanism:
            case NeighbourSelectionMechanism.NEAREST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, ks, isMin=True)
            case NeighbourSelectionMechanism.FARTHEST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, ks, isMin=False)
            case NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pickOrientationNeighbours(positions, orientations, neighbours, ks, isMin=True)
            case NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pickOrientationNeighbours(positions, orientations, neighbours, ks, isMin=False)
            case NeighbourSelectionMechanism.RANDOM:
                pickedNeighbours = self.pickRandomNeighbours(positions, neighbours, ks)
            case NeighbourSelectionMechanism.ALL:
                pickedNeighbours = neighbours
        return pickedNeighbours

    def padArray(self, a, n, kMin, kMax, paddingValue=-1):
        if kMax > len(a[0]):
            minusDiff = np.full((n,kMax-kMin), paddingValue)
            return np.concatenate((a, minusDiff), axis=1)
        return a