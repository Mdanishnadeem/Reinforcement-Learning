# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        print('here', self.iterations)

        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
          temp = util.Counter()
          states = self.mdp.getStates()
          for state in states:
            ##we set value for terminal state equal to zero and get actions and rewards for each state
            max_val = -99999
            temp[state] = 0.0
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              qval = self.computeQValueFromValues(state, action)
              temp[state] = max(max_val, qval)
              max_val = max(max_val, qval)
          self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            next_state = transition[0]
            prob_next_state = transition[1]
            next_qVal = self.values[next_state]
            qVal = qVal + prob_next_state * (self.mdp.getReward(state, action, next_state) + self.discount * next_qVal) 
            #qVal += transition[1] * (self.mdp.getReward(state, action, transition[1]) \
            #+ self.discount * self.values[transition[0]])
        return qVal

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        #find all actions and the corresponding value and then return action
        #corresponding to the maximum value
        allActions = {}
        for action in actions:
            allActions[action] = self.computeQValueFromValues(state, action)

        return max(allActions, key=allActions.get)
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        length = len(states)

        for iteration in range(self.iterations):
            ind = iteration % length
            state  = states[ind]
            if self.mdp.isTerminal(state) == False:
                actions = self.mdp.getPossibleActions(state)
                maxVal = max([self.getQValue(state,action) for action in actions])
                self.values[state] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        priorityqueue = util.PriorityQueue()

        predecessors = {}

        #fid all predecessors
        states = self.mdp.getStates()
        for state in states:
            if  self.mdp.isTerminal(state) == False:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                        if transition[0] in predecessors:
                            predecessors[transition[0]].add(state)
                        else:
                            predecessors[transition[0]] = {state}


        for state in states:
            if  self.mdp.isTerminal(state) == False:
                difference = abs(self.values[state] - max([ \
                self.computeQValueFromValues(state, action) for action in \
                self.mdp.getPossibleActions(state) ]) )
                #pushing negative of difference into the queue
                priorityqueue.push(state, -difference)

        for iteration in range(self.iterations):
            if priorityqueue.isEmpty():
                return
            current_state = priorityqueue.pop()
            if not self.mdp.isTerminal(current_state):
                self.values[current_state] = max([self.computeQValueFromValues(current_state, action)\
                 for action in self.mdp.getPossibleActions(current_state)])

            for predecessor in predecessors[current_state]:
                if not self.mdp.isTerminal(predecessor):
                    diff = abs(self.values[predecessor] - max([self.computeQValueFromValues(predecessor, action)\
                     for action in self.mdp.getPossibleActions(predecessor)]))

                    if diff > self.theta:
                            priorityqueue.update(predecessor, -diff)

