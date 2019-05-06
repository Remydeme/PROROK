"""
 @Author : DEME Remy
 This file contains the implementation of the soft actor critic policy.
 RL technic for continuous environment

"""



from SoftPolicyGradient.Sac.Networks.DQN import DQN



if __name__ == "__main__":
    mo = DQN(input_size=2,action_size=2)
    print(mo.computeQ([2.0,1.0,0.5, 1.0]))