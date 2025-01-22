from gymnasium.spaces import Box

# for my trading environment
low = np.array([-1, 1, 1, 1])
high = np.array([1, 10, 500, 500])
action_space = Box(low, high, shape=(4,), dtype=np.int32)
action_space.sample()

# for my trading environment
low = np.array([[-1, 1],[1, 1]])
high = np.array([[1, 10],[500, 500]])
observation_space = Box(low, high, shape=(2, 2), dtype=np.int32)
observation_space.sample()

# array([[  1,   3],
#        [215, 255]])
# element (0, 0) is between -1 and 1. -1:sell 0:do nothing 1:buy
# element (0, 1) is between 1 and 10. lot size
# element (1, 0) is between 1 and 500. stop loss (in percentage)
# element (1, 1) is between 1 and 500. take profit (in percentage)

# other examples

observation_space = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
observation_space.sample()

# array([[-0.4158095 ,  1.385939  , -0.14797208,  1.7796195 ],
#        [ 0.20189   , -0.7850134 , -0.01907982,  1.0457855 ],
#        [ 0.8359633 ,  0.5205814 , -0.38575202,  0.27911523]],
#       dtype=float32)

observation_space = Box(np.array([-1, -2]), np.array([2, 4]), shape=(2,), dtype=np.float32)
observation_space.sample()

# array([0.14957984, 3.431901  ], dtype=float32)


from gymnasium.spaces import Discrete
observation_space = Discrete(2, seed=42) # {0, 1}
observation_space.sample()
observation_space = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}
observation_space.sample()


from gymnasium.spaces import MultiBinary
observation_space = MultiBinary(5, seed=42) # 5 bits of binary values (0 or 1)
observation_space.sample()

# array([1, 0, 1, 0, 1], dtype=int8)

observation_space = MultiBinary([3, 2], seed=42)
observation_space.sample() # 3x2 bits of binary values (0 or 1)

# array([[1, 0],
#        [1, 0],
#        [1, 1]], dtype=int8)


from gymnasium.spaces import MultiDiscrete
import numpy as np
observation_space = MultiDiscrete(np.array([[1, 2], [3, 4]]), seed=42)
# 2x2, elements indicate the number of discrete values
#e.g 1 means 0; 2 means 0, 1; 3 means 0, 1, 2; 4 means 0, 1, 2, 3
observation_space.sample()

# array([[0, 0],
#        [2, 2]], dtype=int64)

