from model import MarshallingState
import mcts
import numpy as np

def read_file(file, h):
    with open(file) as f:
        _, _ = [int(x) for x in next(f).split()] # read first line
        stacks = []
        for line in f: # read rest of lines
            stack = [int(x) for x in line.split()[1::]]
            #if stack[0] == 0: stack.pop()
            stacks.append(stack)

        S = len(stacks)
        cells = np.zeros((S, h), dtype=int)

        for stack in range(S):
            for tier in range(len(stacks[stack])):
                cells[stack][tier] = stacks[stack][tier]

    return (cells, S)


H = 5
cells, stacks = read_file("instancias\\BF\\BF1\\cpmp_16_5_48_10_29_1.bay", H)

state = MarshallingState(cells, stacks, H)

agent = mcts.MCTS()
agent.search(state)
print(agent.best_state.get_reward())
print(agent.best_state.cells)
