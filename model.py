import numpy as np
import random

class State:

    def is_terminal(self) -> bool:
        pass

    def get_actions(self) -> list:
        pass

    def get_reward(self) -> int:
        pass

    def transition(self, action):
        pass

    def simulation(self):
        pass


class MarshallingState(State):

    def __init__(self, cells, stacks: int, tiers: int):
        self.stacks: int = stacks
        self.tiers: int = tiers
        self.cells = cells
        self.height = np.zeros(stacks, dtype=int)
        self.move_list = list()
        self.sorted_stacks = list()
        self.sorted_elements = list()
        self.unsorted_stacks = 0
        self.steps = 0
        self.calculate_heights()
        self.calculate_sorted()


    def is_terminal(self) -> bool:
        return self.unsorted_stacks == 0

    def get_actions(self) -> list:
        actions = list()

        for i in range(self.stacks):
            if self.height[i] > 0:
                for j in range(self.stacks):
                    if i != j and self.height[j] < self.tiers:
                        actions.append((i, j))

        return actions

    def get_reward(self) -> int:
        return self.steps

    def transition(self, action):
        m_from = action[0]
        m_to = action[1]

        c = self.cells[m_from][self.height[m_from] - 1]

        if self.is_sorted_stack(m_from):
            self.sorted_elements[m_from] -= 1

        if self.is_sorted_stack(m_to) and self.gvalue(self.cells[m_to], m_to) >= c:
            self.sorted_elements[m_to] += 1

        self.cells[m_to][self.height[m_to]] = c
        self.cells[m_from][self.height[m_from] - 1] = 0

        self.height[m_to] += 1
        self.height[m_from] -= 1

        self.is_sorted_stack(m_from)
        self.is_sorted_stack(m_to)

        self.steps += 1

        self.move_list.append(action)

    def calculate_heights(self):
        for stack in range(self.stacks):
            for tier in range(self.tiers):
                if self.cells[stack][tier] == 0:
                    self.height[stack] = tier
                    break

                if tier == self.tiers - 1:
                    self.height[stack] = self.tiers

    def compute_sorted_elements(self, stack, i):
        if np.all(stack == 0):
            return 0

        sorted_elements = 1

        while(sorted_elements < self.height[i] and stack[sorted_elements] <= stack[sorted_elements - 1]):
            sorted_elements +=1

        return sorted_elements

    def calculate_sorted(self):
        j = 0
        for i in range(self.stacks):
            self.sorted_elements.append(self.compute_sorted_elements(self.cells[i], i))
            if not self.is_sorted_stack(j):
                self.unsorted_stacks += 1
                self.sorted_stacks.append(False)
            else:
                self.sorted_stacks.append(True)
                j += 1

    def is_sorted_stack(self, j):
        sorted = self.height[j] == self.sorted_elements[j]

        if j < len(self.sorted_stacks) and self.sorted_stacks[j] != sorted:
            self.sorted_stacks[j] = sorted

            if sorted == True:
                self.unsorted_stacks -= 1
            else:
                self.unsorted_stacks += 1

        return sorted


    def gvalue(self, cell, j):
        if self.height[j] == 0:
            return 100
        else:
            return cell[self.height[j] - 1]

    def select_destination_stack(self, orig, black_list=[], max_pos=100):
        s_o = self.cells[orig]
        c = s_o[self.height[orig] - 1]
        best_eval=-1000000
        best_dest=None
        dest=-1

        for dest in range(self.stacks):
            if orig == dest or dest in black_list:
                continue

            s_d = self.cells[dest]

            if self.tiers == self.height[dest]:
                 continue

            top_d = self.gvalue(s_d, dest)

            ev = 0

            if self.is_sorted_stack(dest) and c <= top_d:
              #c can be well-placed: the sorted stack minimizing top_d is preferred.
              ev = 10000 - 100*top_d
            elif not self.is_sorted_stack(dest) and c >= top_d:
              #unsorted stack with c>=top_d maximizing top_d is preferred
              ev = top_d
            elif self.is_sorted_stack(dest):
              #sorted with minimal top_d
              ev = -100 - top_d
            else:
              #unsorted with minimal numer of auxiliary stacks
              ev = -10000 #+ required_stacks(dest)

            if self.tiers - self.height[dest] > max_pos:
              ev -= 10000

            if ev > best_eval:
                best_eval=ev
                best_dest=dest

        return best_dest

    def select_origin_stack(self, dest, ori, rank):
        s_d = self.cells[dest]
        top_d = s_d[self.height[dest] - 1]
        best_eval = -1000000
        best_orig = None
        orig = -1

        for orig in range(self.stacks):
            if orig == dest or orig == ori:
                 continue

            s_o = self.cells[orig]

            if np.all(s_o == 0):
                continue

            c = self.gvalue(s_o, orig)

            if c in rank and rank[c] < self.tiers - self.height[dest]:
                continue

            ev=0

            if self.is_sorted_stack(dest) and c <=top_d:
                #c can be well-placed: the sorted stack maximizing c is preferred.
                ev = 10000 + 100*c
            elif not self.is_sorted_stack(dest) and c >= top_d:
                #unsorted stack with c>=top_d minimizing c is preferred
                ev = -c
            else:
                ev = -100 - c

            if ev > best_eval:
                best_eval=ev
                best_orig=orig

        return best_orig

    def reachable_height(self, i):
        if not self.is_sorted_stack(i):
            return -1

        top = self.gvalue(self.cells[i], i)
        h = self.height[i]

        if h == self.tiers:
            return self.tiers

        all_stacks = True #True: all the bad located tops can be placed in stack

        for k in range(self.stacks):
            if k == i:
                continue

            if self.is_sorted_stack(k):
                continue

            stack_k = self.cells[k]
            unsorted = self.height[k] - self.sorted_elements[k]
            prev = 1000

            for j in range (1, unsorted + 1):

                if stack_k[-j] <= prev and stack_k[-j] <=top:
                    h += 1

                    if h == self.tiers:
                        return h

                    prev = stack_k[-j]
                else:
                    if j == 1:
                        all_stacks=False
                    break

        if all_stacks:
            return self.tiers
        else:
            return h

    def SF_move(self):
        actions = []

        for i in range(self.stacks):
            if self.is_sorted_stack(i) and self.height[i] < self.tiers:
                top = self.gvalue(self.cells[i], i)
                for k in range(self.stacks):
                    if k!=i and not self.is_sorted_stack(k):
                        if self.cells[k][self.height[k] - 1] <= top :
                            actions.append((k, i))

        #actions.sort()

        pos = np.random.randint(len(actions) + 10, size=1)[0]

        if len(actions) > pos:
            action = actions[pos]
            self.transition(action)
            return True

        return False


    def SD_move(self):
        best_ev = 0
        actions = []
        for i in range(self.stacks):
            prom = self.cells[i].mean()
            ev = 10000 - 100*self.height[i] - prom

            actions.append( (-ev, i))
            if ev > best_ev:
                best_ev = ev
                s_o = i

        #actions.sort()

        pos = np.random.randint(len(actions) + 10, size=1)[0]

        if len(actions) <= pos:
             return False

        while self.height[s_o] > 0:
            ev, s_o = actions[pos]
            s_d = self.select_destination_stack(s_o)
            self.transition((s_o,s_d))
            if self.reachable_height(s_o) == self.tiers:
                 return True

        return True

    def simulation(self):
        while not self.is_terminal():
            if not self.SF_move():
                self.SD_move()

            if self.steps > 1000:
                return
