import numpy as np
from maze import GraphicDisplay, Env

class PolicyIteration:
    # 환경 시작!
    def __init__(self, env):
        self.env = env
        # 가치 함수 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 상 하 좌 우 같은 확률로 정책 초기화
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]

        # 마침 상태의 설정
        self.policy_table[2][2] = []
        # 감가율
        self.discount_factor = 0.9


    # 밸만 기대 방정식을 사용해서 다음 가치함수 계산
    def policy_evaluation(self):
        # 다음 가치함수 초기화
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        # 모든 상태의 밸만 기대 방정식을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 골에 도착하면 가치 함수 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            # 밸만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))
            
            next_value_table[state[0]][state[1]] = value

        self.value_table = next_value_table


    # 현재 가치 함수에 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state ==  [2, 2]:
                continue

            value_list = []
            # 반환활 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)]
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

            # 받을 보상이 최대인 행동들에 대해 탐욕정책 발전
            max_idx_list = np.argwhere(value_list == np.amax(value_list))
            max_idx_list = max_idx_list.flatten().tolist()
            prob = 1 / len(max_idx_list)

            for idx in max_idx_list:
                result[idx] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # 특정 상태에서 무작위 행동 반환
    def get_action(self, state):
        policy = self.get_policy(state)
        policy = np.array(policy)
        return np.random.choice(4, 1, p=policy)[0]

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    # 가치 함수에 값을 반환
    def get_value(self,state):
        return self.value_table[state[0]][state[1]]

if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    maze = GraphicDisplay(policy_iteration)
    maze.mainloop()
            
