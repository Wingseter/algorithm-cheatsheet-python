import tkinter as tk
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image
# from os import *

PhotoImage = ImageTk.PhotoImage
UNIT = 100 # 유닛 크기
HEIGHT = 5
WIDTH = 5
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
REWARDS = []

class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')                                                 # 타이틀 
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT + 50))             # 창 크기
        self.texts = []                                                                # 
        self.arrows = []                                                               # 
        self.env = Env()                                                               # 
        self.agent = agent                                                             # 
        self.evaluation_count = 0                                                      # 
        self.improvement_count = 0                                                     # 
        self.is_moving = 0                                                             # 
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()  # 
        self.canvas = self._build_canvas()                                             # 
        self.text_reward(2, 2, "R : 1.0")                                              # 
        self.text_reward(1, 2, "R : -1.0")                                             # 
        self.text_reward(2, 1, "R : -1.0")                                             # 

    # 캔버스 만들기
    def _build_canvas(self):
        # 버튼 초기화
        canvas = tk.Canvas(self, bg = 'white', height = HEIGHT * UNIT, width = WIDTH * UNIT)

        # Evaluate 버튼
        iteration_button = Button(self, text = "Evaluate", command = self.evaluate_policy)
        iteration_button.configure(width=10, activebackground = "#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.13, HEIGHT * UNIT + 10, window=iteration_button)
        
        # Improve 버튼
        policy_button = Button(self, text = "Improve", command = self.improve_policy)
        policy_button.configure(width = 10, activebackground = "#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 10, window =policy_button)

        # move 버튼
        move_button = Button(self, text = "move", command = self.move_by_policy)
        move_button.configure(width = 10, activebackground = "#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.62, HEIGHT * UNIT + 10, window =move_button)

        #reset 버튼
        reset_button = Button(self, text = "reset", command = self.reset)
        reset_button.configure(width = 10, activebackground = "#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 10, window =reset_button)

        # 그리드 생성
        for col in range(0, WIDTH * UNIT, UNIT):            # 가로 선 긋기
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT , UNIT):          # 세로 선 긋기
            x0, y0, x1, y1 = 0, row, HEIGHT * UNIT , row
            canvas.create_line(x0, y0, x1, y1)

        self.player = canvas.create_image(50, 50, image=self.shapes[0])
        canvas.create_image(250, 150, image=self.shapes[1])
        canvas.create_image(150, 250, image=self.shapes[1])
        canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    # 이미지 불러오기
    def load_images(self):
        up = PhotoImage(Image.open("maze-algorithem\\img\\up.png").resize((13,13)))
        right = PhotoImage(Image.open("maze-algorithem\\img\\right.png").resize((13,13)))
        left = PhotoImage(Image.open("maze-algorithem\\img\\left.png").resize((13,13)))
        down = PhotoImage(Image.open("maze-algorithem\\img\\down.png").resize((13,13)))
        player = PhotoImage(Image.open("maze-algorithem\\img\\player.png").resize((65,65)))
        wall = PhotoImage(Image.open("maze-algorithem\\img\\wall.png").resize((65,65)))
        goal = PhotoImage(Image.open("maze-algorithem\\img\\goal.png").resize((65,65)))
        return (up, down, left, right), (player, wall, goal)

    # 캔바스 초기화
    def reset(self):
        if self.is_moving == 0:
            self.evaluation_count = 0
            self.improvement_count = 0
            for i in self.texts:
                self.canvas.delete(i)

            for i in self.arrows:
                self.canvas.delete(i)
            
            self.agent.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)] # _는 단순 반복문에 사용
            self.agent.policy_table = ([[[0.25, 0,25, 0,25, 0,25]] * WIDTH for _ in range(HEIGHT)])

            self.agent.policy_table[2][2] = []
            x, y = self.canvas.coords(self.player)
            self.canvas.move(self.player, UNIT / 2 - x, UNIT / 2 - y)

    # 보상 값을 텍스트로 나타냄
    def text_value(self, row, col, contents, font = 'Helvetica', size = 10, style = 'normal', anchor = "nw"):
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        value_text = self.canvas.create_text(x, y, fill="black", text = contents, font = font, anchor = anchor)
        return self.texts.append(value_text)

    # 가치 값을 텍스트로 나타냄 사실 위랑 같은 함수다 이게 필요할까?
    def text_reward(self, row, col, contents, font='Helvetica', size = 10, style = 'normal', anchor = "nw"):
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        reward_text = self.canvas.create_text(x, y, fill="black", text = contents, font = font, anchor = anchor)
        return self.texts.append(reward_text)

    # 플레이어 움직이기
    def player_move(self,action):
        base_action = np.array([0, 0])
        location = self.find_player()
        self.render()
        if action == 0 and location[0] > 0: # 상
            base_action[1] -= UNIT
        elif action == 1 and location[0] < HEIGHT -1: # 하
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0: # 좌
            base_action[0] -= UNIT
        elif action == 3 and location[1] < WIDTH - 1: # 우
            base_action[0] += UNIT
        # 플레이어 움직임
        self.canvas.move(self.player, base_action[0], base_action[1])

    # 현제 플레이어의 위치
    def find_player(self):
        temp =self.canvas.coords(self.player)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5
        return int(y), int(x)

    # 움직임을 받아와서 움직인다.
    def move_by_policy(self):
        if self.improvement_count != 0 and self.is_moving != 1:
            self.is_moving = 1

            x, y = self.canvas.coords(self.player)
            self.canvas.move(self.player, UNIT / 2 - x, UNIT / 2 - y)

            x, y = self.find_player()
            while len(self.agent.policy_table[x][y]) != 0:
                self.after(100, self.player_move(self.agent.get_action([x, y])))
                x, y = self.find_player()
            self.is_moving = 0

    # 화살표를 그린다
    def draw_one_arrow(self, col, row, policy):
        if col == 2 and row == 2:
            return 

        if policy[0] > 0: # 위로 향하는 화살표
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.up))

        if policy[1] > 0: # 아래로 향하는 화살표
            origin_x, origin_y = 50 + (UNIT * row), 90 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.down))

        if policy[2] > 0: # 왼쪽으로 향하는 화살표
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.left))

        if policy[3] > 0: # 오른쪽으로 향하는 화살표
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.right))


    def draw_from_policy(self, policy_table):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.draw_one_arrow(i, j, policy_table[i][j])

    def print_value_table(self, value_table):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.text_value(i, j, round(value_table[i][j], 2))

    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.player)
        self.update()

    def evaluate_policy(self):
        self.evaluation_count += 1
        for i in self.texts:
            self.canvas.delete(i)
        self.agent.policy_evaluation()
        self.print_value_table(self.agent.value_table)
    
    def improve_policy(self):
        self.improvement_count += 1
        for i in self.arrows:
            self.canvas.delete(i)
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)

class Env:
    def __init__(self):
        self.transition_probability = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[2][2] = 1   # goal 의 보상 1
        self.reward[1][2] = -1  # wall 의 보상 -1
        self.reward[2][1] = -1  # wall 의 보상 -1
        self.all_state = []

        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1 if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT -1 if state[1] > HEIGHT -1 else state[1])
        return state
    
    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state



    
        
    
        
