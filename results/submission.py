
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import numpy as np
import random

def get_nearest_cells(x,y):
    #현재 cell에서 접근 가능한 모든 cell을 반환
    result = []
    for i in (-1,+1):
        result.append(((x+i+7)%7, y))
        result.append((x, (y+i+11)%11))
    return result
 #그대로 사용해도 별 문제 없을것으로 보임
#새로운 함수 추가
def find_closest_food(table):
    # returns the first step toward the closest food item
    new_table = table.copy()
    #7 *11 게임 테이블 카피해서 새로 만듬 #이동 위해 임시적으로 쓰는 것으로 보임
    
    # (direction of the step, axis, code)
    
    possible_moves = [
        (1, 0, 1), #방향쪽으로 직진 축 0 남
        (-1, 0, 2), #방향쪽으로 후진 축 0 북
        (1, 1, 3), #방향쪽으로 직진 축 1 동
        (-1, 1, 4) #방향쪽으로 후진 축 1 서
    ]
    
    # shuffle possible options to add variability
    random.shuffle(possible_moves)
    
    
    updated = False
    for roll, axis, code in possible_moves:

        shifted_table = np.roll(table, roll, axis)
        #이동해서 이렇게 된거 같음
        #테이블을 축에 맞게 -1/+1 시프트작업 수행
        if (table == -2).any() and (shifted_table[table == -2] == -3).any(): # we have found some food at the first step
            #any함수 -> 배열 안에 해당 원소가 존재하는지
            #음식이 테이블에 존재하고 존재한 위치에 내 헤드가 있을 경우 
            return code
        else:
            mask = np.logical_and(new_table == 0,shifted_table == -3)
            # 비교연산자 사용할 경우 true false 배열이 반환됨
            # 원래의 맵이 비어 있으며 헤드가 존재한다 #이동 가능
            if mask.sum() > 0:
                updated = True
                #원래 비어있는곳에 헤드가 존재 -> 이동할 수 있으니 업데이트 하십시오.
            new_table += code * mask
            #new_table에 이동지표 더함
        if (table == -2).any() and shifted_table[table == -2][0] > 0: # we have found some food
            #원래 테이블 안에 음식 존재하고, 이동된 테이블의 음식 존재하는 
            #a[b==-1] -> 1차원 행렬 반환
            return shifted_table[table == -2][0]
        
        # else - update new reachible cells
        mask = np.logical_and(new_table == 0,shifted_table > 0)
        if mask.sum() > 0:
            updated = True
        new_table += shifted_table * mask

    # if we updated anything - continue reccurison
    if updated:
        return find_closest_food(new_table)
    # if not - return some step
    else:
        return table.max()

last_step = None

def agent(obs_dict, config_dict):
    global last_step
    
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)


    table = np.zeros((7,11))
    # 0 - emply cells
    # -1 - obstacles
    # -4 - possible obstacles
    # -2 - food
    # -3 - head
    # 1,2,3,4 - reachable on the current step cell, number is the id of the first step direction
    
    legend = {
        1: 'SOUTH',
        2: 'NORTH',
        3: 'EAST',
        4: 'WEST'
    }
    
    # let's add food to the map
    for food in observation.food:
        x,y = row_col(food, configuration.columns)
        table[x,y] = -2 # food
    #음식 테이블에 추가
    # let's add all cells that are forbidden
    for i in range(4):
        opp_goose = observation.geese[i]
        if len(opp_goose) == 0:
            continue
            
        is_close_to_food = False
            
        if i != player_index:
            x,y = row_col(opp_goose[0], configuration.columns)
            possible_moves = get_nearest_cells(x,y) # head can move anywhere
            
            for x,y in possible_moves:
                if table[x,y] == -2:
                    is_close_to_food = True
            
                table[x,y] = -4 # possibly forbidden cells
        
        # usually we ignore the last tail cell but there are exceptions
        tail_change = -1
        if obs_dict['step'] % 40 == 39:
            tail_change -= 1
        
        # we assume that the goose will eat the food
        if is_close_to_food:
            tail_change += 1
        if tail_change >= 0:
            tail_change = None
            

        for n in opp_goose[:tail_change]:
            x,y = row_col(n, configuration.columns)
            table[x,y] = -1 # forbidden cells
    #만질필요 없음
    # going back is forbidden according to the new rules
    x,y = row_col(player_head, configuration.columns)
    if last_step is not None:
        if last_step == 1:
            table[(x + 6) % 7,y] = -1
        elif last_step == 2:
            table[(x + 8) % 7,y] = -1
        elif last_step == 3:
            table[x,(y + 10)%11] = -1
        elif last_step == 4:
            table[x,(y + 12)%11] = -1
        
    # add head position
    table[x,y] = -3
    
    # the first step toward th e nearest food
    step = int(find_closest_food(table))
    #그냥 음식을 향해 그리디 하게 가는것이 전부
    #다른요소를 고려하고 추가하면 될거같아 보임
    # if there is not available steps try to go to possibly dangerous cell
    if step not in [1,2,3,4]:
        x,y = row_col(player_head, configuration.columns)
        if table[(x + 8) % 7,y] == -4:
            step = 1
        elif table[(x + 6) % 7,y] == -4:
            step = 2
        elif table[x,(y + 12)%11] == -4:
            step = 3
        elif table[x,(y + 10)%11] == -4:
            step = 4
                
    # else - do a random step and lose
        else:
            step = np.random.randint(4) + 1
    
    last_step = step
    return legend[step]
