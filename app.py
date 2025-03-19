from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

grid_size = 5  # 預設網格大小
grid = np.zeros((grid_size, grid_size))
start_pos = None
goal_pos = []
dead_pos = []
policy = {}  # 全局策略變數
@app.route('/')
def index():
    return render_template('index.html', grid_size=grid_size)

@app.route('/set_grid', methods=['POST'])
def set_grid():
    global grid_size, grid, start_pos, goal_pos, dead_pos, policy
    data = request.json
    grid_size = int(data['size'])
    grid = np.zeros((grid_size, grid_size))
    start_pos = None
    goal_pos = []
    dead_pos = []
    policy = {}
    return jsonify({'message': 'Grid size updated', 'grid_size': grid_size})

@app.route('/set_cell', methods=['POST'])
def set_cell():
    global start_pos, goal_pos, dead_pos
    data = request.json
    x, y, cell_type = data['x'], data['y'], data['type']
    
    if cell_type == 'start':
        start_pos = (x, y)
    elif cell_type == 'goal':
        goal_pos.append((x, y))
    elif cell_type == 'dead':
        dead_pos.append((x, y))
    
    return jsonify({'message': 'Cell updated', 'start': start_pos, 'goal': goal_pos, 'dead': dead_pos})

@app.route('/get_grid', methods=['GET'])
def get_grid():
    return jsonify({
        'grid_size': grid_size,
        'start': start_pos,
        'goal': goal_pos,
        'dead': dead_pos
    })

@app.route('/get_policy', methods=['GET'])
def get_policy():
    global policy
    actions = ['↑', '↓', '←', '→']
    policy = {}

    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) not in goal_pos and (i, j) not in dead_pos:
                policy[f"{i},{j}"] = np.random.choice(actions)  # 用字串化座標

    return jsonify({'policy': policy})
@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    global policy
    gamma = 0.9  # 折扣因子
    threshold = 0.0001  # 收斂閥值
    V = np.zeros((grid_size, grid_size))

    while True:
        delta = 0
        new_V = np.copy(V)

        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in goal_pos:
                    new_V[i, j] = 1  # 終點值設為 1
                    continue
                if (i, j) in dead_pos:
                    new_V[i, j] = -1  # 障礙物設為 -1
                    continue

                action = policy.get(f"{i},{j}", '↑')  # 讀取策略
                next_i, next_j = i, j
                if action == '↑': next_i = max(i - 1, 0)
                if action == '↓': next_i = min(i + 1, grid_size - 1)
                if action == '←': next_j = max(j - 1, 0)
                if action == '→': next_j = min(j + 1, grid_size - 1)

                reward = -0.04  # 移動懲罰 (-1 調整為 -0.04)
                if (next_i, next_j) in goal_pos:
                    reward = 1
                elif (next_i, next_j) in dead_pos:
                    reward = -1

                new_V[i, j] = reward + gamma * V[next_i, next_j]
                delta = max(delta, abs(new_V[i, j] - V[i, j]))

        V = new_V
        if delta < threshold:
            break

    return jsonify({'values': V.tolist()})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
