import copy
import importlib
import itertools
from typing import Tuple, Dict, Callable, List, Optional, Union, Sequence
import sys
sys.path.append('C:/Users/cml/PycharmProjects/highway-env/do-mpc')
import do_mpc
from casadi import *
import numpy as np

def bring_positions(vehicle_list : list):
    """ list를 풀기"""
    l = len(vehicle_list)
    vehicle = np.empty((0, 4))
    for i in range(0, l):
        vehicle = np.append(vehicle, np.array(
            [[vehicle_list[i].position[0], vehicle_list[i].position[1], vehicle_list[i].speed,
              vehicle_list[i].lane_index[2]]]), axis=0)

    return vehicle

def overtake_lane(ego_pos, vehicles, lanes) -> Tuple[str, int]:
    """
    서행 전방차량을 추월 할 차선을 선택. pos : x,y,v,lane_id
    gap 이 더 큰 위치로 간다.
    """
    if ego_pos[3] == 0:
        decision = 'RIGHT'
        lane_overtake = ego_pos[3] + 1
    elif ego_pos[3] == lanes-1:
        decision = 'LEFT'
        lane_overtake = ego_pos[3] - 1
    else:
        front_left = np.where((vehicles[:, 3] == ego_pos[3]-1) & (vehicles[:, 0] > ego_pos[0]))[0]
        front_right = np.where((vehicles[:, 3] == ego_pos[3]+1) & (vehicles[:, 0] > ego_pos[0]))[0]

        left_gap = 1000 if front_left.size == 0 else vehicles[front_left[0], 0] - ego_pos[0]
        right_gap = 1000 if front_right.size == 0 else vehicles[front_right[0], 0] - ego_pos[0]

        lane_overtake = ego_pos[3]-1 if left_gap > right_gap else ego_pos[3]+1
        decision = 'LEFT' if left_gap > right_gap else 'RIGHT'
    return (decision, lane_overtake)


def get_index_list(list, index):
    return [list[i] for i in index]

def emap(d: float) -> float:
    """Exponential map of value v with range x to desired range y."""
    a = 0.0004516
    b =  0.6997

    return min(a*np.exp(b*d)-1, 1)

def front_dist(ego, road_vehicles) :
    vehicles = bring_positions(road_vehicles)  # vehicles : x, y, v, lane id
    front_id = np.where((vehicles[:, 3] == ego[3]) & (vehicles[:, 0] > ego[0]))[0]
    if len(front_id) == 0:
        return 100
    else:
        front = vehicles[front_id[0]]
        return front[0] - ego[0]

def front_vehicle(ego, close_vehicles, target_lane = None):
    vehicles = bring_positions(close_vehicles)
    if target_lane:
        front_id = np.where((vehicles[:, 3]*4 == target_lane) & (vehicles[:, 0] > ego[0]))[0]
    else:
        front_id = np.where((vehicles[:, 3] == ego[3]) & (vehicles[:, 0] > ego[0]))[0]
    if len(front_id) == 0:
        ego[0] = ego[0] + 100
        return ego
    else:
        front = vehicles[front_id[0]]
        return front

def front_speed(ego, road_vehicles):
    vehicles = bring_positions(road_vehicles)  # vehicles : x, y, v, lane id
    front_id = np.where((vehicles[:, 3] == ego[3]) & (vehicles[:, 0] > ego[0]))[0]
    if len(front_id) == 0:
        return 0
    else:
        front = vehicles[front_id[0]]
        return front[2] - ego[2]


def car_dist(ego, close_vehicles):
    vehicles = bring_positions(close_vehicles)  # vehicles : x, y, v, lane id
    same_lane_id = np.where((vehicles[:, 3] == ego[3]))[0]
    dist = np.zeros([5,1])
    if len(same_lane_id) == 0:
        return dist

    else:
        same_lane_vehicles = vehicles[same_lane_id,:]
        same_lane_id += 1
        dist[same_lane_id] = np.reshape(same_lane_vehicles[:,0] - ego[0],(len(same_lane_id),1))
        dist = np.clip(dist,-100,100) * 0.01
        return dist


def Model():
    #action = [acceleration, steering angle]
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    """x,y,psi,v"""
    _x = model.set_variable(var_type='_x', var_name='x', shape=(5, 1))
    _x_obs = model.set_variable(var_type='_x', var_name='x_obs', shape=(3, 1))
    _distance = model.set_variable(var_type='_x', var_name='distance')
    # Input struct (optimization variables):
    """beta, acc"""
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))


    # target_pos = model.set_variable(var_type='_tvp', var_name='target_pos')
    # model.set_variable(var_type='_p', var_name='target_v')
    Wv = 1
    Wp = 1

    Wdy,Wdv, Wdh = 10, 10, 10
    target_v = 30
    const_v = 30
    const_psi = 0
    const_beta = 0
    # dt = model.set_variable(var_type='_p', var_name='dt')
    # x, y
    dt = 0.1
    # close_vehicles = model.set_variable(var_type='_tvp', var_name='close_vehicles', shape=(4, 2))

    car_size = np.array([5, 2])

    # A = np.array([[1, 0, 0, dt, 0],
    #               [0, 1, dt*const_v, 0, 0],
    #               [0, 0, 1, 0, 0],
    #               [0, 0, 0, 1, 0],
    #               [0, 0, 0, 0, 1]])
    #
    # B = np.array([[0, 0],
    #               [0, dt/2*const_v/7],
    #               [0, dt/5*const_v/7],
    #               [dt, 0],
    #               [0, 0]])
    A = np.array([[1, 0, 0, dt, 0],
                  [0, 1, dt/2*const_v, dt/2*(const_psi+1/2*const_beta), 0],
                  [0, 0, 1, dt/8*const_beta, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    B = np.array([[0, 0],
                  [0, dt/4*const_v],
                  [0, dt/8*const_v],
                  [dt, 0],
                  [0, 0]])

    A_obs = np.array([[1., 0., dt],
                      [0., 1., 0.],
                      [0., 0., 1.]])


    x_next = A@_x + B@_u
    x_next_obs = A_obs@_x_obs

    # model.set_rhs('distance', x_next_obs[0] - x_next[0] - 1.698*x_next[3])
    model.set_rhs('distance', x_next_obs[0] - x_next[0])

    model.set_rhs('x', x_next)
    model.set_rhs('x_obs', x_next_obs)

    L = 5 * 1.5
    W = 2 * 1.5
    # model.set_variable('_z', 'area', shape=(4, 1))
    # area = (_x[0] - _x_obs[1, 4, 7, 10]) ** 2 / L ** 2 + (_x[1] - _x_obs[2, 5, 8, 11]) ** 2 / W ** 2

    # model.set_rhs('area', area)


    cost_v = Wv*(_x[3]-target_v)**2
    cost_pos = Wp*(_x[1]-_x[4])**2
    cost_derivationY = Wdy*(x_next[1]-_x[1])**2
    cost_derivationH = Wdh*(x_next[2]-_x[2])**2
    cost_derivationV = Wdv*(x_next[3]-_x[3])**2
    model.set_expression('cost_v', cost_v)
    model.set_expression('cost_pos', cost_pos)
    model.set_expression('cost_derivationY', cost_derivationY)
    model.set_expression('cost_derivationH', cost_derivationH)
    model.set_expression('cost_derivationV', cost_derivationV)

    # model.set_expression('area', area)


    pos_set = model.set_variable('_tvp', 'pos_set', shape=(1,1))
    model.set_expression('tvp', pos_set)

    model.setup()
    return model

def MPC(model):
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 30,
        't_step': 0.1,
        'state_discretization': 'discrete',
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }
    # global_path = np.load('save_positions/save_global_4.npy')

    mpc.set_param(**setup_mpc)

    # objective function : follow target position and speed of 30m/s
    # mterm = (model.x[1] - model.target_pos) ** 2 + (np.sqrt(model.x[2] ** 2 + model.x[3] ** 2) - target_v) ** 2

    # global_path = np.load('/Users/cml/highway-env/scripts/save_positions/save_global_2.npy')
    #
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_ind):
        L = 5 * 2.5
        W = 2 * 1.5
        for k in range(setup_mpc['n_horizon'] + 1):
            # if mpc.opt_x_num is None :
            #     tvp_template['_tvp', k, 'pos_set'] = np.zeros([1,1]) #mpc.opt_x_num['_x'][k]
            # else:
            #     x_ego = mpc.opt_x_num['_x'][k][0][0][0]
            #     y_ego = mpc.opt_x_num['_x'][k][0][0][1]
            #     x_obs = mpc.opt_x_num['_x'][k][0][0][5,8,11,14]
            #     y_obs = mpc.opt_x_num['_x'][k][0][0][6,9,12,15]
            #     dx = np.abs(x_ego-x_obs)
            #     dy = np.abs(y_ego-y_obs)
            #     for i in range(4):
            #         if dx[i] < L and dy[i] < W:
            #             dx[i] = L/max(0.5,dx[i]) # min(L/dx[i],1)
            #             dy[i] = W/max(0.2,dy[i]) #min(W/dy[i],1)
            #         else:
            #             dx[i] = 0
            #             dy[i] = 0
            #     tvp_template['_tvp', k, 'pos_set'] = sum(dx*dy)
            tvp_template['_tvp', k, 'pos_set'] = np.zeros([1, 1])

                # tvp_template['_tvp', k, 'pos_set'] = (x_e3go-x_obs)**2/L**2 + (y_ego-y_obs)**2/W**2
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mterm = model.aux['cost_v'] #+ model.aux['cost_pos']

    lterm = model.aux['cost_v'] + model.aux['cost_pos'] + model.aux['cost_derivationY'] + model.aux[
        'cost_derivationV'] + model.aux['cost_derivationH']

    # mterm = model.aux['cost_v']+ model.aux['cost_pos']
    #
    # lterm = model.tvp['pos_set']

    mpc.set_objective(mterm=mterm, lterm=lterm)

    # constraints : bound / collision avoidance
    # y boundary 는 4차선을 기준으로 함

    max_x = np.array([[5000.0], [14.0], [np.pi/2], [40], [14.0]])
    min_x = np.array([[0.0], [-2.0], [-np.pi/2], [13], [-2]])

    max_u = np.array([[5], [np.pi/10]])
    min_u = np.array([[-5],[-np.pi/10]])

    # lower bounds of the states
    mpc.bounds['lower', '_x', 'x'] = min_x
    mpc.bounds['upper', '_x', 'x'] = max_x

    mpc.bounds['lower', '_u', 'u'] = min_u
    mpc.bounds['upper', '_u', 'u'] = max_u

    mpc.bounds['lower', '_x', 'distance'] = 12.5 #-2.563

    mpc.set_rterm(u= np.array([[1], [1]]))

    mpc.setup()
    return mpc

# def Model_Continuous():
#     #action = [acceleration, steering angle]
#     model_type = 'discrete'
#     model = do_mpc.model.Model(model_type)
#
#     """x,y,psi,v"""
#     dx = model.set_variable('_x','dx')
#     dy = model.set_variable('_x','dx')
#     dpsi = model.set_variable('_x','dx')
#     dv = model.set_variable('_x','dx')
#     dbeta = model.set_variable('_x', 'dx')
#
#     # Input struct (optimization variables):
#     """beta, acc"""
#     _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))
#
#     # target_pos = model.set_variable(var_type='_tvp', var_name='target_pos')
#     # model.set_variable(var_type='_p', var_name='target_v')
#     Wv = 1
#     Wp = 50
#     target_v = 30
#     const_v = target_v
#     const_psi = 0
#     const_beta = 0
#     # dt = model.set_variable(var_type='_p', var_name='dt')
#     # x, y
#     dt = 0.3
#     # close_vehicles = model.set_variable(var_type='_tvp', var_name='close_vehicles', shape=(4, 2))
#
#     car_size = np.array([5, 2])
#
#     A = np.array([[1, 0, 0, dt, 0],
#                   [0, 1, dt*const_v, 0, 0],
#                   [0, 0, 1, 0, 0],
#                   [0, 0, 0, 1, 0],
#                   [0, 0, 0, 0, 1]])
#
#     B = np.array([[0, 0],
#                   [0, dt/2*const_v],
#                   [0, dt/5*const_v],
#                   [dt, 0],
#                   [0, 0]])
#
#     # A = np.array([[1, 0, 0, dt, 0],
#     #               [0, 1, dt / 2 * const_v, dt / 2 * (const_psi + 1 / 2 * const_beta), 0],
#     #               [0, 0, 1, dt / 8 * const_beta, 0],
#     #               [0, 0, 0, 1, 0],
#     #               [0, 0, 0, 0, 1]])
#     #
#     # B = np.array([[0, 0],
#     #               [0, dt / 4 * const_v],
#     #               [0, dt / 8 * const_v],
#     #               [dt, 0],
#     #               [0, 0]])
#
#     x_next = A@_x + B@_u
#
#     model.set_rhs('x', x_next)
#
#     cost_v = Wv*(_x[3]-target_v)**2
#     cost_pos = Wp*(_x[1]-_x[4])**2
#
#     model.set_expression('cost_v', cost_v)
#     model.set_expression('cost_pos', cost_pos)
#
#     pos_set = model.set_variable('_tvp', 'pos_set')
#     model.set_expression('tvp', pos_set)
#
#     model.setup()
#     return model
#
# def MPC_Continuous(model):
#     mpc = do_mpc.controller.MPC(model)
#     setup_mpc = {
#         'n_robust': 0,
#         'n_horizon': 5,
#         't_step': 0.1,
#         'state_discretization': 'discrete',
#         'store_full_solution': True,
#         # Use MA27 linear solver in ipopt for faster calculations:
#         # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
#     }
#     # global_path = np.load('save_positions/save_global_4.npy')
#
#     mpc.set_param(**setup_mpc)
#
#     # objective function : follow target position and speed of 30m/s
#     # mterm = (model.x[1] - model.target_pos) ** 2 + (np.sqrt(model.x[2] ** 2 + model.x[3] ** 2) - target_v) ** 2
#
#     # global_path = np.load('/Users/cml/highway-env/scripts/save_positions/save_global_2.npy')
#     #
#     tvp_template = mpc.get_tvp_template()
#
#     # When to switch setpoint:
#     t_switch = 4  # seconds
#     ind_switch = t_switch // setup_mpc['t_step']
#
#     def tvp_fun(t_ind):
#         ind = t_ind // setup_mpc['t_step']
#         if ind <= ind_switch:
#             tvp_template['_tvp', :, 'pos_set'] = 1
#         else:
#             tvp_template['_tvp', :, 'pos_set'] = 1
#
#         return tvp_template
#
#     mpc.set_tvp_fun(tvp_fun)
#
#     mterm = model.aux['cost_v'] + model.aux['cost_pos']
#
#     lterm = model.aux['cost_v'] + model.aux['cost_pos']
#
#     mpc.set_objective(mterm=mterm, lterm=lterm)
#
#     # constraints : bound / collision avoidance
#     # y boundary 는 4차선을 기준으로 함
#
#     max_x = np.array([[5000.0], [14.0], [np.pi/2], [40], [14.0]])
#     min_x = np.array([[0.0], [-2.0], [-np.pi/2], [10], [-2]])
#
#     max_u = np.array([[5],[np.pi/10]])
#     min_u = np.array([[-5],[-np.pi/10]])
#
#     # lower bounds of the states
#     mpc.bounds['lower', '_x', 'x'] = min_x
#     mpc.bounds['upper', '_x', 'x'] = max_x
#
#     mpc.bounds['lower', '_u', 'u'] = min_u
#     mpc.bounds['upper', '_u', 'u'] = max_u
#
#
#     mpc.set_rterm(u= np.array([[1], [1]]))
#
#     mpc.setup()
#     return mpc

# def tvp_func(t_now):
#     for k in range(n)






