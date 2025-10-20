import numpy as np 
import gymnasium as gym 
from gymnasium import error ,spaces ,utils 
from gymnasium .utils import seeding 
from typing import List ,Optional 

import imageio ,random ,copy 
import sys ,time ,os ,base64 ,io ,random 
os .environ ['PYOPENGL_PLATFORM']='egl'

import IPython ,functools ,matplotlib 
import matplotlib .pyplot as plt 
from matplotlib import colors 
from PIL import Image as Image 
from tqdm import tqdm 

from IPython .display import HTML 

from IPython .display import HTML 
def eval_policy (env ,policy =None ,num_episodes =10 ):
    """Evaluate a model (i.e., policy) running on the input environment"""

    obs ,_ =env .reset ()
    prev_obs =obs 
    counter =0 
    done =False 
    num_runs =0 
    episode_reward =0 
    episode_rewards =[]
    while num_runs <num_episodes :
        if policy is not None :
            action =policy (obs )
        else :
            action =env .action_space .sample ()

        prev_obs =obs 
        obs ,reward ,done ,truncated ,info =env .step (action )
        counter +=1 
        episode_reward +=reward 
        if done or truncated :
            counter =0 
            num_runs +=1 
            obs ,_ =env .reset ()
            episode_rewards .append (episode_reward )
            episode_reward =0 
    return episode_rewards 

def save_video_of_model (env_name ,model =None ,suffix ="",num_episodes =10 ):
    """
    Record a video that shows the behavior of an agent following a model
    (i.e., policy) on the input environment
    """
    from moviepy .video .io .ImageSequenceClip import ImageSequenceClip 
    from pyvirtualdisplay import Display 
    display =Display (visible =0 ,size =(400 ,300 ))
    display .start ()

    env =gym .make (env_name )
    obs ,_ =env .reset ()
    prev_obs =obs 

    filename =env_name +suffix +".mp4"
    recorded_frames =[]

    counter =0 
    done =False 
    num_runs =0 
    returns =0 
    while num_runs <num_episodes :
        frame =env .render ()
        recorded_frames .append (frame )

        if "Gridworld"in env_name :
            input_obs =obs 
        else :
            raise ValueError (f"Unknown env for saving: {env_name}")

        if model is not None :
            action =model (input_obs )
        else :
            action =env .action_space .sample ()

        prev_obs =obs 
        obs ,reward ,done ,truncated ,info =env .step (action )
        counter +=1 
        returns +=reward 
        if done or truncated :
            num_runs +=1 
            obs ,_ =env .reset (options ={'random':True })

    clip =ImageSequenceClip (recorded_frames ,fps =8 )
    clip .write_videofile (filename ,logger =None )

    print ("Successfully saved {} frames into {}!".format (counter ,filename ))
    return filename ,returns /num_runs 

def play_video (filename ,width =None ):
    """Play the input video"""

    from base64 import b64encode 
    mp4 =open (filename ,'rb').read ()
    data_url ="data:video/mp4;base64,"+b64encode (mp4 ).decode ()
    html ="""
    <video width=400 controls>
          <source src="%s" type="video/mp4">
    </video>
    """%data_url 
    return html 

def render_value_map_with_action (env ,Q ,policy =None ):
    '''
    Render a state (or action) value grid map.
    V[s] = max(Q[s,a])
    '''
    Q =Q .copy ()
    from matplotlib .colors import LinearSegmentedColormap 
    n =env .unwrapped .grid_map_shape [0 ]
    m =env .unwrapped .grid_map_shape [1 ]

    if len (np .shape (Q ))>1 :
        V =np .amax (Q ,axis =1 )
        V =V .reshape ((m ,n )).T 
    else :
        V =Q .reshape ((m ,n )).T 

    import itertools 
    symbol =['.','^','v','<','>']
    x =range (0 ,env .unwrapped .grid_map_shape [0 ])
    y =range (0 ,env .unwrapped .grid_map_shape [1 ])

    min_val =V [0 ,0 ]
    obstacles =np .zeros ([n ,m ])
    for obstacle in env .unwrapped .obstacles :
        posx =obstacle %env .unwrapped .grid_map_shape [0 ]
        posy =obstacle //env .unwrapped .grid_map_shape [0 ]
        V [posx ,posy ]=min_val 
        obstacles [posx ,posy ]=1 

    plt .imshow (V ,cmap ='jet',interpolation ='nearest')
    for s in range (env .observation_space .n ):
        twod_state =env .unwrapped .serial_to_twod (s )
        state_inds =s 
        best_action =policy (s )
        plt .plot ([twod_state [1 ]],[twod_state [0 ]],marker =symbol [best_action ],linestyle ='none',color ='k')

    dark_low =((0. ,1. ,1. ),
    (.3 ,1. ,0. ),
    (1. ,0. ,0. ))

    cdict ={'red':dark_low ,
    'green':dark_low ,
    'blue':dark_low }

    cdict3 ={'red':dark_low ,
    'green':dark_low ,
    'blue':dark_low ,
    'alpha':((0.0 ,0.0 ,0.0 ),
    (0.3 ,0.0 ,1.0 ),
    (1.0 ,1.0 ,1.0 ))
    }
    dropout_high =LinearSegmentedColormap ('Dropout',cdict3 )
    plt .imshow (obstacles ,cmap =dropout_high )
    plt .show ()

def collect_traj (env ,policy =None ,num_episodes =10 ):
    """Collect trajectories (rollouts) following the input policy"""
    obs ,_ =env .reset ()
    prev_obs =obs 
    done =False 
    num_runs =0 
    episode_rewards =[]
    episode_reward =0 
    traj =[]
    trajs =[]

    while num_runs <num_episodes :
        input_obs =obs 
        if policy is not None :
            action =policy (input_obs )
        else :
            action =env .action_space .sample ()
        prev_obs =obs 
        traj .append (obs )
        obs ,reward ,done ,truncated ,info =env .step (action )
        episode_reward +=reward 
        if done or truncated :
            num_runs +=1 
            traj .append (obs )
            trajs .append (traj )

            traj =[]
            obs ,_ =env .reset ()

            episode_rewards .append (episode_reward )
            episode_reward =0 
    return trajs 

def plot_trajs (env ,trajectories ):
    """Plot the input trajectories"""
    from matplotlib .colors import LinearSegmentedColormap 
    n =env .unwrapped .grid_map_shape [0 ]
    m =env .unwrapped .grid_map_shape [1 ]
    V =np .zeros ([n ,m ])
    obstacles =np .zeros ([n ,m ])
    for obstacle in env .unwrapped .obstacles :
        posx =obstacle %env .unwrapped .grid_map_shape [0 ]
        posy =obstacle //env .unwrapped .grid_map_shape [0 ]
        obstacles [posx ,posy ]=1 

    dark_low =((0. ,1. ,1. ),
    (.3 ,1. ,0. ),
    (1. ,0. ,0. ))
    cdict ={'red':dark_low ,
    'green':dark_low ,
    'blue':dark_low }
    cdict3 ={'red':dark_low ,
    'green':dark_low ,
    'blue':dark_low ,
    'alpha':((0.0 ,0.0 ,0.0 ),
    (0.3 ,0.0 ,1.0 ),
    (1.0 ,1.0 ,1.0 ))
    }
    dropout_high =LinearSegmentedColormap ('Dropout',cdict3 )
    plt .imshow (obstacles ,cmap =dropout_high )
    for trajectory in trajectories :
        traj_2d =np .array ([env .unwrapped .serial_to_twod (s )for s in trajectory ])
        x =traj_2d [:,0 ]
        y =traj_2d [:,1 ]
        plt .plot (y ,x ,alpha =0.1 ,color ='r')

    plt .show ()

def plot_grid (env ):
    """Plot the input trajectories"""
    from matplotlib .colors import LinearSegmentedColormap 
    n =env .unwrapped .grid_map_shape [0 ]
    m =env .unwrapped .grid_map_shape [1 ]
    V =np .zeros ([n ,m ])
    grid_map =np .zeros ([n ,m ])

    for obstacle in env .unwrapped .obstacles :
        posx =obstacle %env .unwrapped .grid_map_shape [0 ]
        posy =obstacle //env .unwrapped .grid_map_shape [0 ]
        grid_map [posx ,posy ]=1 

    for trap in env .unwrapped .traps :
        posx =trap %env .unwrapped .grid_map_shape [0 ]
        posy =trap //env .unwrapped .grid_map_shape [0 ]
        grid_map [posx ,posy ]=-1 

    cmap =colors .ListedColormap (['red','white','black'])
    bounds =[-1.5 ,-0.5 ,0.5 ,1.5 ]
    norm =colors .BoundaryNorm (bounds ,cmap .N )

    plt .imshow (grid_map ,cmap =cmap ,norm =norm )

    x1 =range (env .unwrapped .grid_map_shape [0 ])
    y1 =range (env .unwrapped .grid_map_shape [1 ])

    for i in range (env .unwrapped .grid_map_shape [0 ]):
        for j in range (env .unwrapped .grid_map_shape [1 ]):
            binvalue =int (j *env .unwrapped .grid_map_shape [0 ]+i )
            plt .text (y1 [j ]+0.0 ,
            x1 [i ]+0.0 ,
            binvalue ,
            color ='white'if binvalue in env .unwrapped .obstacles else 'black',
            ha ='center',va ='center',size =8 )

    plt .xticks ([])
    plt .yticks ([])
    plt .show ()

class BaseGridEnv (gym .Env ):
    metadata ={'render_modes':['human','rgb_array'],
    "render_fps":4 }
    def __init__ (self ,render_mode ='rgb_array',size =[8 ,10 ],start =None ,
    epsilon =0.2 ,obstacle =None ,trap =None ):
        """
        An initialization function

        Parameters
        ----------
        size: a list of integers
            the dimension of 2D grid environment
        start: integer
            start state (i.e., location)
        epsilon: float
            the probability of taking random actions
        obstacle:

        """
        self .render_mode =render_mode 
        self .unwrapped .grid_map_shape =[size [0 ],size [1 ]]
        self .epsilon =epsilon 
        self .obstacles =obstacle 
        self .traps =trap 

        ''' set observation space and action space '''
        self .observation_space =spaces .Discrete (size [0 ]*size [1 ])
        self .action_space =spaces .Discrete (5 )

        self .start_state =0 
        self .terminal_state =size [0 ]*size [1 ]-1 

    def serial_to_twod (self ,ind ):
        """Convert a serialized state number to a 2D map's state coordinate"""
        return np .array ([ind %self .unwrapped .grid_map_shape [0 ],ind //self .unwrapped .grid_map_shape [0 ]])

    def twod_to_serial (self ,twod ):
        """Convert a 2D map's state coordinate to a serialized state number"""
        return np .array (twod [1 ]*self .unwrapped .grid_map_shape [0 ]+twod [0 ])

    def reset (self ,
    *,
    seed :Optional [int ]=None ,
    options :Optional [dict ]=None ,):
        """Rest the environment by initializaing the start state """
        super ().reset (seed =seed )
        if options is not None and 'random'in options .keys ():
            while True :
                self .start_state =random .randrange (0 ,self .unwrapped .grid_map_shape [0 ]*self .unwrapped .grid_map_shape [1 ]-1 )
                if self .start_state not in self .obstacles :
                    break 
        else :
            self .start_state =0 

        self .observation =self .start_state 
        return self .observation ,{'prob':1 }

    def render (self ):
        """Render the agent state"""
        pixel_size =20 
        img =np .zeros ([pixel_size *self .unwrapped .grid_map_shape [0 ],pixel_size *self .unwrapped .grid_map_shape [1 ],3 ],dtype =np .uint8 )
        for obstacle in self .obstacles :
          pos_x ,pos_y =self .serial_to_twod (obstacle )
          img [pixel_size *pos_x :pixel_size *(1 +pos_x ),pixel_size *pos_y :pixel_size *(1 +pos_y )]+=np .array ([255 ,0 ,0 ],dtype =np .uint8 )
        agent_state =self .serial_to_twod (self .observation )
        agent_target_state =self .serial_to_twod (self .terminal_state )
        img [pixel_size *agent_state [0 ]:pixel_size *(1 +agent_state [0 ]),pixel_size *agent_state [1 ]:pixel_size *(1 +agent_state [1 ])]+=np .array ([0 ,0 ,255 ],dtype =np .uint8 )
        img [pixel_size *agent_target_state [0 ]:pixel_size *(1 +agent_target_state [0 ]),pixel_size *agent_target_state [1 ]:pixel_size *(1 +agent_target_state [1 ])]+=np .array ([0 ,255 ,0 ],dtype =np .uint8 )
        if self .render_mode =='human':
          fig =plt .figure (0 )
          plt .clf ()
          plt .imshow (img ,cmap ='gray')
          fig .canvas .draw ()
          plt .pause (0.01 )
        if self .render_mode =='rgb_array':
          return img 
        return 

    def _close_env (self ):
        """Close the environment screen"""
        plt .close (1 )
        return 

class GridEnv (BaseGridEnv ):
    """
    A grid-world environment.
    """
    def transition_model (self ,state ,action ):
        """
        A transition model that return a list of probabilities of transitions
        to next states when the agent select 'action' at the 'state': T(s' | s,a)

        In our envrionemnt, if the state is in a goal,
        it will stay in its state at any action

        Parameters
        ----------
        state: integer
            a serialized state index
        action: integer
            action index

        Returns
        -------
        probs: numpy array with a length of {size of state space}
            probabilities of transition to the next_state ...
        """
        if not isinstance (state ,int ):
            state =state .item ()

        probs =np .zeros (self .observation_space .n )

        action_pos_dict ={0 :[0 ,0 ],1 :[-1 ,0 ],2 :[1 ,0 ],3 :[0 ,-1 ],4 :[0 ,1 ]}

        if state ==self .terminal_state :
            probs =np .zeros (self .observation_space .n )
            probs [state ]=1.0 
            return probs 
        num_actions =len (action_pos_dict )
        action_probs =[self .epsilon /(num_actions -1 )]*num_actions 
        action_probs [action ]=1.0 -self .epsilon 

        for new_action ,prob in enumerate (action_probs ):
            delta =action_pos_dict [new_action ]
            current_xy =self .serial_to_twod (state )
            new_xy =current_xy +np .array (delta )

            x ,y =new_xy 
            if (x <0 or x >=self .unwrapped .grid_map_shape [0 ]or y <0 or y >=self .unwrapped .grid_map_shape [1 ]):
                next_state =state 
            else :
                cand =int (self .twod_to_serial (new_xy ))
                if self .obstacles and cand in self .obstacles :
                    next_state =state 
                else :
                    next_state =cand 

            probs [next_state ]+=prob 

        return probs 

    def compute_reward (self ,state ,action ,next_state ):
        """
        A reward function that returns the total reward after selecting 'action'
        at the 'state'. In this environment,
        (a) If it reaches a goal state, it terminates returning a reward of +5
        (b) If it reaches a trap, it receives a penalty of -10
        (c) For any action, it add a step penalty of -0.1

        Parameters
        ----------
        state: integer
            a serialized state index
        action: integer
            action index
        next_state: integer
            a serialized state index

        Returns
        -------
        reward: float
            a total reward value
        """

        reward =0 

        reward =-0.1 

        if next_state ==self .terminal_state :
            reward +=5.0 

        if self .traps and next_state in self .traps :
            reward +=-10.0 

        return reward 

    def is_done (self ,state ,action ,next_state ):
        """
        Return True when the agent is in a terminal state or a trap,
        otherwise return False

        Parameters
        ----------
        state: integer
            a serialized state index
        action: integer
            action index
        next_state: integer
            a serialized state index

        Returns
        -------
        done: Bool
            the result of termination or collision
        """
        done =None 

        if next_state ==self .terminal_state or (self .traps and next_state in self .traps ):
            done =True 
        else :
            done =False 

        return done 

    def step (self ,action ):
        """
        A step function that applies the input action to the environment.

        Parameters
        ----------
        action: integer
            action index

        Returns
        -------
        observation: integer
            the outcome of the given action (i.e., next state)... s' ~ T(s'|s,a)
        reward: float
            the reward that would get for ... r(s, a, s')
        done: Bool
            the result signal of termination or collision
        truncated: Bool
            A boolean of whether a truncation condition is satisfied or not
        info: Dictionary
            Information dictionary containing miscellaneous information...
            (Do not need to implement info)

        """
        done =False 
        truncated =False 
        action =int (action )

        probs =self .transition_model (self .observation ,action )

        old_state =self .observation 
        next_state =int (np .random .choice (self .observation_space .n ,p =probs ))
        p =probs [next_state ]

        reward =self .compute_reward (old_state ,action ,next_state )
        done =self .is_done (old_state ,action ,next_state )

        self .observation =next_state 

        return (self .observation ,reward ,done ,truncated ,{"prob":p })

from gymnasium .envs .registration import register 

register (
id ='Gridworld-v0',
entry_point =GridEnv ,
max_episode_steps =30 ,
reward_threshold =100 ,
kwargs ={'epsilon':0.05 ,'size':[8 ,10 ],
'obstacle':[15 ,18 ,26 ,27 ,37 ,47 ,50 ,51 ,59 ,66 ,54 ],
'trap':[34 ,35 ,36 ,77 ]}
)
env =gym .make ("Gridworld-v0")

plot_grid (env )

def dummy_policy (state ):
    """A dummy random policy"""
    return np .random .choice ([1 ,2 ,3 ,4 ],1 ).item ()

trajs =collect_traj (env ,policy =dummy_policy ,num_episodes =100 )
plot_trajs (env ,trajs )

returns =eval_policy (env ,policy =dummy_policy ,num_episodes =100 )
plt .hist (returns ,bins =20 )
plt .xlabel ("Episode rewards")
plt .ylabel ("Counts")
plt .show ()

p =env .unwrapped .transition_model (env .unwrapped .twod_to_serial ([3 ,1 ]),2 )

print (p .reshape (env .unwrapped .grid_map_shape [::-1 ]).T )

p =env .unwrapped .transition_model (env .unwrapped .twod_to_serial ([0 ,6 ]),4 )

print (p .reshape (env .unwrapped .grid_map_shape [::-1 ]).T )

p =env .unwrapped .transition_model (env .unwrapped .twod_to_serial ([3 ,5 ]),4 )

print (p .reshape (env .unwrapped .grid_map_shape [::-1 ]).T )

saved_run ,returns =save_video_of_model ("Gridworld-v0",
model =dummy_policy ,
suffix ='_dummy',
num_episodes =10 )

class ValueIteration :
    """
    Value Iteration
    """
    def __init__ (self ,env ,theta =0.0001 ,discount_factor =0.9 ):
        """
        Initialize the VI class

        Parameters
        ----------
        env: object
            an OpenAI Gym compatible environment
        theta: float
            a max error (termination) threshold
        discount_factor: float
            discount factor (i.e., gamma)

        Returns
        -------
        probs: numpy array with a length of {size of state space}
            probabilities of transition to the next_state ...
        """
        self .env =env 
        self .V =np .zeros (env .observation_space .n )
        self .discount_factor =discount_factor 
        self .theta =theta 

    def value_iteration (self ):
        """
        A value iteration function. Until the error bound reaches the threshold
        (theta), The value table is updated by the Dynamic Programming
        (refer to the lecture)
        """
        errors =[]
        episode_rewards =[]
        count =0 
        while True :
            max_error =0 
            for state in range (env .observation_space .n ):

                Q =np .zeros (env .action_space .n )

                for action in range (env .action_space .n ):
                    trans_probs =env .unwrapped .transition_model (state ,action )
                    for next_state ,p in enumerate (trans_probs ):
                        if p >0 :
                            r =env .unwrapped .compute_reward (state ,action ,next_state )
                            Q [action ]+=p *(r +self .discount_factor *self .V [next_state ])
                v_new =np .max (Q )
                max_error =max (max_error ,abs (v_new -self .V [state ]))
                self .V [state ]=v_new 

            errors .append (max_error )
            mean_ep_reward =np .mean (eval_policy (env ,policy =self .get_action ,num_episodes =10 ))
            episode_rewards .append (mean_ep_reward )

            if count %10 ==0 :
                print ("Episode reward: {}".format (mean_ep_reward ))
            count +=1 

            if max_error <self .theta :
                break 
        return episode_rewards ,errors 

    def get_action (self ,state ):
        """
        Return the best action.
        HINT: how do you handle if there are multiple actions with the highest value?

        Parameters
        ----------
        state: integer
            a serialized state index

        Returns
        -------
        action: integer
            an action index
        """

        action =None 
        Q =np .zeros (env .action_space .n )

        for action in range (env .action_space .n ):
            trans_probs =env .unwrapped .transition_model (state ,action )
            for next_state ,p in enumerate (trans_probs ):
                if p >0 :
                    r =env .unwrapped .compute_reward (state ,action ,next_state )
                    Q [action ]+=p *(r +self .discount_factor *self .V [next_state ])

        best_acts =np .flatnonzero (Q ==Q .max ())
        action =int (np .random .choice (best_acts ))

        return action 

env =gym .make ("Gridworld-v0")
vi =ValueIteration (env ,discount_factor =0.9 )
ep_rews ,errors =vi .value_iteration ()

print (vi .V [:8 ])

saved_run ,_ =save_video_of_model ("Gridworld-v0",vi .get_action ,suffix ='_vi1')
render_value_map_with_action (env ,vi .V ,vi .get_action )
trajs =collect_traj (env ,policy =vi .get_action ,num_episodes =100 )
plot_trajs (env ,trajs )

def plot_reward_error_graphs (ep_rews ,errors ):
    ax =plt .subplot (2 ,1 ,1 )
    ax .plot (ep_rews )
    bx =plt .subplot (2 ,1 ,2 )
    bx .plot (errors )
    ax .set_ylabel ('Mean episode rewards')
    ax .ticklabel_format (axis ='x',style ='sci',scilimits =(0 ,0 ))
    bx .set_xlabel ('Number of iterations')
    bx .set_ylabel ('Errors')
    labels =[item .get_text ()for item in ax .get_xticklabels ()]
    empty_string_labels =['']*len (labels )
    ax .set_xticklabels (empty_string_labels )
    plt .show ()

plot_reward_error_graphs (ep_rews ,errors )

register (
id ='Gridworld-v0',
entry_point =GridEnv ,
max_episode_steps =30 ,
reward_threshold =100 ,
kwargs ={'epsilon':0.05 ,'size':[8 ,10 ],
'obstacle':[15 ,18 ,26 ,27 ,37 ,47 ,50 ,51 ,59 ,66 ,54 ],
'trap':[34 ,35 ,36 ,77 ]}
)
env =gym .make ("Gridworld-v0")

vi =ValueIteration (env ,discount_factor =0.9 )
ep_rews ,errors2 =vi .value_iteration ()

plt .figure ()
plt .plot (errors2 )
plt .xlabel ("Iteration")
plt .ylabel ("Max Bellman error")
plt .title ("Bellman error vs iteration (ε=0.05)")
plt .show ()

plt .figure ()
plt .plot (ep_rews )
plt .xlabel ("Iteration")
plt .ylabel ("Mean episodic return")
plt .title ("Mean episodic return vs iteration (ε=0.05)")
plt .show ()

Q_table =np .zeros ((env .observation_space .n ,env .action_space .n ))
for s in range (env .observation_space .n ):
    for a in range (env .action_space .n ):
        for s2 ,p in enumerate (env .unwrapped .transition_model (s ,a )):
            if p >0 :
                r =env .unwrapped .compute_reward (s ,a ,s2 )
                Q_table [s ,a ]+=p *(r +vi .discount_factor *vi .V [s2 ])

render_value_map_with_action (env ,Q_table ,policy =vi .get_action )

trajs =collect_traj (env ,policy =vi .get_action ,num_episodes =100 )
plot_trajs (env ,trajs )

register (
id ='Gridworld-v0',
entry_point =GridEnv ,
max_episode_steps =30 ,
reward_threshold =100 ,
kwargs ={'epsilon':0.4 ,'size':[8 ,10 ],
'obstacle':[15 ,18 ,26 ,27 ,37 ,47 ,50 ,51 ,59 ,66 ,54 ],
'trap':[34 ,35 ,36 ,77 ]}
)
env =gym .make ("Gridworld-v0")

vi =ValueIteration (env ,discount_factor =0.9 )
ep_rews ,errors =vi .value_iteration ()

plt .figure ()
plt .plot (errors )
plt .xlabel ("Iteration")
plt .ylabel ("Max Bellman error")
plt .title ("Bellman error vs iteration (ε=0.4)")
plt .show ()

plt .figure ()
plt .plot (ep_rews )
plt .xlabel ("Iteration")
plt .ylabel ("Mean episodic return")
plt .title ("Mean episodic return vs iteration (ε=0.4)")
plt .show ()

Q_table =np .zeros ((env .observation_space .n ,env .action_space .n ))
for s in range (env .observation_space .n ):
    for a in range (env .action_space .n ):
        for s2 ,p in enumerate (env .unwrapped .transition_model (s ,a )):
            if p >0 :
                r =env .unwrapped .compute_reward (s ,a ,s2 )
                Q_table [s ,a ]+=p *(r +vi .discount_factor *vi .V [s2 ])

render_value_map_with_action (env ,Q_table ,policy =vi .get_action )

trajs =collect_traj (env ,policy =vi .get_action ,num_episodes =100 )
plot_trajs (env ,trajs )
