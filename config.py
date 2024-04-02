class Args:
    def __init__(self, 
                 seed: int = 1, 
                 torch_deterministic: bool = True, 
                 track: bool = False,
                 wandb_project_name: str = "cleanRL", 
                 wandb_entity: str = None, 
                 capture_video: bool = False,
                 state_dim: list = [4],  
                 action_dim: int = 2, 
                 action_high: float = 1,
                 buffer_size: int = int(1e5),           # 1e6
                 gamma: float = 0.99, 
                 tau: float = 0.005, 
                 batch_size: int = 5,                  # 256
                 learning_starts: int = int(5e3), 
                 alpha: float = 3e-4,
                 beta: float = 0.003, 
                 reward_scale: float = 2,
                 policy_frequency: int = 2, 
                 target_network_frequency: int = 1, 
                 noise_clip: float = 0.5,
                 auto_alpha: float = 0.2,
                 auto_tune: bool = True):
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.track = track
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.capture_video = capture_video
        # Ensure state_dim is initialized to a new list for each instance
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.action_high = action_high
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.beta = beta
        self.reward_scale = reward_scale
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.noise_clip = noise_clip
        self.auto_alpha = auto_alpha
        self.auto_tune = auto_tune
       

