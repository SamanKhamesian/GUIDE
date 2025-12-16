# Define glucose thresholds
class Threshold:
    HYPOGLYCEMIA = 70
    HYPERGLYCEMIA = 180


# Input data Configuration
class DataConfig:
    PATIENT_ID = "20"
    DATASET = "azt1d"


# GLIMMER Predictor Configuration
class PredictorConfig:
    ACTIVATION = 'relu'
    OPTIMIZER = 'adam'
    MA_WINDOW_SIZE = 200
    SPLIT_RATIO = 0.2
    TRAIN_WINDOW_SIZE = 72
    N_PREDICTION = 12
    BATCH_SIZE = 48
    EPOCHS = 30
    WEIGHTS = [1, 3.296363582, 2.382397706]


class EnvConfig:
    STATE_DIM = 6 * 12 * 7

    CARB_RANGE = (5, 50)
    INSULIN_RANGE = (2, 15)


class RewardShaping:
    IDEAL_CGM = 125
    WEIGHTS = [100, 7.0, 2.0]

    HYPO_HYPER_1_PENALTY = 100
    HYPO_HYPER_2_PENALTY = 200
    HYPO_HYPER_3_PENALTY = 300


# TD3_BC Model Configuration
class TD3Config:
    MAX_EPISODES = 20
    MAX_STEPS_PER_EPISODE = 24
    TRAINING_STEPS = 10_000
    TESTING_STEPS = 24
    BATCH_SIZE = 256
    NUM_TRAIN_INIT_STATE = 100
    NUM_TEST_INIT_STATE = 10

    GAMMA = 0.98
    TAU = 0.005
    ALPHA = 1.5
    NOISE_CLIP = 0.5
    POLICY_NOISE = 0.2
    POLICY_FREQ = 2
    ACTOR_LEARNING_RATE = 3e-4
    CRITIC_LEARNING_RATE = 1e-4

    CARB_RANGE = (5, 50)
    INSULIN_RANGE = (2, 15)


# Reward Function Shaping for PPO Model
class PPOConfig:
    MAX_EPOCHS = 20
    MAX_STEPS_PER_EPISODE = 24
    TESTING_STEPS = 24
    BATCH_SIZE = 256
    HIDDEN_SIZE = 256
    NUM_TRAIN_INIT_STATE = 100
    NUM_TEST_INIT_STATE = 10

    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_RATIO = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.001
    LEARNING_RATE = 3e-4

    CARB_RANGE = [5, 50]
    INSULIN_RANGE = [2.0, 15.0]
    TIME_INDEX_RANGE = [0, 11]


# Action Categories
class Action:
    NOTHING = 0
    EAT = 1
    INJECT = 2

# Sleep Hours
class SleepTime:
    WAKE_UP = 6
    BED_TIME = 0