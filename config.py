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


# TD3_BC Model Configuration
class TD3Config:
    MAX_EPISODES = 10
    MAX_STEPS_PER_EPISODE = 24
    TRAINING_STEPS = 10_000
    TESTING_STEPS = 24
    BATCH_SIZE = 256
    NUM_TRAIN_INIT_STATE = 50
    NUM_TEST_INIT_STATE = 10

    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 2.5
    NOISE_CLIP = 0.5
    POLICY_NOISE = 0.2
    POLICY_FREQ = 2
    LEARNING_RATE = 3e-4

    STATE_SIZE = 6 * 12 * 8 # 8 features, and 6-hour data for each
    CARB_RANGE = (5, 100)
    INSULIN_RANGE = (1, 10)


# Reward Function Shaping for TD3-BC Model
class TD3RewardShaping:
    IDEAL_CGM = 125
    WEIGHTS = [100, 11.0, 1.09018]

    DO_NOTHING_IN_SLEEP = 50
    DO_NOTHING_BONUS = 20
    REPEATED_ACTION_PENALTY = 30
    EAT_COUNT_REWARD = 60

    EARLY_MEAL_PENALTY = 50
    EARLY_INJECTION_PENALTY = 50
    GOOD_MEAL_TIMING_BONUS = 50

    HYPO_HYPER_1_PENALTY = 50
    HYPO_HYPER_2_PENALTY = 100
    HYPO_HYPER_3_PENALTY = 200


# Reward Function Shaping for PPO Model
class PPOConfig:
    MAX_STEPS_PER_EPISODE = 24
    TRAINING_EPOCHS = 5
    TESTING_STEPS = 24
    BATCH_SIZE = 256
    HIDDEN_SIZE = 256
    NUM_TRAIN_INIT_STATE = 50
    NUM_TEST_INIT_STATE = 10

    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_RATIO = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.001
    LEARNING_RATE = 3e-4

    STATE_SIZE = 6 * 12 * 8
    CARB_RANGE = [5, 100]
    CARB_STEP = 1
    INSULIN_RANGE = [0.1, 10.0]
    INSULIN_STEP = 0.1
    TIME_INDEX_RANGE = [0, 11]
    TIME_STEP = 1
    N_ACTION_TYPE = 3


# Reward Function Shaping for PPO Model
class PPORewardShaping:
    IDEAL_CGM = 125
    WEIGHTS = [100, 11.0, 1.09018]

    DO_NOTHING_IN_SLEEP = 50
    DO_NOTHING_BONUS = 20
    REPEATED_ACTION_PENALTY = 30
    EAT_COUNT_REWARD = 60

    EARLY_MEAL_PENALTY = 40
    EARLY_INJECTION_PENALTY = 40
    GOOD_MEAL_TIMING_BONUS = 60

    HYPO_HYPER_1_PENALTY = 50
    HYPO_HYPER_2_PENALTY = 100
    HYPO_HYPER_3_PENALTY = 200


# Action Categories
class Action:
    NOTHING = 0
    EAT = 1
    INJECT = 2

# Sleep Hours
class SleepTime:
    WAKE_UP = 6
    BED_TIME = 0