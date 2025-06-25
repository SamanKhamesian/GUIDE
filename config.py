# Define glucose thresholds
class Threshold:
    HYPOGLYCEMIA = 70
    HYPERGLYCEMIA = 180


# RL Agent Experiment Configuration
class RLConfig:
    MAX_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 72
    TRAINING_STEPS = 10000
    TESTING_STEPS = 72


# Custom LSTM Configuration
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


class TargetModel:
    BASAL = 'basal'
    BOLUS = 'bolus'
    GLUCOSE = 'glucose'
    CARBS = 'carbs'


# Knowledge Base for constraints
class KnowledgeBase:
    CARB_RANGE = (0, 50)  # Max 50g carbs
    INSULIN_RANGE = (0, 3)  # Max 3 units basal insulin


# Action Space
class Action:
    NOTHING = 0
    EAT = 1
    INJECT = 2


class SleepTime:
    WAKE_UP = 7
    BED_TIME = 23


class RewardFunction:
    IDEAL_CGM = 125
    WEIGHTS = [100, 11.0, 1.09018]

    DO_NOTHING_IN_SLEEP = 50
    DO_NOTHING_BONUS = 20
    REPEATED_ACTION_PENALTY = 30

    EARLY_MEAL_PENALTY = 50
    EARLY_INJECTION_PENALTY = 50
    GOOD_MEAL_TIMING_BONUS = 50

    HYPO_HYPER_1_PENALTY = 50
    HYPO_HYPER_2_PENALTY = 100
    HYPO_HYPER_3_PENALTY = 200