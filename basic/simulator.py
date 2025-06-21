import numpy as np

from config import Action, TargetModel, PredictorConfig, KnowledgeBase
from predictor import Predictor
from preprocess import DataController


class Simulator:
    def __init__(self, patient_id):
        self.data = DataController(patient_id, TargetModel.GLUCOSE)
        self.predictor = Predictor()

    def train(self):
        self.predictor.train(self.data.X_train_seq_,
                             self.data.y_train_seq_,
                             self.data.X_val_seq_,
                             self.data.y_val_seq_,
                             epochs=PredictorConfig.EPOCHS,
                             batch_size=PredictorConfig.BATCH_SIZE)

    def predict_next_cgm(self):
        return self.predictor.predict(self.data.X_test[:, :, 1:])[0]

    def get_full_current_window(self):
        return self.data.X_test[0]

    def apply_action_to_inputs(self, full_current_window, action_type):
        # Step 1: Inverse transform to real-world values
        full_real = self.data.get_inverse_transform(full_current_window)  # shape: (72, 7)

        # --> TODO: Currently, we inject the action effect at t=0 only; for realism, consider distributed insulin absorption curves
        # Step 2: Inject into existing data (accumulate at t=0)
        if action_type == Action.EAT:
            full_real[-12:, 2][0] += np.random.uniform(*KnowledgeBase.CARB_RANGE)
        elif action_type == Action.INJECT:
            full_real[-12:, 3][0] += np.random.uniform(*KnowledgeBase.INSULIN_RANGE)

        # Step 3: Re-transform and commit
        full_scaled = self.data.get_transform(full_real)
        self.data.X_test[0] = full_scaled

        print("\nAction applied and added to existing input.")

    def commit_next_input(self, predicted_cgm):
        self.data.commit_shift(predicted_cgm)

    def reset(self):
        print("\nResetting simulator...")
        self.data = DataController(self.data.patient_id, TargetModel.GLUCOSE)