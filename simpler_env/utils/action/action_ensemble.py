from collections import deque

import numpy as np


class ActionEnsembler:
    def __init__(self, pred_action_horizon, action_ensemble_temp=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_ensemble_temp = action_ensemble_temp
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        # if temp > 0, more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.action_ensemble_temp * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
  
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action

class AvgEnsembler:
    def __init__(self, pred_action_horizon):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        weights = np.ones(num_actions)
        weights = weights / weights.sum()
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return cur_action


class voteEnsembler:
    def __init__(self, pred_action_horizon, init_threshold=0.5):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.init_threshold = init_threshold
        
        self.current_threshold = init_threshold

    def reset(self):
        self.action_history.clear()
        self.current_threshold = self.init_threshold

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  

        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
        
        mask = cos_similarity > self.current_threshold
        if np.sum(mask) >= len(cos_similarity) // 2:
            mask = mask
        else:
            mask = ~mask
        
        masked_weights = np.ones(num_actions) * mask

        
        masked_weights = masked_weights / masked_weights.sum()
        cur_action = np.sum(masked_weights[:, None] * curr_act_preds, axis=0)
        return cur_action
