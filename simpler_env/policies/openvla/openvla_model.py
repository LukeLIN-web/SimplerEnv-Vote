from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
from simpler_env.main_inference import GenerateConfig
try:
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_lora_model,
    )
    from prismatic.models.action_heads import L1RegressionActionHead
except ImportError as e:
    print("efficient VLA is not correctly imported.")
    print(e)
import re
from simpler_env.utils.action.action_ensemble import ActionEnsembler, AdaptiveEnsembler, AvgEnsembler, voteEnsembler
from experiments.robot.openvla_utils import (
    model_is_on_hf_hub,
    update_auto_map,
    check_model_logic_mismatch,
)
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

class OpenVLAInference:
    def __init__(
        self,
        saved_model_path: str = "openvla/openvla-7b",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg: GenerateConfig = None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            # self.sticky_gripper_num_repeat = 15
            self.sticky_gripper_num_repeat = 10 #设成10 , 和cogact一样
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        
        if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
            # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

            # Update config.json and sync model files
            update_auto_map(cfg.pretrained_checkpoint)
            check_model_logic_mismatch(cfg.pretrained_checkpoint)

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            # "openvla/openvla-7b",
            saved_model_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()
        

        self.image_size = image_size
        self.action_scale = action_scale
        # self.horizon = horizon # 好像没用到
        self.pred_action_horizon = pred_action_horizon
        # self.exec_horizon = exec_horizon #  好像没用到

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0# 似乎没有用到

        if cfg.ensembler == "temp":
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, cfg.action_ensemble_temp
            )
        elif cfg.ensembler == "adapt":
            self.action_ensembler = AdaptiveEnsembler(
                self.pred_action_horizon, adaptive_ensemble_alpha=0.1 # same with cogact paper
            )
        else:
            self.action_ensembler = None
        print(f"*** action_ensembler: {self.action_ensembler} ***")

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None


    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)

        image: Image.Image = Image.fromarray(image)
        prompt = task_description

        # predict action (7-dof; un-normalize for bridgev2)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        # raw_actions,_ = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None] 
        raw_actions,_ = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False) # for openvla-opt
        # print(f"*** raw actions {raw_actions} ***")
        if self.action_ensembler != None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        } 

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)



def initialize_model(cfg: GenerateConfig):
    model = get_lora_model(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

    return model, action_head, processor

class LisaOpenVLAInference(OpenVLAInference):
    def __init__(
        self,
        saved_model_path: str = None, #没有用到 
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = -999,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg: GenerateConfig = None,
    ) -> None:

        if hasattr(cfg, 'pretrained_checkpoint') and cfg.pretrained_checkpoint:
            checkpoint_path = cfg.pretrained_checkpoint
            match = list(re.finditer(r'(\d+)act', checkpoint_path))
            if match:
                chunksize = int(match[-1].group(1))
                cfg.num_actions_chunk = chunksize
                self.pred_action_horizon = chunksize
                # self.pred_action_horizon = 2
                print(f"*** Extracted chunksize: {chunksize} from checkpoint path ***")
                print(f"*** cfg.num_actions_chunk: {cfg.num_actions_chunk}, cfg.num_actions_per_token: {cfg.num_actions_per_token} ***")
            else:
                raise ValueError("No action horizon found in checkpoint path")

            match_apt = re.search(r'(\d+)apt', checkpoint_path)
            if match_apt:
                apt = int(match_apt.group(1))
                if hasattr(cfg, 'num_actions_per_token'):
                    cfg.num_actions_per_token = apt
                print(f"*** Extracted actions_per_token: {apt} from checkpoint path ***")
            else:
                raise ValueError("No actions per token found in checkpoint path")

            print(f"*** cfg.num_actions_chunk: {cfg.num_actions_chunk}, cfg.num_actions_per_token: {cfg.num_actions_per_token} ***")
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            # self.sticky_gripper_num_repeat = 15 #设成10 , 和cogact一样, 不过似乎没啥用. 
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for  models. The other datasets can be found in the huggingface config.json file."
            )
        # if action_ensemble:
        if cfg.ensembler  != None:
            print("Action ensemble is enabled!")
        else:
            print("Action ensemble is disabled !!")

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        self.vla, self.action_head, self.processor = initialize_model(cfg)
        if cfg.mode == "mul":
            self.vla.predict_action = self.vla.mul_predict_action
        elif cfg.mode == "dit":
            self.vla.predict_action = self.vla.diffusion_predict_action
        else:
            raise ValueError(f"Invalid mode: {cfg.mode}")

        self.cfg = cfg  

        self.image_size = image_size
        self.action_scale = action_scale
        # self.horizon = horizon # 用在image history ,好像没啥用. 
        print(f"*** pred_action_horizon: {self.pred_action_horizon} ***")

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        if cfg.ensembler == "temp":
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, cfg.action_ensemble_temp
            )
        elif cfg.ensembler == "adapt":
            self.action_ensembler = AdaptiveEnsembler(
                self.pred_action_horizon, adaptive_ensemble_alpha=0.1 # same with cogact paper
            )
        elif cfg.ensembler == "avg":
            self.action_ensembler = AvgEnsembler(
                self.pred_action_horizon
            )
        elif cfg.ensembler == "vote":
            self.action_ensembler = voteEnsembler(
                self.pred_action_horizon, init_threshold=cfg.threshold
            )
        else:
            self.action_ensembler = None
        print(f"*** action_ensembler: {self.action_ensembler} ***")

        self.task = None
        self.task_description = None
        self.num_image_history = 0


    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description) # 不对, 应该每个episode也要换

        assert image.dtype == np.uint8
        image = self._resize_image(image)

        image: Image.Image = Image.fromarray(image)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        # predict action (7-dof; un-normalize for bridgev2)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        raw_actions, _ = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False, action_head=self.action_head, cfg=self.cfg)
        assert raw_actions.shape[0] == self.cfg.num_actions_chunk, f"Expected {self.cfg.num_actions_chunk} actions, got {raw_actions.shape[0]}"

        if self.action_ensembler != None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
            # 粘性动作模式下，夹爪动作会重复执行相同的动作多次（sticky_gripper_num_repeat次，对Google Robot设为15）
            # 这种机制确保夹爪动作能够完全执行，而不是每一步都变化
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat: 
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action# 强制覆盖, 保证夹爪动作能够完全执行, 而不是每一步都变化

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            

        action["terminate_episode"] = np.array([0.0])
        # print(f"*** action: {action} ***")
        # import ipdb; ipdb.set_trace()

        return raw_action, action

class LisaOpenVLAInferenceDebug(LisaOpenVLAInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("*** Initialized LisaOpenVLAInferenceDebug: Horizon visualization enabled ***")

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action_dict: dict; raw policy action output for the chosen/ensembled action
            action_processed: dict; processed action to be sent to the maniskill2 environment
            predicted_actions_horizon: Optional[np.ndarray]; full predicted action horizon (e.g., shape (H, 7))
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description) 

        assert image.dtype == np.uint8
        # 和父类 LisaOpenVLAInference.step 保持一致，先调整图像大小
        resized_image_np = self._resize_image(image) 

        pil_image: Image.Image = Image.fromarray(resized_image_np)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        
        inputs = self.processor(prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
        
        # 这是模型预测出的完整动作序列 (horizon)
        predicted_actions_horizon, _ = self.vla.predict_action(
            **inputs, 
            unnorm_key=self.unnorm_key, 
            do_sample=False, 
            action_head=self.action_head, 
            cfg=self.cfg
        )
        assert predicted_actions_horizon.shape[0] == self.cfg.num_actions_chunk, \
            f"Expected {self.cfg.num_actions_chunk} actions, got {predicted_actions_horizon.shape[0]}"

        # 确定用于创建 raw_action_dict 的动作来源
        # 如果有集成器，则使用集成后的动作；否则，使用预测序列中的第一个动作
        actions_for_dict_creation_source = predicted_actions_horizon
        if self.action_ensembler is not None:
            ensembled_action = self.action_ensembler.ensemble_action(predicted_actions_horizon) 
            actions_for_dict_creation_source = ensembled_action[None] 
        
        # 根据选定的动作来源创建 raw_action_dict
        raw_action_dict = {
            "world_vector": np.array(actions_for_dict_creation_source[0, :3]),
            "rotation_delta": np.array(actions_for_dict_creation_source[0, 3:6]),
            "open_gripper": np.array(actions_for_dict_creation_source[0, 6:7]),
        }

        # 处理 raw_action_dict 以获得发送到环境的 action_processed
        action_processed = {}
        action_processed["world_vector"] = raw_action_dict["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action_dict["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action_processed["rot_axangle"] = action_rotation_axangle * self.action_scale

        # 夹爪动作处理逻辑 (与父类一致)
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action_dict["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat: 
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action_processed["gripper"] = relative_gripper_action
        elif self.policy_setup == "widowx_bridge":
            action_processed["gripper"] = 2.0 * (raw_action_dict["open_gripper"] > 0.5) - 1.0

        action_processed["terminate_episode"] = np.array([0.0])

        # 返回选择的原始动作、处理后的动作以及完整的预测动作序列
        return raw_action_dict, action_processed, predicted_actions_horizon

    def visualize_current_horizon(
        self,
        predicted_horizon_actions_array: np.ndarray, 
        current_image: np.ndarray, 
        timestep: int,
        save_dir: str,
    ) -> None:
        """Visualizes the predicted action horizon at a single timestep."""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"horizon_step_{timestep:04d}.png")

        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        
        if predicted_horizon_actions_array.shape[1] != len(ACTION_DIM_LABELS):
            print(f"Warning: Action dimension mismatch in visualize_current_horizon. Expected {len(ACTION_DIM_LABELS)}, got {predicted_horizon_actions_array.shape[1]}.")
            # Robust error handling or label adjustment logic could be added here
            pass

        num_horizon_steps = predicted_horizon_actions_array.shape[0]
        
        display_image = self._resize_image(current_image.copy())

        plt.rcParams.update({"font.size": 8}) 
        # It's good practice to ensure unicode minus is handled correctly, even for English.
        plt.rcParams['axes.unicode_minus'] = False 
        fig, axs = plt.subplot_mosaic(
            [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS], 
            constrained_layout=True 
        )
        fig.set_size_inches([15, 6]) 

        fig.suptitle(f"Predicted Action Horizon at Timestep {timestep} (H={num_horizon_steps})", fontsize=12)

        for action_dim_idx, action_label in enumerate(ACTION_DIM_LABELS):
            if action_dim_idx < predicted_horizon_actions_array.shape[1]:
                axs[action_label].plot(predicted_horizon_actions_array[:, action_dim_idx], marker='o', linestyle='-', markersize=4)
                axs[action_label].set_title(action_label, fontsize=9)
                axs[action_label].set_xlabel(f"Future Horizon Steps", fontsize=8)
                axs[action_label].tick_params(axis='both', which='major', labelsize=7)
                axs[action_label].grid(True, linestyle='--', alpha=0.7)
            else:
                if action_label in axs: 
                    fig.delaxes(axs[action_label])
        
        axs["image"].imshow(display_image)
        axs["image"].set_title(f"Current Observation (t={timestep})", fontsize=9)
        axs["image"].axis('off') 

        plt.savefig(save_path)
        plt.close(fig)