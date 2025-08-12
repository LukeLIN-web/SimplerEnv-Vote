import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.evaluation.maniskill2_evaluator_debug import maniskill2_evaluatordebug
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    base_vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # Whether to include proprio state in input
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # effvla parameters
    mode: str = "mul"
    action_head_name: str = "new"
    num_actions_chunk: int = -999
    num_actions_per_token:int = -999
    action_ensemble_temp: float = -0.8
    ensembler: str = "vote" # "adapt" or "temporal" or None
    num_blocks: int = 4
    hidden_dim: int = 4096
    cfg_scale: float = 1.5
    num_ddim_steps: int = 10
    threshold: float = 0.5                           # Threshold parameter for voteEnsembler
    # fmt: on


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    cfg = GenerateConfig(pretrained_checkpoint=args.ckpt_path, ensembler=args.ensembler, threshold=args.threshold) #变成了传入的args.ensembler
    print(cfg.ensembler)
    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "openvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        model = OpenVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            cfg=cfg,
        )
    elif args.policy_model == "cogact":
        from simpler_env.policies.sim_cogact import CogACTInference
        assert args.ckpt_path is not None
        model = CogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type='DiT-B',
            cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
        )
    elif args.policy_model == "spatialvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
        model = SpatialVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "lisa":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import LisaOpenVLAInference
        model = LisaOpenVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            cfg=cfg
        )
    elif args.policy_model == "debug":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import LisaOpenVLAInferenceDebug
        model = LisaOpenVLAInferenceDebug(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            cfg=cfg
        )
    elif args.policy_model == "spatialvlalisa":
        assert args.ckpt_path is not None
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLALisaInference
        model = SpatialVLALisaInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            cfg=cfg
        )
    else:
        raise NotImplementedError()

    if args.policy_model == "debug":
        success_arr = maniskill2_evaluatordebug(model, args)
    else:
        success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
