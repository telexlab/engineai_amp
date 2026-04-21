from __future__ import annotations

import torch



from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from engineai_lab.tasks.velocity import mdp
from .flat_env_cfg import PM01FlatEnvCfg
from engineai_lab.robots.pm01 import PM01_CFG, PM_WAIST_DFS_JOINT_NAMES, PM01_DFS_JOINT_ORDER_ASSET_CFG

def dummy_history_term(env):
    # zero-out waist joint observations to avoid AMP mismatch between robot variants
    joint_pos = mdp.joint_pos(env).clone()
    joint_vel = mdp.joint_vel(env).clone()
    waist_joint_ids = env.scene["robot"].find_joints(".*WAIST_YAW.*", preserve_order=True)[0]
    if len(waist_joint_ids) > 0:
        waist_joint_ids = torch.as_tensor(waist_joint_ids, device=joint_pos.device)
        joint_pos[:, waist_joint_ids] = 0.0
        joint_vel[:, waist_joint_ids] = 0.0
    lin_vel = mdp.robot_base_lin_vel_b(env)

    return torch.cat([joint_pos * 9, lin_vel * 7], dim=-1)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        actions = ObsTerm(func=mdp.last_action,
                          history_length=15)        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               noise=Unoise(n_min=-0.2, n_max=0.2),
                            history_length=15)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=15
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    params={"command_name": "base_velocity"})        
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        
        actions = ObsTerm(func=mdp.last_action,
                          history_length=15)        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               noise=Unoise(n_min=-0.2, n_max=0.2),
                            history_length=15)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=15
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    params={"command_name": "base_velocity"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class AMPCfg(ObsGroup):
        history = ObsTerm(func=dummy_history_term)
        def __post_init__(self):
            self.history_length = 5 # TODO: is the history from old to new or new to old?
        

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AMPCfg = AMPCfg()
    

@configclass
class PM01AMPFlatEnvCfg(PM01FlatEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
