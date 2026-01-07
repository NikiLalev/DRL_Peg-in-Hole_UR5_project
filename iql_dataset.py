import numpy as np
from stable_baselines3 import SAC

def export_npz(model_path, rb_path, out_path):
    model = SAC.load(model_path, env=None)
    model.load_replay_buffer(rb_path)

    rb = model.replay_buffer
    
    obs = rb.observations["cam_image"]          # (N, 100, 100, 1), uint8
    next_obs = rb.next_observations["cam_image"]
    actions = rb.actions                        # (N, 3), float32
    rewards = rb.rewards.reshape(-1)            # (N,)
    dones = rb.dones.reshape(-1).astype("float32")

    np.savez(
        out_path,
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminals=dones,
    )
    print("Saved:", out_path)

    print("observations:", obs.shape)
    print("actions:", actions.shape)
    print("rewards:", rewards.shape)
    print("next_observations:", next_obs.shape)
    print("terminals:", dones.shape)


if __name__ == "__main__":
    export_npz(
        model_path="./checkpoints/sac/circle/sac_final_model.zip",
        rb_path="./checkpoints/sac/circle/final_replay_buffer.pkl",
        out_path="./checkpoints/sac/circle/peg_in_hole_iql_dataset.npz",
    )

