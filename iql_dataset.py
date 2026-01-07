import numpy as np
from stable_baselines3 import SAC

def export_npz(model_path, rb_path, out_path):
    model = SAC.load(model_path, env=None)
    model.load_replay_buffer(rb_path)

    rb = model.replay_buffer

    obs = rb.observations
    next_obs = rb.next_observations
    actions = rb.actions
    rewards = rb.rewards.reshape(-1)
    dones = rb.dones.reshape(-1).astype(np.float32)

    np.savez(
        out_path,
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminals=dones,
    )
    print("Saved:", out_path)

if __name__ == "__main__":
    export_npz(
        model_path="./checkpoints/sac/circle/sac_final_model.zip",
        rb_path="./checkpoints/sac/circle/final_replay_buffer.pkl",
        out_path="./checkpoints/sac/circle/peg_in_hole_iql_dataset.npz",
    )

