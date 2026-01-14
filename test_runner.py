import os
import time
import argparse
import pandas as pd
import pybullet as p
from stable_baselines3 import PPO, SAC
from rlenv import PegInHoleGymEnv

def test_rl(checkpoint_path, agent_name, shape='circle', reward='old', num_episodes=50, output_csv=None, max_steps_limit=600):
    """
    Test a specific RL checkpoint.
    """
    print(f"Testing model: {checkpoint_path}")
    print(f"Config: Agent={agent_name}, Shape={shape}, Reward={reward}")

    # 1. Setup Environment
    try:
        env = PegInHoleGymEnv(shape_type=shape, reward_typ=reward, render_mode='GUI') 
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # 2. Load Model
    agent_name = agent_name.lower()
    if agent_name == "ppo":
        model_class = PPO
    elif agent_name == "sac":
        model_class = SAC
    else:
        env.close()
        raise ValueError(f"Unsupported agent: {agent_name}")

    try:
        model = model_class.load(checkpoint_path, env=env)
    except Exception as e:
        env.close()
        print(f"Failed to load model: {e}")
        return

    results = []

    # 3. Evaluation Loop
    try:
        for i in range(num_episodes):
            print(f"--- Starting Episode {i+1}/{num_episodes} ---")
            
            obs, info = env.reset()
            done = False
            steps = 0
            outcome = "unknown"
            
            # Watchdog start time
            start_time = time.time()

            while not done:
                # 60-second watchdog
                if time.time() - start_time > 60:
                    truncated = True
                    outcome = "timeout_watchdog"
                    done = True
                    break

                if env.render_mode == 'GUI':
                    time.sleep(0.001)

                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                steps += 1
                
                if steps >= max_steps_limit:
                    truncated = True
                    outcome = "timeout_forced"

                if terminated or truncated:
                    done = True
                    
                    if outcome in ["timeout_watchdog", "timeout_forced"]:
                        pass
                    elif info.get("insertion_success", False):
                        outcome = "success"
                    elif truncated:
                        outcome = "timeout"
                    else:
                        # Collision diagnosis
                        try:
                            raw_env = env.unwrapped
                            pts_table = p.getContactPoints(raw_env.robot_id, raw_env.table_id)
                            pts_hole = p.getContactPoints(raw_env.robot_id, raw_env.hole_id)

                            if len(pts_hole) > 0:
                                outcome = "collision_box" 
                            elif len(pts_table) > 0:
                                outcome = "collision_table" 
                            else:
                                outcome = "collision_other"
                        except:
                            outcome = "collision_unknown"

            print(f"Episode {i+1}: Steps={steps}, Outcome={outcome}")
            
            results.append({
                "episode": i + 1,
                "agent": agent_name,
                "shape": shape,
                "reward_type": reward,
                "steps": steps,
                "success": 1 if outcome == "success" else 0,
                "outcome": outcome
            })

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    
    # 4. Save Results immediately
    df = pd.DataFrame(results)
    if not df.empty:
        if output_csv:
            mode = 'a' if os.path.exists(output_csv) else 'w'
            header = not os.path.exists(output_csv)
            df.to_csv(output_csv, mode=mode, header=header, index=False)
            print(f"Results saved to {output_csv}")
        
        success_rate = (df['success'].sum() / len(df)) * 100
        print(f"Success Rate: {success_rate:.2f}%")
    else:
        print("No results collected.")

    # 5. Cleanup
    try:
        env.close()
        print("Environment closed.")
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL Test Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .zip model")
    parser.add_argument("--agent", type=str, required=True, choices=['sac', 'ppo'])
    parser.add_argument("--shape", type=str, default='circle', choices=['circle', 'square', 'triangle', 'hexagon'])
    parser.add_argument("--reward", type=str, default='old')
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default="results.csv")

    args = parser.parse_args()

    test_rl(
        checkpoint_path=args.checkpoint,
        agent_name=args.agent,
        shape=args.shape,
        reward=args.reward,
        num_episodes=args.episodes,
        output_csv=args.output
    )