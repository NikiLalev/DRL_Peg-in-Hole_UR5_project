import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_all_shapes_comparison(log_dir="./logs", agent="sac", reward_type="old", window_size=5000):
    """
    Plots a 2x2 grid showing training progress for all 4 shapes.
    """
    
    shapes = ["circle", "square", "triangle", "hexagon"]
    milestones = [50000, 100000, 150000, 200000]
    
    # Setup figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharey=True)
    fig.suptitle(f"Policy: {agent.upper()} | {reward_type.upper()} Reward Strategy", fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    found_any = False

    for idx, shape in enumerate(shapes):
        ax = axes_flat[idx]
        
        # Find file
        search_pattern = f"monitor_{agent}_{shape}_{reward_type}_"
        files = [f for f in os.listdir(log_dir) if search_pattern in f and f.endswith(".csv")]
        files.sort()
        
        if not files:
            ax.set_title(f"Shape: {shape.capitalize()} (No Data)")
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, "Log file not found", 
                    ha='center', va='center', color='gray', transform=ax.transAxes)
            continue

        filename = files[-1]
        filepath = os.path.join(log_dir, filename)
        
        try:
            df = pd.read_csv(filepath, skiprows=1)
            
            if df.empty:
                ax.text(0.5, 0.5, "Log is empty", ha='center', va='center', transform=ax.transAxes)
                continue

            found_any = True

            # Process Data
            raw_rewards = np.repeat(df['r'].values, df['l'].values.astype(int))
            timesteps = np.arange(len(raw_rewards))
            smoothed = pd.Series(raw_rewards).rolling(window=window_size, min_periods=1).mean()
            # Find max and final
            max_raw_val = raw_rewards.max()
            final_step = timesteps[-1]
            final_reward = smoothed.iloc[-1]
            
            # Initialize stat lines with the Header
            stats_lines = ["Rewards:"]
            stats_lines.append(f"Max:    {max_raw_val:.0f}")
            
            # Find values at milestones
            milestone_points = []
            for ms in milestones:
                if ms < len(smoothed):
                    val = smoothed.iloc[ms]
                    if not np.isnan(val):
                        milestone_points.append((ms, val))
                        stats_lines.append(f"@{ms//1000}k:   {val:.1f}")
            
            # Add Final
            stats_lines.append(f"Final:    {final_reward:.1f}")
            
            # Actual plot
            ax.plot(timesteps, raw_rewards, color='lightblue', alpha=0.5, linewidth=0.5, label='Raw')
            ax.plot(timesteps, smoothed, color='#0044cc', linewidth=2, label=f'Avg ({window_size})')
            
            # milestones - orange dots
            if milestone_points:
                ms_x, ms_y = zip(*milestone_points)
                ax.scatter(ms_x, ms_y, color='orange', s=60, zorder=5, edgecolors='black', label='Milestones')
            
            # final - green dot
            ax.scatter(final_step, final_reward, color='green', s=60, zorder=5, edgecolors='black', label='Final')

            # max - red star
            max_raw_idx = np.argmax(raw_rewards)
            ax.scatter(max_raw_idx, max_raw_val, color='red', marker='*', s=100, zorder=6, edgecolors='black', label='Max    ')

            # Stats Box
            stats_text = "\n".join(stats_lines)
            ax.text(0.03, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
            
            # Formatting
            ax.set_title(f"Shape: {shape.capitalize()}", fontsize=12)
            ax.set_xlabel("Timesteps")
            if idx % 2 == 0: ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Legend
            ax.legend(loc='lower right', fontsize='small', framealpha=0.9)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # save image at end
    save_filename = f"comparison_{agent}_{reward_type}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {os.path.abspath(save_filename)}")
    
    if not found_any:
        print(f"Warning: No log files found matching keys.")
        
        
if __name__ == "__main__":
    plot_all_shapes_comparison(agent="sac", reward_type="old", window_size=5000)
    plot_all_shapes_comparison(agent="ppo", reward_type="old", window_size=5000)