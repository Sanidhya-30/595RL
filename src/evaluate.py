import os
import json
import matplotlib.pyplot as plt


def extract_results(base_dir, models):
    """Extract episodic rewards over iterations for multiple models."""
    results = {}
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"Directory for model {model} does not exist. Skipping.")
            continue
        
        model_results = {"rewards": [], "iterations": []}

        # Search for result.json in subdirectories
        for root, dirs, files in os.walk(model_dir):
            if "result.json" in files:
                result_path = os.path.join(root, "result.json")
                print(f"Reading results from {result_path}")
                
                with open(result_path, "r") as file:
                    for line in file:
                        data = json.loads(line.strip()) 
                
                # Extract episodic rewards and calculate iteration count
                episode_rewards = data.get("hist_stats", {}).get("episode_reward", [])
                iterations = list(range(1, len(episode_rewards) + 1))
                
                model_results["rewards"].extend(episode_rewards)
                model_results["iterations"].extend(iterations)
        
        if model_results["rewards"]:
            results[model] = model_results
        else:
            print(f"No valid results found for model {model}.")
    
    return results


def plot_results(results, output_file=None):
    """Plot episodic rewards over iterations for all models."""
    plt.figure(figsize=(10, 6))
    
    for model, data in results.items():
        plt.plot(data["iterations"], data["rewards"], label=model)
    
    plt.title("Episodic Reward Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Episodic Reward")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Base directory containing results for all models
    base_results_dir = "/home/lucifer/595RL/src/results"
    
    # Models to include in the plot
    models_to_compare = ["dqn", "ddpg", "maddpg", "ppo", "sac"]
    
    # Extract and plot results
    results_data = extract_results(base_results_dir, models_to_compare)
    plot_results(results_data, output_file="comparison_plot.png")
