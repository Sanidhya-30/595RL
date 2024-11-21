from pettingzoo.mpe import simple_tag_v3
import numpy as np

# Define a simple policy for the agents
def demo_policy(agent, observation, action_space):
    """
    A demo policy for the agents:
    - Predators (adversaries): Move randomly.
    - Prey (good agents): Move away from the closest adversary.
    """
    if "adversary" in agent:
        # Adversaries take random actions
        return action_space.sample()
    else:
        # Prey moves away from the closest adversary
        prey_position = observation[0:2]
        adversary_positions = [observation[i:i+2] for i in range(2, len(observation), 2)]
        # Find the closest adversary
        distances = [np.linalg.norm(prey_position - adv_pos) for adv_pos in adversary_positions]
        closest_adversary = adversary_positions[np.argmin(distances)]
        # Move away from the closest adversary
        move_direction = prey_position - closest_adversary
        normalized_move = move_direction / (np.linalg.norm(move_direction) + 1e-6)
        # Map direction to action (assumes 5 discrete actions: [no-op, up, down, left, right])
        action = np.argmax(normalized_move)
        return action

# Initialize the environment
parallel_env = simple_tag_v3.parallel_env(render_mode="human")
observations, infos = parallel_env.reset(seed=42)

# Main environment loop
while parallel_env.agents:
    # Generate actions using the demo policy
    actions = {
        agent: demo_policy(agent, observations[agent], parallel_env.action_space(agent))
        for agent in parallel_env.agents
    }
    # Step through the environment
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    
parallel_env.close()
