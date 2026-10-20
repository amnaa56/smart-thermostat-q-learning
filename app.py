
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Thermostat with Q-Learning",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Smart Room Thermostat with Q-Learning")
st.write(
    "Train an AI agent to balance room comfort and electricity use, "
    "then compare it with a normal thermostat."
)


class Room:
    def __init__(self):
        self.reset()

    def reset(self):
        self.temp = 30
        self.occupied = 1
        return self.state()

    def state(self):
        return (self.temp - 20, self.occupied)

    def step(self, action):
        if random.random() < 0.1:
            self.occupied = 1 - self.occupied

        if action == 1:
            self.temp -= 1
        elif random.random() < 0.5:
            self.temp += 1

        self.temp = max(20, min(34, self.temp))

        energy = 1.0 if action == 1 else 0.0
        discomfort = (
            abs(self.temp - 24) if self.occupied else 0.0
        )
        reward = -(energy + 0.5 * discomfort)

        return self.state(), reward, energy


def train_agent(episodes=500, steps_per_episode=200):
    q_table = np.zeros((15, 2, 2))

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0

    env = Room()
    training_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(steps_per_episode):
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = int(
                    np.argmax(q_table[state[0], state[1], :])
                )

            next_state, reward, energy = env.step(action)

            current_q = q_table[state[0], state[1], action]
            best_next_q = np.max(
                q_table[next_state[0], next_state[1], :]
            )

            q_table[state[0], state[1], action] = (
                current_q
                + learning_rate
                * (
                    reward
                    + discount_factor * best_next_q
                    - current_q
                )
            )

            state = next_state
            episode_reward += reward

        training_rewards.append(
            episode_reward / steps_per_episode
        )

        epsilon = max(0.05, epsilon * 0.99)

    return q_table, training_rewards


def rl_policy_factory(q_table):
    def policy(state, env):
        return int(
            np.argmax(q_table[state[0], state[1], :])
        )
    return policy


def normal_policy(state, env):
    return 1 if env.temp > 24 else 0


def evaluate_policy(policy, steps=2000):
    env = Room()
    state = env.reset()

    total_reward = 0.0
    total_energy = 0.0
    total_discomfort = 0.0
    ac_on_steps = 0

    for _ in range(steps):
        action = policy(state, env)
        next_state, reward, energy = env.step(action)

        total_reward += reward
        total_energy += energy

        if env.occupied:
            total_discomfort += abs(env.temp - 24)

        ac_on_steps += int(action == 1)
        state = next_state

    return {
        "avg_reward": total_reward / steps,
        "energy": total_energy,
        "avg_discomfort": total_discomfort / steps,
        "ac_on_percent": 100 * ac_on_steps / steps
    }


with st.sidebar:
    st.header("Training Settings")

    episodes = st.slider(
        "Training Episodes",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )

    evaluation_steps = st.slider(
        "Evaluation Steps",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500
    )

    st.caption("More episodes require more training time.")


if st.button("🚀 Train Agent and Compare", type="primary"):

    with st.spinner("Training and evaluating thermostats..."):

        q_table, training_rewards = train_agent(
            episodes=episodes
        )

        rl_results = evaluate_policy(
            rl_policy_factory(q_table),
            steps=evaluation_steps
        )

        normal_results = evaluate_policy(
            normal_policy,
            steps=evaluation_steps
        )

    st.session_state["training_rewards"] = training_rewards
    st.session_state["rl_results"] = rl_results
    st.session_state["normal_results"] = normal_results


if "rl_results" in st.session_state:

    rl = st.session_state["rl_results"]
    normal = st.session_state["normal_results"]

    st.subheader("📊 Evaluation Results")

    st.caption(
        "Higher average reward (closer to zero) is better. "
        "Energy values are simulation units, not real kWh."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Q-Learning Thermostat")
        st.metric("Average Reward", f'{rl["avg_reward"]:.3f}')
        st.metric("Total Energy Units", f'{rl["energy"]:.0f}')
        st.metric("Average Discomfort", f'{rl["avg_discomfort"]:.3f}')
        st.metric("AC On", f'{rl["ac_on_percent"]:.1f}%')

    with col2:
        st.markdown("### Normal Thermostat")
        st.metric("Average Reward", f'{normal["avg_reward"]:.3f}')
        st.metric("Total Energy Units", f'{normal["energy"]:.0f}')
        st.metric("Average Discomfort", f'{normal["avg_discomfort"]:.3f}')
        st.metric("AC On", f'{normal["ac_on_percent"]:.1f}%')

    st.subheader("Controller Comparison")

    chart_data = {
        "Controller": ["Q-Learning", "Normal"],
        "Average Reward": [
            rl["avg_reward"],
            normal["avg_reward"]
        ],
        "Total Energy Units": [
            rl["energy"],
            normal["energy"]
        ],
        "Average Discomfort": [
            rl["avg_discomfort"],
            normal["avg_discomfort"]
        ]
    }

    st.dataframe(
        chart_data,
        use_container_width=True,
        hide_index=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        chart_data["Controller"],
        chart_data["Average Reward"]
    )
    axes[0].set_title("Average Reward (Higher Is Better)")
    axes[0].set_ylabel("Reward per Step")

    axes[1].bar(
        chart_data["Controller"],
        chart_data["Total Energy Units"]
    )
    axes[1].set_title("Total Energy Use (Lower Is Better)")
    axes[1].set_ylabel("Simulation Energy Units")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📈 Learning Curve")

    curve_fig, curve_ax = plt.subplots(figsize=(10, 4))

    curve_ax.plot(
        range(
            1,
            len(st.session_state["training_rewards"]) + 1
        ),
        st.session_state["training_rewards"]
    )

    curve_ax.set_xlabel("Episode")
    curve_ax.set_ylabel("Average Reward")
    curve_ax.set_title("Q-Learning Training Progress")

    curve_fig.tight_layout()
    st.pyplot(curve_fig)
    plt.close(curve_fig)

    st.info(
        "This is an educational simulation. Results can vary, "
        "and Q-Learning is not guaranteed to outperform the baseline. "
        "Real buildings require realistic temperature models and "
        "safety controls."
    )

else:
    st.info(
        "Choose your settings in the sidebar and click "
        "'Train Agent and Compare'."
    )

    st.subheader("How It Works")

    st.markdown(
        "1. The simulation tracks room temperature and occupancy.\n"
        "2. The agent decides whether to turn the AC on or off.\n"
        "3. The reward penalizes energy use and occupied-room discomfort.\n"
        "4. The trained agent is compared with a fixed-rule thermostat."
    )
