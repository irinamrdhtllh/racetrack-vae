import gymnasium as gym
import numpy as np

from PIL import Image


env = gym.make("racetrack-v0", render_mode="human")
env.configure(
    {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (256, 256),
            "stack_size": 1,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 4,
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
        "manual_control": True,
    }
)
observation, info = env.reset()

j = 1
for i in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    image = Image.fromarray(observation[0], mode="L")
    if i % 10 == 0:
        image.save(f"images/train/image_{j}.png")
        j += 1

    if terminated or truncated:
        # observation, info = env.reset()
        break

env.close()
