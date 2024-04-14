# FQI Bird: Flappy Q-Learning and Reflex Agent for Flappy Bird

This project is a modified version of the original [FlapPyBird](https://sourabhv.github.io/FlapPyBird) game, which now includes basic AI agents designed to autonomously play the game. Modifications to the original game include the addition of a `get_state()` method alongside cool slow-motion and fast-forward features to enhance the fun of training and testing.

## Project Overview

The AI started with a QDN (Q-Network Deep Neural) agent but later transitioned to a combination of a Reflex-approved agent and a Q-Learning agent. This hybrid approach enables the system to identify `Killer Moves` when possible, and otherwise, it relies on the Q-Learning agent to derive the optimal game-playing policy.

### Reflex Agent

The Reflex agent employs a simple heuristic, primarily based on the positions of the bird and the pipes, as well as the bird’s velocity. It searches for potential `Killer Moves` to navigate the game's challenges effectively.

### Q-Learning Agent

Our Q-Learning agent experiments with various reward functions to optimize performance. After rigorous testing, including time-based rewards and distance-based rewards, the most effective metric has been the absolute distance from the bird to the closest pipe's middle point.

## Feedback and Contributions

If you have any ideas or need further information, please feel free to contact me via my LinkedIn profile: [Dor Pascal](https://www.linkedin.com/in/dor-pascal/).

## Acknowledgments

Credit for the original FlapPyBird game goes to its creator, [sourabhv](https://www.github.com/sourabhv). Please refer to the original game at the link provided above. This project is a derivative and does not claim original ownership of the underlying game concept.
