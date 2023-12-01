# From ttenv to my work

Origin repo:https://github.com/coco66/ttenv

ttenv is just an environment for RL, but it doesn't provide the algorithm framework to solve it.(In fact,it solves by tf1 in other paper)

I want to build a **Reinforcement Learning framework** with the **stable baseline 3 (SB3)** and **use ttenv** to train the agent for active target tracking. To get training more easy!

the following is the target of the repositories:

### Train test

- [x] train the agent in ENV1,2,3 using DQN(without map inf)
- [x] train the agent in ENV5 using DQN(with map inf)

​	(it can work,but is traing now) :smiley: 

### Build a more flexible framework using custom tools

- [x] according to the SB3, add hyperparameters in model training
- [x] build own callback in training process
- [x] build custom policy network

 	(using "cnnpolicy" to transfer map inf and agent state)

The next target is to use more real-world simulator to explore reinforcement learning in AUV agent. :smiley_cat:

### Citing

```bibtex
@misc{ttenv,
    author = {Heejin Jeong, Brent Schlotfeldt, Hamed Hassani, Manfred Morari, Daniel D. Lee, and George J. Pappas},
    title = {Target tracking environments for Reinforcement Learning},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/coco66/ttenv.git}},
}
