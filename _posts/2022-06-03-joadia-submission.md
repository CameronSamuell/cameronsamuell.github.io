---
layout: post
title: Disaster at the Joadia Islands
cover-img: /assets/img/joadia-map-simple-top.png
thumbnail-img: /assets/img/joadia-map-simple.png
comments: false
---

The Joadia Islands are a dark and sinister place. Or at least, it would seem so if you were to look at the fatality rate during recent tsunamis that occurred on my laptop. Sorry folks. It turns out that C130 wasn't coming for you after all. 

The task here was to create an agent that would play the Joadia turn-based tabletop game and play to win. This was my first experience with reinforcement learning, and I used to it explore imitation learning with the hope of replicating some of the success that transfer learning has had in other machine learning fields. I figure that I had so many ideas for how to improve the model, I couldn't wait around for it to train from scratch every time. I wanted to start with a pre-trained model that I could finesse.  

Did it work? No, it did not. 

### Having a peek at the machinery beneath Joadia

Before getting to the actual agent-training, my first step was to try and gain some intuition for how the Joadia game mechanics worked and understand the underlying data structures. I needed to know how the state of the game was stored, what information was available to the game at any given time for it to make a decision, and how that decision would be communicated to the game. But I was also curious to see if I could glean some hints from looking at the spatial distribution of the various populations, or the trajectory of the rescues and deaths as a game progressed. 

To gather the information, I had a script that ran through a single iteration of the game while keeping track of all the status values for each territory at each turn. These values had to be remapped into a rectilinear coordinate for plotting which also gave me a nice way to plot a reference map of the terrain. I stumbled here by not considering the fog-of-war effect. The agent's view of the world is limited to the places it has visited. By including both the real data and the blue agent's view of the data in my visualisation, we have a chance to observe why the agent might have made a particular decision, but also _what was really happening_ in Joadia that the agent was oblivious to. 

Here's a General Heuristic, our best performing example agent, playing a game:

<p align="center">
  <img src="/assets/img/general_heuristic.gif" alt="drawing" width="600" />
</p>

 You can see the blue view expanding over the course of the game as well as the reduction in civilian population numbers at the island outskirts as people are healed and moved to the forward operating base. You can also see the rising rescues and deaths as the larger story unfolds turn-by-turn. 

Now, here is Random Legal playing a game:

<p align="center">
  <img src="/assets/img/random_legal.gif" alt="drawing" width="600" />
</p>

Unsurprisingly, making random decisions didn't work out well for Joadia's residents. Comparing the two cases though, you can see that trajectory of deaths is about the same. General Heuristic rescued a lot more people and that made a world of difference because the scoring weights rescues over deaths. That points to the potential for training agents on loss functions that relate more directly to either maximising rescues or minimising deaths rather than the score.  Not a lot can be gleaned from differences in the population distributions, but that might just be the granularity of the map.

Perhaps a game with more turns is better suited to this kind of visualisation, but even here I think it would be possible to see the difference in agents that prioritised particular outcomes, or had radically different spatial distributions of healthy or injured civilians, that might help understand the strengths and weaknesses of  agents. 

### Creating an agent that learns from a pro

For image recognition ML tasks, transfer learning has been a revelation and I wanted to explore how that translates to the RL world. For instance, pre-trained models that have already learned to classify some set of images can be fine-tuned to learn to classify something new. Faster training times and lower requirements on the size of the training data set. The idea was that for experimenting with RL training mechanisms, starting from a pre-trained base would provide a way to iterate quickly on ideas as compared to a cold start for every idea. Exploring what that would mean for training an agent to play Joadia is ultimately where I spent most of my time. 

For our purposes, let's consider a battle-hardened but stuck-in-their ways military commander named Heuristic. His rank is General. He does an okay job; makes some sensible decisions; saves lives. General Heuristic will be our guide and the starting place for our young and scrappy junior recruit, Lieutenant Learning. Someday he's going to have ideas of his own. But for now, our lieutenant must learn the ropes. 

I generated a training dataset by recording both the observations and associated actions of General Heuristic. I chose to run many full games and take the status at each turn rather than generate a random set of observations and associated actions. While it's possible there are areas of the observation space that General Heuristic would never encounter, it seemed less risky than generating random sets of observations. For instance, I wanted to make sure I included the fog-of-war effects that the visualisation
exercise had shown me were important to capture. The observation-action pairs had to be valid situations that would be obtained in a real game. The dataset was split up into an 80/20 train/test split and fed to a training routine that aimed to create an agent that would replicate every decision. 

While there are a few details in how the training routine was set up, the most critical decision in this process was deciding on a loss function. It wouldn't have been sufficient to just take the difference in observed and predicted actions; similar numerical ids of the territories do not correspond to similar decisions. Sending a unit to territory T9 isn't almost the same as sending the unit to territory T10. Instead, we need to compare the predicted action with the policy distribution across all actions. The extra layer of detail we capture here for example might be that General Heuristic was 60% sure that the unit be sent to T11, 23% sure that they be sent to territory T16, and 17 sure that they go to T23 given a particular observation. It's that detail that our lieutenant needed to learn.

On this measure Lieutenant Learning certainly improved - the performance gradually went up round-by-round for a PPO agent. But the system never got _good_. This is how Lieutenant Learning, at the completion of his education, compared to General Heuristic:

![](/assets/img/pretraining_score.png)
*The best pre-trained model scored -39.75 compared to scores of -44.91 for the Random Legal agent and +10.28 for General Heuristic.*

The improvement compared to a Random Legal agent suggests the pre-trained model learned _something_, but it was nowhere near the General's benchmark. To try and correct this, I did a fair amount of hyperparameter tuning including the optimizer, the number of epochs, the learning rate, the learning rate schedule, learning rate decay, and so on. Increasing the training dataset size made the biggest positive impact on the performance, but training became very slow as a result.

So, not great! I didn't get to the bottom of why this was. Did my training dataset not capture real games well enough? Not enough data or enough training time to learn from it? Was my loss function off? Was PPO a fundamentally bad choice? The base model choice is the next thing I would have swapped out, followed by finding a way to simplify the observation data to reduce the training scope (and therefore the amount of data/training time) to at least get a demonstration that the basics were sound and then increase scope and dataset size from there. 

### Producing an actual agent

In a simple submission one could make for this competition, you might just tweak some parameters for training models from `Stable-Baseline3`. The final agent that I present here is precisely that. That's not a very satisfying outcome, but it's better than anything that our friend Lieutenant Learning could come up with! It's a DQN agent that learned for 5e7 time-steps and at a learning rate of 1e-4. 

In addition to probably not being all that good _objectively_, I suspect that this agent isn't very generalisable.  If nothing else, I would have liked to have had the training games be using a random weighting for contributions from the 'Red Heuristic' and 'Random Legal' agents. It would have taken longer to train of course, but ultimately been less sensitive to that artificially imposed parameter. Training a red agent and then having a mix of agents that include both random legal, heuristics, and RL, would be even better. 

This was the best score achieved during training. It's not quite on par with General Heuristic, but close enough that I suspect that more hyperparameter tuning and adjustment/scaling of the observation and loss function would probably have gotten there. 

![](/assets/img/dqn_score.png)
*Best mid-training test score of 1.66 for a vanilla DQN agent.*

_Note_: `model.load` didn't function as expected for running `test.py` here. Potentially an unexpected consequence of my upgrading `stable_baselines3` from 1.0 to 1.5.0 (`gym` 0.21.0) to get access to `model.policy.get_distribution()` for the pre-training experiment. 

### Next steps and outstanding ideas

Six weeks ultimately turned out to not nearly be enough for me to work through my ideas list. If I were to keep going, there's a whole host of things I'd want to try. For example:
* I didn't experiment with the simplified observation options, but it's not because I didn't like the idea. In fact, the more time I spent training models and staring at the observation data, the more I realised the task would be difficult for a model to get to grips with because the important details for any given decision were contained in only a subset of the features. What I'd like to try is having an abstraction layer that would allow me to first train a model on a simplified picture of the game status, and then progressively give it a more detailed set of observations in an iterative training process.
* Changing and scaling the reward function is a natural step to take that I didn't get around to. This has the capacity to not only affect the performance, but also the training speed and numerical stability. 
* Having an agent learn how to play many _styles_ of Joadia game seems important to avoid overfitting. One thing I'd be curious to try though is to train a set of agents with various specialisations. For instance, an agent that focuses on maximising rescues, one that minimises deaths, one that prioritises exploration, one that aggressively responds to red agent action, and so on. These models are intentionally overfitted however they could be taken as an ensemble with an agent trained in assessing which of those models is best to act in any given scenario. 
* It probably goes without saying, but I've focussed on Stable Baselines models. Testing the waters with newer algorithms and a broader set of architectures would be super interesting. I'd want to exhaust simple approaches first though. 

But if this was a project for my day job, the place I'd be starting would be to build a system _around_ this training process to speed up the iterative process. Being able to only do 1-2 experiments per day was a bottleneck for me. I suspect training on a GPU would speed a lot of these training runs up, but I'd first look to parallelization. Doing a hyperparameter search of five parameters, each with five values, can be done using a cloud VM for less than $1/hour. Being able to spawn training jobs on a cluster with elastic size, or submit them to a serverless utility would be even better. But of course, this comes with a lot of overhead. In either case, parallelised training would be a game-changer for exploring the Joadia solution space.

Here's an example of a small scan of the learning rate which clearly demonstrates the sensitivity of the training process to these values, and the need for wide experimentation even for simple changes. 

![](/assets/img/learning_rate.png)
*Performance evolution of a DQN agent being trained with learning rates of 1e-4 (orange), 1e-3 (blue), and 1e-4 (grey). The grey curve continues to evolve after 32 hours of training on a MacBook Pro.*

Ultimately, I'd want to be testing a _ton_ of those small changes to make sure I'm not being fooled by local minima or bad hyperparameter choice which in turn would let me spend more time thinking about the big changes I could be making. 

### Finally...

Making Joadia available like this for a short challenge was probably a ton of work for someone/some people. I had a great time and learned all kinds of weird things. RL. SPOD. A2C. OPV. PPO. DQN. Mot Coy! 

So, thank you!

### Downloads

#### Agents
* The best lieutenant that Cameron's bootcamp supplied was [pretrained_student_2022-05-16.zip](/assets/downloads/pretrained_student_2022-05-16.zip).
* The best vanilla DQN agent was ['DQN_2022-06-02.zip'](/assets/downloads/DQN_2022-06-02.zip).

#### Code

This code is pretty hacky, but if you want to see how I went about all of this:
* Exploring how individual games run and the generation of the images for the gifs was done in [run_single_game.py](/assets/downloads/run_single_game.py)
* The training of an imitation model was achieved in [train_lieutenant.py](/assets/downloads/train_lieutenant.py) which calls functions defined in [pretrain_fun.py](/assets/downloads/pretrain_fun.py). 

