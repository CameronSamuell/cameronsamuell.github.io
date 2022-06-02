---
layout: post
title: Disaster at the Joadia Islands
cover-img: /assets/img/joadia-map-simple-top.png
thumbnail-img: /assets/img/joadia-map-simple.png
comments: false
---

# DDDDDDRRRRRAAAAAAAAAFT. This is a Draft. 


The Joadia Islands are a dark and sinister place. Or at least, it would seem so if you were to look at the fatality rate during recent tsunamis that occured on my laptop. Sorry folks, turns out that C130 wasn't coming for you after all. 

The task here was create an agent that would play the Joadia turn-based tabletop game, and play to win. For me, this was my first experience with reinforcement learning and I used to it explore imitation learning with the hope of replicating some of the success that transfer learning has had in other machine learning fields. I figure that I had so many ideas for how to improve the model, I couldn't wait around for it to train from scratch every time. I wanted to start with a pre-trained model that I could finesse.  


Did it work? No it did not. 



### Having a peek at the machinery beneath Joadia

The first step for me was to gain some intuition on how the Joadia game worked, and understand the underlying data structures of the game. I needed to know how the state of the game was stored, what information was available to the game at any given time for it to make a decision, and how would that decision be communicated to the game. But I was also curious to see if I could glean some hints from looking at the spatial distribution of the population, or the trajectory of the

The natural way for me to go about this task was to create a visualisation of progress of a game so I could see how different agents played the game. My theory was that some agents would prioritise rescues over healing, or 

This was done just by having a single play-through of the game where I kept track of all the status values for each territory as turns progressed.  


The best-performing example model was 'General Heuristic', who we'll spend a lot more time with later. 

One of the most interesting parts of this is the difference between the blue agent's view of the world and the real view. Not fully realising that their views were different was something I stumbled 

Perhaps a game with more turns is better suited to this kind of visualisation, but even here I think it would be possible to see the difference in agents that prioritised early vs late-game actions, or had radically different spatial distributions of healthy or injured civillians.  


Here's a General Heuristic game:

<p align="center">
  <img src="/assets/img/general_heuristic.gif" alt="drawing" width="600" />
</p>


Here's Random Legal playing a game. It doesn't work out well for Joadia's residents. Comparing the two cases though, you can see that trajectory of deaths is about the same. General Heuristic rescued a lot more people though and that made a big difference to the score, even though the score is weighted towards reducing the death count. That in itself points to the potential for training agents on loss funtions that relate more directly to either maximising rescues or minimising deaths rather than the score.  Not a lot can be gleaned for differences in the population distributions, but that might just be the granularity of the map at play. This could be certainly be fine-tuned for in


<p align="center">
  <img src="/assets/img/random_legal.gif" alt="drawing" width="600" />
</p>



### Creating an agent that learns from a pro

Let's consider a battle-hardened but stuck-in-their ways military commander named Heuristic. His rank is General. He does an okay job, makes some sensible decisions, saves lives. General Heuristic will be our guide and the starting place for our young and scrappy junior recruit, Lieutenant Learning. Some day he's going to have ideas of his own. But for now, our lieutenant must learn the ropes. 

For image recognition ML tasks, transfer learning has been a revelation. Pre-trained models that have already learned to classify some set of images can be fine-tuned to learn to classify something new. For a neural network for example, the process is to unfreeze the last couple of layers of the network, set the learning rate to be very low, and so gradually ease the model into its new task. I've used it for a couple of problems recently and it was like _magic_. Training times were fast and I needed a _fraction_ of the data I'd have needed if I were starting from scratch. 

Exploring how this idea translates to the RL space is where I spent the vast majority of my time. The idea was that for experimenting with RL training mechanisms, starting from a pre-trained base would provide a way to iterate quickly on ideas as compared to a cold start for every idea. 

It seemed to me that transfer learning would provided both a mechanism for faster and more numerically stable training cycles. 


I generated a training dataset by recording both the observations and actions of General Heuristic and outputted the dataset to file. I took care here, to run many full games and take the status at each turn rather than generate a random set of observations and associated actions. It was important that the game status (including the fog of war effects) be valid situations that might be obtained in a real game. The dataset was split up into a 80/20 train/test split and fed to a training routine that aimed to create an agent that would replicate every decision. 


The most critical decision in this process was deciding on a loss function. It wouldn't have been sufficient to just take the difference in observed and predicted action, similar numerical ids of the territories does not correspond similar decisions. Sending a unit to territory T9 isn't almost the same as sending the unit to territory T14 Instead we need to compare the prediction action with the policy distribution across all actions. We want our Lietenant to repliacate that given a particular observation (the state of the game and the unit to command), General Heuristic was 60% sure that the unit be sent to territory T11 and 23% sure that they be sent to territory T16. 

On this measure Lieutenant Learning certainly improved - the performance gradually went up round-by-round. But the system never got _good_.   

This is how Lietenant Learning, at the completion of his education, compared to General Heuristic:


![](/assets/img/pretraining_score.png)
*Best score achieved during training of a vanilla DQN agent.*


To try and correct this, I did a fair amount of hyperparameter tuning including the optimizer, the number of epochs, the learning rate , the learning rate scheduling, learning rate decay, and so on. The thing that made the biggest difference was raising the number of interactions in the expert dataset. Performance steadily increased as the training dataset grew, but the training time became miserably slow by the time I got to a 


So, not great! I didn't get to the bottom of why this was. Did my training dataset not capture real games well enough? Not enough data or enough training time to learn from it? Was my loss function off? Was PPO a fundamentally bad choice? The base model choice is the next thing I would have swapped out, followed by finding a way to simplify the observation data to reduce the training scope (and therefore amount of data/training time) to at least get a demonstration that the rest of the imitation learning setup was bug free and then increase scope dataset size from there. 

### Producing an actual agent

In the simplest submissions one could make for this competition, one would just tweak some parameters for training models from `Stable-Baseline3`. The final agent that I present here is precisely that. That's not a very satisfying outcome, but it's better than anything that our friend Lieutenant Learning could come up with! It's a DQN agent that learned for 5e7 timesteps and at a learning rate of 1e-4. 

In addition to probably not being very good _objectively_, I suspect that this agent isn't very generalisable.  If nothing else, I would have liked to have had the training games be using a random weighting for contributions from the 'RedHeuristic' and 'Random Legal' agents. It would have taken longer to train I suspect, but ultimately been less sensitive to that artificially imposed parameter. Training a red agent and then having a mix of agents that include both random legal, heuristics, and RL, would be even better. 

This was the best core achieved during training, not quite on par with General Heurisitc, but close enough that I suspect that more hyperparameter tuning, and adjustment/scaling of the observation and loss function would probably have gotten there. 


![](/assets/img/dqn_score.png)
*Best score achieved during training of a vanilla DQN agent.*


_Note_: `model.load` didn't function as expected for running `test.py` here. Potentially an unexpected consequence of my upgrading `stable_baselines3` from 1.0 to 1.5.0 (`gym` 0.21.0) to get access to `model.policy.get_distribution()` for the pre-training experiment. 

### Next steps and outstanding ideas

Six weeks ultimatly turned out to not nearly be enough for me to work through my ideas list. If I were to keep going though, there's a whole host of things I'd want to try. For example:
* I didn't experiment with the simplified observation options, but it's not because I didn't like the idea. In fact, the more time I spent training models, the more I realised the task would be difficult for a NN to get to grips with because the important details for any given decision were contained in only a subset of the features. What I'd like to try though is having an abstraction layer that would allow me to first train a model on a simplified picture of the game status, and the training it in future training stages that progressively give a more detailed set of observations. 
* Changing and scaling the reward function is a natural step to take that I didn't get around to. This has the capacity to not only affect the performance, but also the training speed and numerical stability. 
* Having an agent learn how to play many _styles_ of Joadia game seems important to avoid overfitting. One thing I'd be curious to try though is to training a set of agent with various specialisations. For instance an agent that focusses on maximising rescues, one that minimises 
* It probably goes without saying, but I've focussed on Stable Baselines models. Testing the waters with newer algorithms and a broader set of architecutes would wwould be super interesting. I'd want to exhaust simple approaches first though. 
* DQN instead of PPO for the pretraining run. 

But if this was a project for my day job, the place I'd be starting would be to build a system _around_ this training process to speed up the iterative process. Really flexing that engineering muscle. Being able to only do 1-2 experiments per day was a bottleneck for me. I suspect training on a GPU would speed a lot of these training runs up, but I'd first look to parallelization. Doing a hyperparameter search of a five parameters, each with five values can be done using a cloud VM for less than $1/hour. Being able to spawn training jobs on a cluster would be even better, but of course comes with a lot of overhead. In either case, parallelised training would be a game changer for exploring the Joadia solution space.

Here's an example of a small scan of the learning rate which clearly demonstrates the sensitivity of the training process to these values, and the need for wide experimentation even for simple changes. 

![](/assets/img/learning_rate.png)
*Performance evolution of a DQN agent being trained with learning rates of 1e-4 (orange), 1e-3 (blue), and 1e-4 (gray). The gray curve continues to evolve after 32 hours of training on a Macbook Pro.*

Ultimately, I'd want to be testing both a _ton_ of those small changes to makes sure I'm not being fooled by local minimal or bad hyper parameter choice which in turn would let me spend more time thinking about the big changes i could be making. 

### Downloads

#### Agent

#### Code

This code is pretty hacky, but if you want to see how I went about all of this:
* Exploring how individual games run and the generation of the images for the viz gifs is in [run_single_game.py](/assets/downloads/run_single_game.py)
* The training of an imitation model was achieved in [train_lieutenant.py](/assets/downloads/train_lieutenant.py) which calls functions defined in [pretrain_fun.py](/assets/downloads/pretrain_fun.py). 

### Finally...

Making Joadia available like this for a short challenge was probably a ton of work for someone(s). I had a great time, and learned all kinds of weird things. RL. SPOD. A2C. OPV. PPO. DQN. Mot Coy! 

So, thank you!
