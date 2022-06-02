---
layout: post
title: Disaster at the Joadia Islands
cover-img: /assets/img/joadia-map-simple-top.png
thumbnail-img: /assets/img/joadia-map-simple.png
comments: false
---

## DDDDDDRRRRRAAAAAAAAAFT. This is a Draft. 


The Joadia Islands are a dark and sinister place. Or at least, it would seem so if you were to look at the fatality rate during recent tsunamis that occured on my laptop. Sorry folks, turns out that C130 wasn't coming for you after all. 

The task here was create an agent that would play the Joadia turn-based tabletop game, and play to win. For me, this was my first experience with reinforcement learning and I used to it explore imitation learning with the hope of replicating some of the success that transfer learning has had in other machine learning fields. I figure that I had so many ideas for how to improve the model, I couldn't wait around for it to train from scratch every time. 


Did it work? No it did not. 









## Having a peek at the machinery beneath Joadia

The first step for me was to gain some intuition on how the Joadia game worked, and understand the underlying data structures of the game. I needed to know how the state of the game was stored, what information was available to the game at any given time for it to make a decision, and how would that decision be communicated to the game. 

The natural way for me to go about this task was to create a visualisation of progress of a game so I could see how different agents played the game. My theory was that some agents would prioritise rescues over healing, or 

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






# This is a big title

sdfsf

## this is a small title test


asdfs

### Creating an agent that learns from a pro

The General Heurisitc already does a pretty reasonable job. That's not a bad place to begin. Let's consider a battle-hardened but stuck-in-their ways military commander named Heuristic. His rank is General. General Heuristic will be our guide and the starting place for our young and scrappy junior recruit, Lieutenant Learning. Someday he's going to have ideas of his own. But for now, our lieutenant must learn the ropes. 


Given that the PPO seems to reach a convergence within an hour of so of training, it would seem that pre-training might only reduce the training time, but not actually result in a score improvement. Firstly, this assumption should be tested, the RL algorithm has to test a huge space and becoming stuck in local minima seems a real positilbilyt that pre-training might overcome. But perhaps more importantly, a faster training cycle means faster feedback for me to try some other ideas, particuarly ones where i feel i can create better foundations to form a basis for the RL algnorithm to train from. 




For image recognition ML tasks, transfer learning has been a revelation. Pre-trained models that have already learned to classify some set of images can be fine-tuned to learn to classify something new. For a neural network for example, the process is to unfreeze the last couple of layers of the network, set the learning rate to be very low, and so gradually ease the model into its new task. I've used it for a couple of problems recently and it was like _magic_. Training times were fast and I needed a _fraction_ of the data I'd have needed if I were starting from scratch. 

This is where I spent the vast majority of my time. 

It seemed to me that transfer learning would provided both a mechanism for faster and more numerically stable training cycles. 


### Producing an actual agent

In the simplest submissions one could make for this competition, one would just tweak some parameters for training models from `Stable-Baseline3`. The final agent that I present here is precisely that. That's not a very satisfying outcome, but it's better than anything that our friend Lieutenant Learning could come up with! It's a DQN agent that learned for 5e7 timesteps and at a learning rate of 1e-4. 

In addition to probably not being very good _objectively_, I suspect that this agent isn't very generalisable.  If nothing else, I would have liked to have had the training games be using a random weighting for contributions from the 'RedHeuristic' and 'Random Legal' agents. It would have taken longer to train I suspect, but ultimately been less sensitive to that artificially imposed parameter. Training a red agent and then having a mix of agents that include both random legal, heuristics, and RL, would be even better. 


### Next steps and outstanding ideas

Six weeks ultimatly turned out to not nearly be enough for me to work through my ideas list. If I were to keep going though, there's a whole host of things I'd want to try. 

But if this was a project for my day job, the place I'd be starting would be to build a system _around_ this training process to speed up the iterative process. Being able to only do 1-2 experiments per day was a bottleneck for me. I suspect training on a GPU would speed a lot of these training runs up, but I'd first look to parallelization. Doing a hyperparameter search of a 5 parameters, each with 5 values can be done using a cloud VM for $1/hour. Being able to spawn training jobs on a cluster would be even better, but of course comes with a lot of overhead. In either case, parellized training would be a game changer for exploring the Joadia solution space.


Here's an example of a small scan of the learning rate which clearly demonstrates the sensitivity of the training process to these values, and the need for wide experimentation even for simple changes. 

![](/assets/img/learning_rate.png)
*Performance evolution of a DQN agent being trained with learning rates of 1e-4 (orange), 1e-3 (blue), and 1e-4 (gray). The gray curve continues to evolve after 32 hours of training on a Macbook Pro.*



* I didn't experiment with the simplified observation options, but it's not because I didn't like the idea. In fact, the more time I spent training models, the more I realised the task would be difficult for a NN to get to grips with becuase the important details for any given decision were contained in only a subset of the features. 

## Downloads


### Agent

### Code



This code is pretty hacky, but if you want to see how I went about all of this:
* Exploring how individual games run and the generation of the images for the viz gifs is in [run_single_game.py](https://drive.google.com/file/d/1bUuucYxkW_bZ2eLIR3S_LBXWi7H2prEY/view?usp=sharing
