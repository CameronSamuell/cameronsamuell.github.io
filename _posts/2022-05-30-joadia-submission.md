---
layout: post
title: Disaster at the Joadia Islands
cover-img: /assets/img/joadia-map-simple-top.png
thumbnail-img: /assets/img/joadia-map-simple.png
comments: false
---


The Joadia Islands are a dark and sinister place. Or at least, it would seem so if you were to look at the fatality rate during recent tsunamis that occured on my laptop. Sorry folks, turns out that C130 wasn't coming for you. 

The task here was create an agent that would play the Joadia turn-based tabletop game, and play to win. For me, this was my first experience with reinforcement learning and I used to it explore the idea of imitation learning with the hope of replicating some of the success that transfer learning has had in other machine learning fields. I figure that I had so many ideas for how to improve the model, I couldn't wait around for it to train from scratch every time. 


Did it work? No it did not. 









## Visualisation 

The first step for me was to gain some intuition on how the Joadia game worked, and understand the underlying data structures of the game. I needed to know how the state of the game was stored, what information was available to the game at any given time for it to make a decision, and how would that decision be communicated to the game. 

The natural way for me to go about this task was to create a visualisation of progress of a game so I could see how different agents played the game. My theory was that some agents would prioritise rescues over healing, or 

The best-performing example model was 'General Heuristic', who we'll spend a lot more time with later. 

One of the most interesting parts of this is the difference between the blue agent's view of the world and the real view. Not fully realising that their views were different was something I stumbled 




<p align="center">
  <img src="/assets/img/general_heuristic.gif" alt="drawing" width="600" />
</p>


sadfjlskfj


<p align="center">
  <img src="/assets/img/random_legal.gif" alt="drawing" width="600" />
</p>







##

### Transfer Learning

For image recognition ML tasks, transfer learning has been a revelation. Pre-trained models that have already learned to classify some set of images can be fine-tuned to learn to classify something new. For a neural network for example, the process is to unfreeze the last couple of layers of the network, set the learning rate to be very low, and so gradually ease the model into its new task. I've used it for a couple of problems recently and it was like _magic_. Training times were fast and I needed a _fraction_ of the data I'd have needed if I were starting from scratch. 



### Next steps and outstanding ideas

Six weeks ultimatly turned out to not nearly be enough for me to work through my ideas list. If I were to keep going though, there's a whole host of things I'd want to try. 




But if this was a project for my day job, the place I'd be starting would be to build a system _around_ this training process to speed up the iterative process. Being able to only do 1-2 experiments per day was a bottleneck for me. I suspect training on a GPU would speed a lot of these training runs up, but I'd first look to parallelization. Doing a hyperparameter search of a 5 parameters, each with 5 values can be done using a cloud VM for $1/hour. Being able to spawn training jobs on a cluster would be even better, but of course comes with a lot of overhead. In either case, parellized training would be a game changer for exploring the Joadia solution space.


 
Here's an example of a small scan of the learning rate which clearly demonstrates the sensitivity of the training process to these values, and the need for wide experimentation even for simple changes. 

![](/assets/img/learning_rate.png)
*Performance evolution of a DQN agent being trained with learning rates of 1e-4 (orange), 1e-3 (blue), and 1e-4 (gray). The gray curve continues to evolve after 32 hours of training on a Macbook Pro.*




### Code

This code is pretty hacky, but if you want to see how I went about this:
* Exploring how individual games run and the generation of the images for the viz gifs is in [run_single_game.py](https://drive.google.com/file/d/1bUuucYxkW_bZ2eLIR3S_LBXWi7H2prEY/view?usp=sharing). 
