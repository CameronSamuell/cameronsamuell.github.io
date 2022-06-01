---
layout: post
title: Disaster at the Joadia Islands
cover-img: /assets/img/joadia-map-simple-top.png
thumbnail-img: /assets/img/joadia-map-simple.png
comments: false
---



### 

### 

### 


### Next steps and outstanding ideas

Six weeks ultimatly turned out to not nearly be enough for me to work through my ideas list. If I were to keep going though, there's a whole host of things I'd want to try. 


But if this was a project for my day job, the place I'd be focussing my time would be looking to find methods to speed up the iterative process. Being able to only do 1-2 experiments per day was a bottleneck for me. I suspect training on a GPU would speed a lot of these training runs up, but I'd first look to parallelization. Doing a hyperparameter search of a 5 parameters, each with 5 values can be done using a cloud VM for $1/hour and would be a game changer for exploring the Joadia solution space.

Here's an example of a small scan of the learning rate which clearly demonstrates the sensitivity of the training process to these values, and the need for wide experimentation even for simple changes. 

![](/assets/img/learning_rate.png)
*Training evolution of a DQN agent with learning rates of 1e-4 (orange), 1e-3 (blue), and 1e-4 (gray). The gray curve continues to evolve after 32 hours of training on a macbook pro.*
