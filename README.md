## A simple game to explore neural networks.

The game consists of a hero, enemies, houses, and mine.

If the hero defeats an enemy he gets some money. After he may replenish his hp in the houses.

Here is a simple example.

![init](https://github.com/overmesgit/majestyai/assets/2367946/f6e6cf9a-9b0b-4edd-aa7f-98a42ea42eda)

You can see enemies on the left side and houses, mine on the right.
We can skip a couple hundred turns to get the model some time for training.

![mid1](https://github.com/overmesgit/majestyai/assets/2367946/a4529ce0-4558-467b-b069-7c0c5c8fc957)

This is turn 239. On the right side, there are 2 enemies with 0.57/0.04 and 0.88/0.04 (health/money).
I omitted stats for other enemies because they have low rewards.
In this situation, it is more beneficial for a hero to attack the enemy with 0.57/0.04 because it has less hp but the same amount of reward - 0.04.
Next turn(space key):

![mid2](https://github.com/overmesgit/majestyai/assets/2367946/b19c692b-fe29-4f5f-b9b9-cc83dfd0e06c)

As you can see the hero successfully chose the right enemy to attack.

Here are a couple more turns.

![mid3](https://github.com/overmesgit/majestyai/assets/2367946/bfc20f6e-940d-4aa8-953d-50b79cb75cf8)
![mid4](https://github.com/overmesgit/majestyai/assets/2367946/733e670f-ba7c-4efa-b553-527bb7646861)

The hero always chooses the right move. 👍
