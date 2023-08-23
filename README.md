# EvoFed ~ Evolutionary Federated Learning

EvoFed optimize the client model through evolutionary process instead of backpropagation.

The optimization process can be recreated at the server by only transferring *fitness values* instead of model parameters,
saving communication bandwidth.

If we start with the same seed in all clients and server,
and upload *fitness values* every generation,
server can transfer *fitness mean* to clients instead of aggregated model.
This condition not only reduce communication cost, 
but also data distribution among clients become unrelated to model training.

-----

Tasks Accomplished:

| T   | Env    | Task         | ES Tested | ES Tuned | BP Tested | BP Tuned | FedAvg Tested | FedAvg Tuned |
|-----|--------|--------------|:---------:|:--------:|:---------:|:--------:|:-------------:|:------------:|
| SL  | Image  | MNIST        |  &check;  | &check;  |  &check;  | &check;  |    &check;    |   &check;    |
| SL  | Image  | FMNIST       |  &check;  | &check;  |  &check;  | &check;  |    &check;    |   &check;    |
| SL  | Image  | Cifar10      |  &check;  |          |  &check;  |          |    &check;    |              |
| RL  | gymnax | Cartpole     |  &check;  | &check;  |  &check;  | &check;  |    &check;    |   &check;    |
| RL  | gymnax | Acrobot      |  &check;  |          |  &check;  |          |    &check;    |              |
| RL  | gymnax | Pendulum     |  &check;  |          |  &check;  |          |    &check;    |              |
| RL  | gymnax | MountainCar  |  &check;  |          |  &check;  |          |    &check;    |              |
| RL  | gymnax | MountainCarC |  &check;  |          |  &check;  |          |    &check;    |              |
| RL  | brax   | ant          |  &check;  |          |  &check;  |          |    &check;    |              |
| RL  | brax   | halfcheetah  |  &check;  |          |           |          |               |              |
| RL  | brax   | hopper       |  &check;  |          |           |          |               |              |
| RL  | brax   | humanoid     |  &check;  |          |           |          |               |              |
| RL  | brax   | reacher      |  &check;  |          |           |          |               |              |
| RL  | brax   | walker2d     |  &check;  |          |           |          |               |              |
| RL  | brax   | fetch        |  &check;  |          |           |          |               |              |
| RL  | brax   | grasp        |  &check;  |          |           |          |               |              |
| RL  | brax   | ur5e         |  &check;  |          |           |          |               |              |



Results:

| T   | Env    | Task         | Accuracy | Target Accuracy | Communication for Target | Runtime for Target |
|-----|--------|--------------|:--------:|:---------------:|:------------------------:|:------------------:|
| SL  | Image  | MNIST        | &check;  |     &check;     |         &check;          |      &check;       |
| SL  | Image  | FMNIST       | &check;  |     &check;     |         &check;          |      &check;       |
| SL  | Image  | Cifar10      | &check;  |                 |         &check;          |                    |
| RL  | gymnax | Cartpole     | &check;  |     &check;     |         &check;          |      &check;       |
| RL  | gymnax | Acrobot      | &check;  |                 |         &check;          |                    |
| RL  | gymnax | Pendulum     | &check;  |                 |         &check;          |                    |
| RL  | gymnax | MountainCar  | &check;  |                 |         &check;          |                    |
| RL  | gymnax | MountainCarC | &check;  |                 |         &check;          |                    |
| RL  | brax   | ant          | &check;  |                 |         &check;          |                    |
| RL  | brax   | halfcheetah  | &check;  |                 |                          |                    |
| RL  | brax   | hopper       | &check;  |                 |                          |                    |
| RL  | brax   | humanoid     | &check;  |                 |                          |                    |
| RL  | brax   | reacher      | &check;  |                 |                          |                    |
| RL  | brax   | walker2d     | &check;  |                 |                          |                    |
| RL  | brax   | fetch        | &check;  |                 |                          |                    |
| RL  | brax   | grasp        | &check;  |                 |                          |                    |
| RL  | brax   | ur5e         | &check;  |                 |                          |                    |
