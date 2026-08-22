# Network Topology Optimization for Bearing Rigid Multi-Agent Systems via Deep Reinforcement Learning and Graph Neural Networks

Master's thesis, University of Padova. Noyan Erdin Kilic.

A team of robots that can only measure *directions* to one another (bearings) can recover its own shape only if the graph of who-measures-whom is rich
enough. Adding every possible measurement makes that trivial and is wasteful, each link costs
sensing, tracking and communication. This thesis asks which links to keep, and learns the answer
with a graph neural network trained by reinforcement learning.

**Status: work in progress.** Interfaces, configuration formats and model architectures change
frequently, and the repository carries superseded code from earlier experiments.

## Main References

Bearing rigidity theory, and the source of the rigidity matrix formulation used here:

1. The extended bearing rigidity matrix used here restricts degrees of freedom per node rather than
   per edge, which only matters once a formation mixes domains. The construction, and why it differs
   from the one in [2], are written up in
   [docs/dof_restriction_note.pdf](docs/dof_restriction_note.pdf).
2. G. Michieletto, A. Cenedese, and D. Zelazo, "A Unified Dissertation on Bearing Rigidity Theory,"
   *IEEE Transactions on Control of Network Systems*, vol. 8, no. 4, pp. 1624-1636, Dec. 2021.
   [doi:10.1109/TCNS.2021.3077712](https://doi.org/10.1109/TCNS.2021.3077712)
3. M. H. Trinh, Q. Van Tran, and H.-S. Ahn, "Minimal and Redundant Bearing Rigidity: Conditions and
   Applications," *IEEE Transactions on Automatic Control*, vol. 65, no. 10, pp. 4186-4200,
   Oct. 2020. [doi:10.1109/TAC.2019.2958563](https://doi.org/10.1109/TAC.2019.2958563)

Combinatorial structure of the objective:

4. L. A. Wolsey, "An analysis of the greedy algorithm for the submodular set covering problem,"
   *Combinatorica*, vol. 2, no. 4, pp. 385-393, 1982.
   [doi:10.1007/BF02579435](https://doi.org/10.1007/BF02579435)

Reinforcement learning over graph structure:

5. V.-A. Darvariu, S. Hailes, and M. Musolesi, "Graph Reinforcement Learning for Combinatorial
   Optimization: A Survey and Unifying Perspective," *Transactions on Machine Learning Research*,
   Aug. 2024. [arXiv:2404.06492](https://arxiv.org/abs/2404.06492)
6. V.-A. Darvariu, S. Hailes, and M. Musolesi, "Goal-directed graph construction using reinforcement
   learning," *Proceedings of the Royal Society A*, vol. 477, no. 2254, 2021.
   [doi:10.1098/rspa.2021.0168](https://doi.org/10.1098/rspa.2021.0168)

Architectures:

7. V. G. Satorras, E. Hoogeboom, and M. Welling, "E(n) Equivariant Graph Neural Networks,"
   *ICML*, 2021. [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
8. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How Powerful are Graph Neural Networks?," *ICLR*,
   2019. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)

A [draft presentation](resources/rigidity_rl_260807-1.pdf) covers the same material with figures.

## Problem and Methodology

### Research questions
1. Is there an optimal graph topology balancing sparsity against structural rigidity, and what
   characterizes it?
2. Can deep reinforcement learning construct such a topology, and what architecture and training
   procedure does that require?
3. Does a learned policy generalize to networks it was not trained on: different sizes, different
   agent domains, heterogeneous compositions?
4. Where does a learned policy earn its place against classical combinatorial heuristics, and where
   does it not?
5. What are the properties of resultant graphs, and why?

The longer-term motivation is a *distributed* protocol for maintaining rigid formations in swarms. The centralized formulation here is a deliberate first step.

A formation of `n` agents is modelled as a **directed graph**. Each node is an agent with a pose
and a *domain* fixing which degrees of freedom it actually has (`R^2`, `R^3`, `R^2xS^1`, `R^3xS^1`, `SE(3)`). For example, a planar ground robot cannot leave its plane, and a quadrotor with a fixed-axis gimbal can rotate about one axis only. A directed edge `i -> j` means agent `i` measures the bearing to agent `j`, in `i`'s own frame. The relation is not symmetric: `i` seeing `j` does not imply `j` sees `i`.

A framework is **infinitesimally bearing rigid** when those measurements pin down the formation's
shape, leaving only the motions that no bearing can detect (global translation, global uniform scaling, and, where every agent carries its own frame, global rotation). A framework is rigid (IBR) when the rigidity matrix `B` attains the rank of the complete graph on the same poses.

> Given `n` agents at given poses, with given domains, choose the directed edge set so that the
> framework is infinitesimally bearing rigid, using as few edges as possible, and with preference
> to robustness against measurement noise.

The task is cast as a Markov decision process over edge sets. A state is a set of agent poses plus
the current graph, an action edits one directed edge, and the reward is the improvement in a scalar
objective `phi` that rewards rigidity and charges for each edge. A graph neural network encodes
the network into node embeddings, and an action head turns those into Q-values (DQN) or logits
(PPO). Invalid actions are masked inside the model.

## Current state

On heterogeneous networks of 10 agents spanning all five domains at once, a trained DQN
policy reaches 17.05 edges on average against a proven lower bound of 17, rigid on every instance of a frozen
20-instance benchmark and minimal on 95% of them. Greedy hill-climbing on the same objective and
the same instances reaches 80%, and a 20-restart constructive-greedy oracle reaches 50%. The policy
gets there with roughly 270x fewer rigidity-matrix evaluations than the oracle.

Transfer degrades with agent complexity. Without retraining the policy runs at 5, 8 and 16
agents. At 8 agents in homogeneous `R^3` it ties both classical baselines, reaching 50% minimal
where a random policy reaches 0%. At 16 it ties the weaker one and loses to greedy (23.20 edges and
0% minimal against 22.65 and 45%). The one configuration where it beats both classical methods is
the heterogeneous mixture it trained on. Across homogeneous domains at 8 agents,
however, transfer tracks the degrees of freedom each agent carries. It matches the baselines at 3
DOF per agent (`R^3`, `R^2xS^1`) and fails at 4 and 6 (`R^3xS^1` 10% minimal against 85-100% for the
baselines, `SE(3)` 25% against 100%). On `SE(3)` it scores below a uniform random policy on the
objective while remaining rigid everywhere, so the failure is over-density rather than
infeasibility.

Channel-wise ablation, run in three independent modes, shows that
the policy solves the problem from graph structure alone and reads no geometry at all. Destroying
the bearings, the agent coordinates and the null-space channels costs it nothing in any mode. That
is the correct response to an objective that contains no geometric term.

The objective has since been extended past the combinatorial rank. It can now charge for the
rigidity margin as well as for edges, weighted so that the margin is worth a stated number of edges.
On greedy hill-climbing, which needs no training, turning it on raises the margin of the resulting
networks by 2x to 20x at an unchanged edge count.

## Evaluation

![Baseline comparison table](resources/baselines-table.png)
![Run trajectories](resources/baselines-trajectories.png)
![Outcome across networks](resources/baselines-summary.png)
![Final, best and mean outcome per method](resources/baselines-outcomes.png)

## Documentation

| File | Contents |
|---|---|
| [THEORY.md](THEORY.md) | the mathematics: rigidity matrix, rank, null space, objective |
| [DESIGN_NOTES.md](DESIGN_NOTES.md) | why the implementation is the way it is |
| [ROADMAP.md](ROADMAP.md) | what is planned next |
| [docs/](docs/) | note on the heterogeneous rigidity matrix, with verification scripts |

## Running the code

Requires Python 3.12, an NVIDIA GPU, and [`uv`](https://docs.astral.sh/uv/).

```bash
./setup.sh          # installs uv if needed, creates .venv, installs dependencies
```

```bash
uv run environment.py 8 "R^3"          # generate an environment configuration
uv run train_dqn.py <env_name> <run_name>
uv run baselines.py <env_name> --model <run_name>    # against random, greedy, optimal
tensorboard --logdir runs
```

## Repository layout

```
rigidity.py         bearing rigidity matrix, rigidity tests, derived quantities
network.py          Agent and Network, graph features
environment.py      the gymnasium environment and its dispatchers
scenario.py         random and file-backed scenario generation
policy/             GNN backbones and one model per (backbone x action space)
train_dqn.py        training (train_ppo.py for PPO)
baselines.py        evaluation and comparison against baselines
benchmark.py        frozen evaluation instances
manifest.py         run manifests, archived sources and provenance
tests/              invariant tests
tools/              scripts worth keeping, verifications, ablations, measurements
docs/               notes
```

Directories produced by runs (`environments/`, `models/`, `runs/`, `train/`) are not tracked.

## License

MIT, see [LICENSE](LICENSE).
