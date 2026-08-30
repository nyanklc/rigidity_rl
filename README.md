# Network Topology Optimization for Bearing Rigid Multi-Agent Systems via Deep Reinforcement Learning and Graph Neural Networks

Master's thesis, University of Padova. Noyan Erdin Kilic.

A team of robots that can only measure *directions* to one another (bearings) can recover its own shape only if the graph of who-measures-whom is rich
enough. Adding every possible measurement makes that trivial and is wasteful, each link costs
sensing, tracking and communication. This thesis asks which links to keep, and learns the answer
with a graph neural network trained by reinforcement learning.

**Status: work in progress.**

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

[Status presentations](resources/) are used in meetings with the thesis supervisors.

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

One of the main focuses of this work is on generalizability across networks. A key goal is a single policy that generalizes across network sizes and
heterogeneous agent domains. The longer-term motivation is a *distributed* protocol for maintaining rigid formations in swarms.

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

## Example evaluation outputs

![Baseline comparison table](resources/baselines-table.png)
![Run trajectories](resources/baselines-trajectories.png)
![Outcome across networks](resources/baselines-summary.png)
![Final, best and mean outcome per method](resources/baselines-outcomes.png)

## Documentation

| File | Contents |
|---|---|
| [Bearing Rigidity Matrix Formulation](docs/dof_restriction_note.pdf) | Note on the heterogeneous rigidity matrix |
| [THEORY.md](THEORY.md) | Background on rigidity theory and how it is used in the project |
| [DESIGN_NOTES.md](DESIGN_NOTES.md) | implementation details |

## Running the code

Requires Python 3.12, an NVIDIA GPU, and [`uv`](https://docs.astral.sh/uv/).

```bash
./setup.sh          # installs uv if needed, creates .venv, installs dependencies
```

```bash
uv run environment.py 8 "R^3"
uv run train_dqn.py <env_name> <run_name>
uv run evaluation.py <env_name> --model <run_name>
tensorboard --logdir runs
```

## Repository layout

```
rigidity.py         bearing rigidity matrix, rigidity tests, derived quantities
estimation.py       shape recovery from noisy bearings, and the error it leaves
network.py          Agent and Network, graph features
scenario.py         random and file-backed scenario generation
util.py             geometry helpers
environment.py      the gymnasium environment and its dispatchers
policy/             GNN backbones and one model per (backbone x action space)
train_dqn.py        training (train_ppo.py for PPO)
probe.py            periodic deterministic evaluation during training
manifest.py         run manifests, archived sources and provenance
agent_loader.py     rebuilds a trained agent from its manifest
evaluation.py       THE results script: every metric, table and figure, in one run
report.py           the tables, CSVs and figures an evaluation run writes
benchmark.py        frozen evaluation instances
ablation.py         which observation channels a policy actually uses
inference.py        roll out a trained model
manual.py           interactive GUI for editing a graph by hand
benchmarks/         the frozen instance sets, tracked on purpose
tests/              invariant tests
tools/              scripts worth keeping, verifications, ablations, measurements
docs/               notes
resources/          figures used in this README
```

`evaluation.py` is the script to run for results. One invocation scores every method on the same
instances and writes the table, the per-episode and per-step CSVs, and every figure: trajectories,
outcomes, the summary, the comparison table, per-episode detail, measured error under bearing
noise, predicted against measured error, per-agent uncertainty ellipses, the softest deformation
mode, where the error comes from, how much the choice of repair matters, and -- with `--model` --
how the policy's own edits rank among the edits it could have made.

Directories produced by runs (`environments/`, `scenarios/`, `models/`, `runs/`, `train/`,
`runs_evaluation/`) are not tracked.

## License

MIT, see [LICENSE](LICENSE).
