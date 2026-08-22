"""Which model class serves a given (role, backbone, action space).

Replaces the if/elif chains that used to live in train_ppo.py and train_dqn.py.
Roles are skrl's model-dict keys, so build_models() output is passed straight to
the agent.
"""

import inspect

from .actor.AddEdgeDiscreteNoSelfLoops import *
from .actor.AddEdgeDiscreteNoSkipNoSelfLoops import *
from .actor.AddRemoveEdgeDiscreteNoSelfLoops import *
from .actor.AddRemoveEdgeMultiDiscrete import *
from .actor.AllEdges import *
from .actor.DecideOnEdge import *
from .actor.SelectNodesSequentially import *
from .actor.Equivariant_SelectNodesSequentially import *
from .actor.Equivariant_AddRemoveEdgeDiscreteNoSelfLoops import *
from .actor.GINE_AddRemoveEdgeDiscreteNoSelfLoops import *
from .actor.GINE_SelectNodesSequentially import *

from .critic.Default import *
from .critic.Selection import *
from .critic.Equivariant import *
from .critic.Equivariant_Selection import *
from .critic.GINE import *
from .critic.GINE_Selection import *

from .q_func.AddEdgeDiscreteNoSelfLoops import *
from .q_func.AddRemoveEdgeDiscreteNoSelfLoops import *
from .q_func.SelectNodesSequentially import *
from .q_func.GINE_SelectNodesSequentially import *
from .q_func.GINE_AddRemoveEdgeDiscreteNoSelfLoops import *
from .q_func.GINE_AddEdgeDiscreteNoSkipNoSelfLoops import *
from .q_func.Equivariant_SelectNodesSequentially import *
from .q_func.Equivariant_AddRemoveEdgeDiscreteNoSelfLoops import *
from .q_func.Equivariant_AddEdgeDiscreteNoSkipNoSelfLoops import *


BACKBONES = ("Equivariant", "GINE", "Default")

ALGORITHM_ROLES = {
    "PPO": ("policy", "value"),
    "DQN": ("q_network",),
    "DDQN": ("q_network",),
}

# (role, backbone, action_type) -> class.
# A (role, backbone, None) entry is the fallback for that role and backbone,
# which is how the critics cover every action space without a selection stage.
MODELS = {
    # ---- PPO actors
    ("policy", "Equivariant", "SelectNodesSequentially"):
        PPO_ActorModel_Equivariant_SelectNodesSequentially,
    ("policy", "Equivariant", "AddRemoveEdgeDiscreteNoSelfLoops"):
        PPO_ActorModel_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops,
    ("policy", "GINE", "SelectNodesSequentially"):
        PPO_ActorModel_GINE_SelectNodesSequentially,
    ("policy", "GINE", "AddRemoveEdgeDiscreteNoSelfLoops"):
        PPO_ActorModel_GINE_AddRemoveEdgeDiscreteNoSelfLoops,
    ("policy", "Default", "SelectNodesSequentially"):
        PPO_ActorModel_SelectNodesSequentially,
    ("policy", "Default", "AddRemoveEdgeDiscreteNoSelfLoops"):
        PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops,
    ("policy", "Default", "AddEdgeDiscreteNoSelfLoops"):
        PPO_ActorModel_AddEdgeDiscreteNoSelfLoops,
    ("policy", "Default", "AddEdgeDiscreteNoSkipNoSelfLoops"):
        PPO_ActorModel_AddEdgeDiscreteNoSkipNoSelfLoops,
    ("policy", "Default", "AddRemoveEdgeMultiDiscrete"):
        PPO_ActorModel_AddRemoveEdgeMultiDiscrete,
    ("policy", "Default", "AllEdges"):
        PPO_ActorModel_AllEdges,
    ("policy", "Default", "DecideOnEdge"):
        PPO_ActorModel_DecideOnEdge,

    # ---- PPO critics: one per backbone, plus a selection-aware variant
    ("value", "Equivariant", "SelectNodesSequentially"):
        PPO_CriticModel_Equivariant_Selection,
    ("value", "Equivariant", None):
        PPO_CriticModel_Equivariant,
    ("value", "GINE", "SelectNodesSequentially"):
        PPO_CriticModel_GINE_Selection,
    ("value", "GINE", None):
        PPO_CriticModel_GINE,
    ("value", "Default", "SelectNodesSequentially"):
        PPO_CriticModel_Selection,
    ("value", "Default", None):
        PPO_CriticModel_Default,

    # ---- DQN Q-networks
    ("q_network", "Equivariant", "SelectNodesSequentially"):
        DQN_QNetwork_Equivariant_SelectNodesSequentially,
    ("q_network", "Equivariant", "AddRemoveEdgeDiscreteNoSelfLoops"):
        DQN_QNetwork_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops,
    ("q_network", "Equivariant", "AddEdgeDiscreteNoSkipNoSelfLoops"):
        DQN_QNetwork_Equivariant_AddEdgeDiscreteNoSkipNoSelfLoops,
    ("q_network", "GINE", "SelectNodesSequentially"):
        DQN_QNetwork_GINE_SelectNodesSequentially,
    ("q_network", "GINE", "AddRemoveEdgeDiscreteNoSelfLoops"):
        DQN_QNetwork_GINE_AddRemoveEdgeDiscreteNoSelfLoops,
    ("q_network", "GINE", "AddEdgeDiscreteNoSkipNoSelfLoops"):
        DQN_QNetwork_GINE_AddEdgeDiscreteNoSkipNoSelfLoops,
    ("q_network", "Default", "SelectNodesSequentially"):
        DQN_QNetwork_SelectNodesSequentially,
    ("q_network", "Default", "AddRemoveEdgeDiscreteNoSelfLoops"):
        DQN_QNetwork_AddRemoveEdgeDiscreteNoSelfLoops,
    ("q_network", "Default", "AddEdgeDiscreteNoSelfLoops"):
        DQN_QNetwork_AddEdgeDiscreteNoSelfLoops,
}


def instantiate(cls, kwargs):
    """Constructors differ (edge_feat_dim, allow_skip, ...), so pass only what
    this one declares."""
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def resolve(role, backbone, action_type):
    cls = MODELS.get((role, backbone, action_type)) or MODELS.get((role, backbone, None))
    if cls is None:
        available = sorted(a for r, b, a in MODELS if r == role and b == backbone and a)
        raise KeyError(
            f"no {role} model for backbone {backbone!r} and action {action_type!r}; "
            f"{backbone} has {available or 'nothing'}"
        )
    return cls


def build_models(algorithm, backbone, action_type, **kwargs):
    """The skrl model dict for this algorithm, ready to hand to the agent."""
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}, expected one of {BACKBONES}")
    return {
        role: instantiate(resolve(role, backbone, action_type), kwargs)
        for role in ALGORITHM_ROLES[algorithm]
    }
