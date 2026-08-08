from .actor.AddEdgeDiscreteNoSelfLoops import *
from .actor.AddEdgeDiscreteNoSkipNoSelfLoops import *
from .actor.AddRemoveEdgeDiscreteNoSelfLoops import *
from .actor.AddRemoveEdgeDiscreteNoSelfLoops_FC import *
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

from .registry import BACKBONES, MODELS, build_models, instantiate, resolve


__all__ = [
    "BACKBONES",
    "MODELS",
    "build_models",
    "instantiate",
    "resolve",

    "PPO_ActorModel_AddEdgeDiscreteNoSelfLoops",
    "PPO_ActorModel_AddEdgeDiscreteNoSkipNoSelfLoops",
    "PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops",
    "PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops_FC",
    "PPO_ActorModel_AddRemoveEdgeMultiDiscrete",
    "PPO_ActorModel_AllEdges",
    "PPO_ActorModel_DecideOnEdge",
    "PPO_ActorModel_SelectNodesSequentially",
    "PPO_ActorModel_Equivariant_SelectNodesSequentially",
    "PPO_ActorModel_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops",
    "PPO_ActorModel_GINE_AddRemoveEdgeDiscreteNoSelfLoops",
    "PPO_ActorModel_GINE_SelectNodesSequentially",

    "PPO_CriticModel_Default",
    "PPO_CriticModel_Selection",
    "PPO_CriticModel_Equivariant",
    "PPO_CriticModel_Equivariant_Selection",
    "PPO_CriticModel_GINE",
    "PPO_CriticModel_GINE_Selection",

    "DQN_QNetwork_AddEdgeDiscreteNoSelfLoops",
    "DQN_QNetwork_AddRemoveEdgeDiscreteNoSelfLoops",
    "DQN_QNetwork_SelectNodesSequentially",
    "DQN_QNetwork_GINE_SelectNodesSequentially",
    "DQN_QNetwork_GINE_AddRemoveEdgeDiscreteNoSelfLoops",
    "DQN_QNetwork_GINE_AddEdgeDiscreteNoSkipNoSelfLoops",
    "DQN_QNetwork_Equivariant_SelectNodesSequentially",
    "DQN_QNetwork_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops",
    "DQN_QNetwork_Equivariant_AddEdgeDiscreteNoSkipNoSelfLoops",
]
