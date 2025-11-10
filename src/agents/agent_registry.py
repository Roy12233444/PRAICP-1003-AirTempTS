# src/agents/agent_registry.py
from typing import Dict
from agents.agent01_resonant_decomposition_agent import ResonantDecompositionAgent
from agents.agent02_wavelet_transient_agent import WaveletTransientAgent
from agents.agent03_feature_alchemist_agent import FeatureAlchemistAgent
from agents.agent04_bayesian_fusion_agent import BayesianFusionAgent
from agents.agent05_uncertainty_synthesis_agent import UncertaintySynthesisAgent
from agents.agent06_change_point_regime_agent import ChangePointRegimeAgent
from agents.agent07_tesla_oscillation_agent import TeslaOscillationAgent
# optional imports for skeletons
# from agents.koopman_mode_agent import KoopmanModeAgent
# from agents.physics_informed_thermal_agent import PhysicsInformedThermalAgent

REGISTRY: Dict[str, object] = {
    "resonant": ResonantDecompositionAgent,
    "wavelet": WaveletTransientAgent,
    "alchemist": FeatureAlchemistAgent,
    "bayes_fusion": BayesianFusionAgent,
    "uncertainty": UncertaintySynthesisAgent,
    "changepoint": ChangePointRegimeAgent,
    "tesla_osc": TeslaOscillationAgent,
    # "koopman": KoopmanModeAgent,
}

def get_agent(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Agent '{name}' not found. Available: {list(REGISTRY.keys())}")
    return cls(**kwargs)
