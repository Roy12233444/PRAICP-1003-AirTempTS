# src/agents/__init__.py

import importlib
import sys
from typing import Optional, Type, Dict, Any
from .agent_base import BaseAgent

# Dictionary to store agent classes
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}

def safe_import_agent(module_name: str, class_name: str) -> Optional[Type[BaseAgent]]:
    """Safely import an agent class, returning None if import fails."""
    try:
        module = importlib.import_module(f"agents.{module_name}")
        agent_class = getattr(module, class_name, None)
        if agent_class and issubclass(agent_class, BaseAgent):
            return agent_class
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Warning: Could not import {class_name} from {module_name}: {str(e)}")
    return None

# Dynamically import all agents
agent_definitions = [
    ("agent01_resonant_decomposition_agent", "ResonantDecompositionAgent"),
    ("agent02_wavelet_transient_agent", "WaveletTransientAgent"),
    ("agent03_feature_alchemist_agent", "FeatureAlchemistAgent"),
    ("agent04_bayesian_fusion_agent", "BayesianFusionAgent"),
    ("agent05_uncertainty_synthesis_agent", "UncertaintySynthesisAgent"),
    ("agent06_change_point_regime_agent", "ChangePointRegimeAgent"),
    ("agent07_tesla_oscillation_agent", "TeslaOscillationAgent"),
    ("agent08_koopman_mode_agent", "KoopmanModeAgent"),
    ("agent09_physics_informed_thermal_agent", "PhysicsInformedThermalAgent"),
    ("agent10_temporal_graph_resonance_agent", "TemporalGraphResonanceAgent"),
    ("agent11_data_augmentation_agent", "DataAugmentationAgent"),
    ("agent12_data_ingest_agent", "DataIngestAgent"),
    ("agent13_drift_detector_agent", "DriftDetectorAgent"),
    ("agent14_tracker_agent", "TrackerAgent"),
    ("agent15_trainer_agent", "TrainerAgent"),
    ("agent16_uncertainty_agent", "UncertaintyAgent"),
]

# Populate the registry
for module_name, class_name in agent_definitions:
    agent_class = safe_import_agent(module_name, class_name)
    if agent_class:
        AGENT_REGISTRY[class_name] = agent_class
        # Also make available directly in module namespace
        globals()[class_name] = agent_class

def get_agent_class(agent_name: str) -> Optional[Type[BaseAgent]]:
    """Safely get an agent class by name."""
    return AGENT_REGISTRY.get(agent_name)

def create_agent(agent_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
    """Create an agent instance by name with the given config."""
    agent_class = get_agent_class(agent_name)
    if agent_class:
        try:
            return agent_class(config or {})
        except Exception as e:
            print(f"Error creating agent {agent_name}: {str(e)}")
    return None

__all__ = [
    "BaseAgent",
    "ResonantDecompositionAgent",
    "WaveletTransientAgent",
    "FeatureAlchemistAgent",
    "BayesianFusionAgent",
    "UncertaintySynthesisAgent",
    "ChangePointRegimeAgent",
    "TeslaOscillationAgent",
    "KoopmanModeAgent",
    "PhysicsInformedThermalAgent",
    "TemporalGraphResonanceAgent",
    "DataAugmentationAgent",
    "DataIngestAgent",
    "DriftDetectorAgent",
    "TrackerAgent",
    "TrainerAgent",
    "UncertaintyAgent",
]