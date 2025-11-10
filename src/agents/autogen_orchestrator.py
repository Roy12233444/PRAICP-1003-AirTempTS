"""
Auto-generation orchestrator helper (fixed imports + robust instantiation)
- Scans src/agents for classes with empty methods (pass / raise NotImplementedError / TODO)
- Scores candidates and (optionally) writes template implementations
- Backs up original files before writing
- Can optionally discover, instantiate and run existing agents (robustly handling constructors)
- Usage examples:
    # dry-run
    python autogen_orchestrator.py --root <PROJECT_ROOT> --budget 4.0 --lam 0.01
    # write selected templates
    python autogen_orchestrator.py --root <PROJECT_ROOT> --apply --budget 4.0 --lam 0.01
    # force specific families / filenames (comma-separated fragments)
    python autogen_orchestrator.py --root <PROJECT_ROOT> --apply --force "koopman,temporal_graph"
    # optionally run discovered agents (fit/transform) on the dataset (requires src/data_loader.py)
    python autogen_orchestrator.py --root <PROJECT_ROOT> --run-agents --run-csv ".\data\surface-air-temperature-monthly-mean.csv"
"""
import os
import sys
import ast
import time
import shutil
import argparse
import math
import json
import pandas as pd
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# -------------------------- Templates -----------------------------
# Each template is a safe, guarded implementation that only uses light deps
TEMPLATES = {
    "koopman": r'''# Auto-generated: KoopmanModeAgent (basic EDMD) - PLEASE REVIEW
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from .agent_base import BaseAgent

class KoopmanModeAgent(BaseAgent):
    """
    Basic EDMD implementation using polynomial + trig observables.
    Fit: construct G0,G1 from lag vectors and solve K = G1 @ pinv(G0)
    Transform: project state observables onto Koopman eigenvectors (modes).
    Note: this is a starter implementation — review and refine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lag = int(self.config.get("lag", 3))
        self.obs_degree = int(self.config.get("poly_degree", 2))
        self.K = None
        self.modes = None

    def _observables(self, x_vec):
        # build polynomial observables [1, x1, x1^2, x2, x2^2, sin(x1), cos(x1), ...]
        feats = [1.0]
        for xi in x_vec:
            for p in range(1, self.obs_degree + 1):
                feats.append(xi ** p)
            feats.append(np.sin(xi))
            feats.append(np.cos(xi))
        return np.asarray(feats, dtype=float)

    def fit(self, df, target_col="mean_temp"):
        # build lagged state matrix (rows: time, cols: lag features)
        arr = df[target_col].astype(float).values
        N = len(arr) - self.lag
        if N <= 0:
            raise ValueError("Not enough data for Koopman fit with lag={}".format(self.lag))
        G0_list, G1_list = [], []
        for t in range(N):
            x_t = arr[t : t + self.lag]
            x_tp1 = arr[t + 1 : t + 1 + self.lag]
            G0_list.append(self._observables(x_t))
            G1_list.append(self._observables(x_tp1))
        G0 = np.vstack(G0_list).T  # shape (d, N)
        G1 = np.vstack(G1_list).T
        # EDMD: K = G1 @ pinv(G0)
        G0_pinv = np.linalg.pinv(G0)
        K = G1 @ G0_pinv
        self.K = K
        # eigen-decomposition for modes
        try:
            vals, vecs = np.linalg.eig(K)
            self.modes = (vals, vecs)
        except Exception:
            self.modes = None
        return self

    def transform(self, df, target_col="mean_temp"):
        # project last lag into observables and (optionally) return mode amplitudes
        arr = df[target_col].astype(float).values
        if len(arr) < self.lag:
            raise ValueError("Not enough points for transform")
        x = arr[-self.lag :]
        g = self._observables(x)
        if self.modes is not None and self.modes[1] is not None:
            # coordinates in mode basis
            vecs = self.modes[1]
            coords = np.linalg.lstsq(vecs, g, rcond=None)[0]
            out = df.copy().reset_index(drop=True)
            for i, val in enumerate(np.real(coords)):
                out[f"koopman_mode_{i}"] = float(val)
            return out
        else:
            out = df.copy().reset_index(drop=True)
            out["koopman_feat0"] = float(g[0])
            return out

    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self
''',

    "physics": r'''# Auto-generated: PhysicsInformedThermalAgent - PLEASE REVIEW
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from .agent_base import BaseAgent
from scipy.signal import savgol_filter
import joblib

class PhysicsInformedThermalAgent(BaseAgent):
    """
    Practical 'physics-informed' wrapper:
    - Fit a GradientBoostingRegressor on features
    - During predict apply a smoothing penalty using Savitzky-Golay filter
      to enforce smoothness (proxy for physics).
    NOTE: this is a pragmatic approach for the capstone; replace with PINN if desired.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.smooth_window = int(self.config.get("smooth_window", 5))
        self.polyorder = int(self.config.get("polyorder", 2))

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp", feature_cols: Optional[list] = None):
        if feature_cols is None:
            # use simple lags if nothing provided
            arr = df[target_col].astype(float)
            X = pd.DataFrame({f"lag_{i}": arr.shift(i) for i in range(1,4)}).fillna(method="bfill")
        else:
            X = df[feature_cols].fillna(0.0)
        y = df[target_col].astype(float).fillna(method='ffill').fillna(0.0)
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        self.model.fit(X, y)
        self.feature_cols = X.columns.tolist()
        return self

    def predict(self, df: pd.DataFrame):
        X = df[self.feature_cols].fillna(0.0)
        raw = self.model.predict(X)
        # apply Savitzky-Golay smoothing to enforce smoothness (physics proxy)
        win = self.smooth_window
        if len(raw) < win:
            smoothed = raw
        else:
            # ensure window is odd and <= len
            if win % 2 == 0:
                win = win - 1
            win = max(3, min(win, len(raw) if len(raw)%2==1 else len(raw)-1))
            smoothed = savgol_filter(raw, win, self.polyorder)
        # return in uniform dict format
        return {"mu": float(smoothed[-1]) if len(smoothed)>0 else float(np.nan),
                "series": smoothed, "var": float(np.var(raw - smoothed) + 1e-6)}

    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self
''',

    "temporal_graph": r'''# Auto-generated: TemporalGraphResonanceAgent (fallback implementation)
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from .agent_base import BaseAgent
import joblib

class TemporalGraphResonanceAgent(BaseAgent):
    """
    Minimal fallback: if no spatial graph present, compute local aggregations
    across features (mean, std) as proxy for graph summary.
    Full GNN implementation requires PyTorch Geometric and node/edge data.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp"):
        # nothing to fit for fallback
        return self

    def transform(self, df: pd.DataFrame):
        out = df.copy().reset_index(drop=True)
        numeric = out.select_dtypes(include=[np.number])
        out["tg_mean"] = numeric.mean(axis=1)
        out["tg_std"] = numeric.std(axis=1).fillna(0.0)
        return out

    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self
''',

    "augmentation": r'''# Auto-generated: DataAugmentationAgent (bootstrap + jitter)
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from .agent_base import BaseAgent
import joblib

class DataAugmentationAgent(BaseAgent):
    """
    Simple bootstrap-time-series augmentation:
     - sliding windows sampled with replacement
     - small gaussian jitter and optional scaling of anomalies
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window = int(self.config.get("window", 12))

    def bootstrap_augment(self, df: pd.DataFrame, n_samples: int = 100):
        y = df['mean_temp'].astype(float).values
        N = len(y)
        if N < self.window:
            raise ValueError("Not enough data to bootstrap windows")
        samples = []
        for i in range(n_samples):
            start = np.random.randint(0, N - self.window + 1)
            w = y[start:start + self.window].copy()
            # jitter small noise
            w = w + np.random.normal(0, 0.05 * np.std(y), size=w.shape)
            samples.append(w)
        # return as DataFrame of shape (n_samples, window)
        return pd.DataFrame(samples, columns=[f"t_{i}" for i in range(self.window)])

    def fit(self, df, **kwargs):
        # nothing to fit
        return self

    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self
''',
}

# -------------------------- Utility functions ----------------------
def find_agent_files(root_dir: Path) -> List[Path]:
    p = root_dir / "src" / "agents"
    if not p.exists():
        raise FileNotFoundError(f"No agents dir found at expected: {p}")
    files = [f for f in p.glob("*.py") if f.name != "__init__.py"]
    return files

def parse_classes_with_empty_methods(path: Path) -> Dict[str, List[str]]:
    """
    Return dict: {class_name: [list of empty method names]}
    Empty method = body contains pass OR raises NotImplementedError OR contains TODO
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    res: Dict[str, List[str]] = {}
    for node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        method_names = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # detect pass-only
                if len(item.body) == 1 and isinstance(item.body[0], ast.Pass):
                    method_names.append(item.name)
                    continue
                # detect raise NotImplementedError
                if len(item.body) == 1 and isinstance(item.body[0], ast.Raise):
                    method_names.append(item.name)
                    continue
                # detect TODO in body strings/comments (simple check)
                bodies = ast.get_source_segment(src, item)
                if bodies and ("TODO" in bodies or "NotImplemented" in bodies):
                    method_names.append(item.name)
        if method_names:
            res[node.name] = method_names
    return res

def score_candidate(file_path: Path, class_name: str) -> Tuple[float, str]:
    """
    Simple heuristic score and family detection based on file/class name and docstrings.
    Returns (similarity_score in [0,1], family_key)
    """
    text = file_path.read_text(encoding="utf-8").lower()
    name = class_name.lower()
    # keywords map
    families = {
        "koopman": ["koopman", "edmd"],
        "physics": ["physics", "pinn", "pin", "thermal"],
        "temporal_graph": ["graph", "spatio", "spatio", "temporalgraph", "graph"],
        "augmentation": ["augment", "augmentation", "gan", "vae"],
        "resonant": ["resonant", "resonance", "fft", "dft", "harmonic"],
        "wavelet": ["wavelet", "pywt"],
        "alchemist": ["alchemist", "alchemist", "feature", "boruta", "shap"],
    }
    scores = {}
    for fam, keys in families.items():
        s = 0
        for k in keys:
            if k in text or k in name:
                s += 1
        scores[fam] = s
    # choose max
    best = max(scores.items(), key=lambda x: x[1])
    max_score = best[1]
    # compute normalized similarity 0..1 (simple)
    sim = min(1.0, max_score / 2.0)
    # fallback family if no strong match
    family = best[0] if sim > 0 else "koopman"
    return sim, family

def estimate_cost_and_benefit(family: str, empty_methods: List[str]) -> Tuple[float, float]:
    """
    Return (C_j, DeltaPerf_j) heuristics.
    Cost roughly relates to number of empty methods and family heaviness.
    Benefit prior by family (simple heuristics).
    """
    # base cost per method
    base = 0.5 * len(empty_methods)
    heavy_factor = 1.0
    if family in ("temporal_graph", "augmentation"):
        heavy_factor = 3.0
    elif family in ("koopman", "physics"):
        heavy_factor = 1.5
    else:
        heavy_factor = 1.0
    C = base * heavy_factor
    # benefit priors (expected RMSE reduction or normalized score)
    priors = {
        "resonant": 0.25,
        "wavelet": 0.20,
        "koopman": 0.1,
        "physics": 0.2,
        "bayes": 0.15,
        "temporal_graph": 0.2,
        "augmentation": 0.05,
        "alchemist": 0.3
    }
    Delta = priors.get(family, 0.08)
    return C, Delta

def decide_generation(candidates: List[Tuple[Path, str, List[str]]], budget: float, lam: float = 0.5):
    """
    candidates: list of (file_path, class_name, empty_methods)
    Return list of (file_path, class_name, family, sim, C, Delta, U)
    """
    scored = []
    for fp, cname, methods in candidates:
        sim, family = score_candidate(fp, cname)
        p_success = 1.0 / (1.0 + math.exp(- ( -1.0 + 2.5 * sim)))  # sigmoid with tuned params
        C, Delta = estimate_cost_and_benefit(family, methods)
        U = p_success * Delta - lam * C
        scored.append((fp, cname, family, sim, C, Delta, p_success, U, methods))
    # select by U/C ratio greedily under budget
    scored_sorted = sorted(scored, key=lambda x: (x[7] / (x[4] + 1e-6)), reverse=True)
    chosen = []
    rem = budget
    for item in scored_sorted:
        if item[4] <= rem and item[7] > 0:
            chosen.append(item)
            rem -= item[4]
    return chosen, scored_sorted

def generate_template_for_family(family: str) -> str:
    tpl = TEMPLATES.get(family)
    if tpl:
        return tpl
    # fallback default skeleton
    return r'''# Auto-generated skeleton - please implement
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from .agent_base import BaseAgent
import joblib

class AutoGeneratedAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def fit(self, df, target_col="mean_temp"):
        return self

    def transform(self, df):
        out = df.copy()
        return out

    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self
'''

def backup_and_write_file(path: Path, content: str):
    ts = int(time.time())
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy(path, bak)
    path.write_text(content, encoding="utf-8")
    return bak

# -------------------------- Optional data-loader import ----------------------
def import_data_loader_module(project_root: Path):
    """
    Safely import src/data_loader.py as a module (returns module or raises).
    Only used if you pass --run-agents.
    """
    dl_path = project_root / "src" / "data_loader.py"
    if not dl_path.exists():
        raise FileNotFoundError(f"data_loader.py not found at expected: {dl_path}")
    spec = importlib.util.spec_from_file_location("project_data_loader", str(dl_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_and_run_agents(root_dir: Path,
                        run_on_csv: Optional[Path] = None,
                        load_results: bool = False,
                        mapping_file: Optional[str] = None):
    """
    Dynamically import all BaseAgent subclasses, instantiate them robustly, and run fit/transform on provided data.
    This is optional and only triggered by --run-agents. It uses src/data_loader.py when available.
    Can also load pre-computed results from a mapping file.
    """
    # Ensure src and src/agents are on path for relative imports inside agent modules
    src_path = str(root_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Import the agents module to register all agents
    try:
        import agents
        from agents import AGENT_REGISTRY, create_agent
    except ImportError as e:
        print(f"Failed to import agents module: {e}")
        return {}

    # Create instances of all registered agents
    discovered = []
    for agent_name in AGENT_REGISTRY.keys():
        try:
            agent = create_agent(agent_name)
            if agent:
                discovered.append((agent_name, agent))
        except Exception as e:
            print(f"Failed to create agent {agent_name}: {e}")

    print(f"Discovered {len(discovered)} agent instances.")
    
    # If no agents were discovered, try fallback discovery method
    if not discovered:
        print("Using fallback agent discovery method...")
        files = find_agent_files(root_dir)
        for file_path in files:
            try:
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Look for agent classes in the module
                for name, obj in module.__dict__.items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, BaseAgent) and 
                        obj != BaseAgent):
                        try:
                            inst = obj()
                            discovered.append((name, inst))
                        except Exception as e:
                            print(f"Failed to instantiate {name} from {file_path.name}: {e}")
            except Exception as e:
                print(f"Failed to import {file_path.name}: {e}")
        
        print(f"Discovered {len(discovered)} agent instances using fallback method.")
    # optionally load CSV using data_loader if provided
    df = None
    if run_on_csv is not None:
        try:
            dl_mod = import_data_loader_module(root_dir)
            # allow pass either Path or string
            csv_path = str(run_on_csv) if not isinstance(run_on_csv, Path) else str(run_on_csv)
            df = dl_mod.load_airtemp(csv_path=csv_path)
            print(f"Loaded data for run_agents: {len(df)} rows")
        except Exception as e:
            print("Failed to load CSV via data_loader module:", e)
    # If loading results, use the mapping file. Otherwise, run agents normally.
    if load_results:
        print("--- Loading pre-computed results from mapping file ---")
        mapping_path = root_dir / mapping_file
        if not mapping_path.exists():
            print(f"Error: Mapping file not found at {mapping_path}")
            return {}
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        results = {}
        for name, agent in discovered:
            if name in mapping:
                try:
                    # Load the model state into the agent instance
                    model_path_str = mapping[name].get('model')
                    if model_path_str:
                        model_path = root_dir / model_path_str
                        if hasattr(agent, 'load') and model_path.exists():
                            agent.load(str(model_path))
                            print(f"Loaded model for {name} from {model_path.name}")
                        elif not hasattr(agent, 'load'):
                             print(f"Warning: Agent {name} has no .load() method.")
                        else:
                            print(f"Warning: Model file not found for {name} at {model_path}")

                    # Load the pre-computed CSV result
                    csv_path_str = mapping[name].get('csv')
                    if csv_path_str:
                        csv_path = root_dir / csv_path_str
                        if csv_path.exists():
                            df_out = pd.read_csv(csv_path)
                            results[name] = {"transform_head": df_out.head(1).to_dict(orient="list")}
                            print(f"Loaded CSV result for {name} from {csv_path.name}")
                        else:
                            results[name] = {"error": f"CSV file not found at {csv_path}"}
                    else:
                        results[name] = {"status": "No CSV path in mapping"}

                except Exception as e:
                    results[name] = {"error": f"Failed to load results for {name}: {e}"}
            else:
                results[name] = {"status": "not_in_mapping"}
        return results

    # --- Original behavior: run fit/transform ---
    print("--- Running agents (fit/transform) ---")
    results = {}
    for name, agent in discovered:
        try:
            if df is not None:
                try:
                    agent.fit(df)
                except Exception:
                    # non-fatal: continue
                    pass
                try:
                    out = agent.transform(df)
                    # try to get small readable head
                    if hasattr(out, "head"):
                        results[name] = {"transform_head": out.head(1).to_dict(orient="list")}
                    else:
                        results[name] = {"transform_result_type": str(type(out))}
                except Exception as e:
                    # try predict_with_fallback if available
                    if hasattr(agent, "predict_with_fallback"):
                        try:
                            pf = agent.predict_with_fallback(df)
                            results[name] = {"predict_with_fallback": pf}
                        except Exception as e2:
                            results[name] = {"transform": f"failed ({e}), fallback failed ({e2})"}
                    else:
                        results[name] = {"transform": f"failed ({e})"}
            else:
                results[name] = {"status": "ready"}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

# -------------------------- Main run ------------------------------
def main(project_root: str,
         apply_changes: bool = False,
         budget: float = 3.0,
         lam: float = 0.5,
         force: str = "",
         run_agents: bool = False,
         run_csv: Optional[str] = None,
         load_results: bool = False,
         mapping_file: Optional[str] = None):
    root = Path(project_root)
    files = find_agent_files(root)
    candidates: List[Tuple[Path, str, List[str]]] = []
    for f in files:
        classes = parse_classes_with_empty_methods(f)
        for cls, empty_methods in classes.items():
            candidates.append((f, cls, empty_methods))
    if not candidates:
        print("No empty agent classes detected. Nothing to autogenerate.")
        # optionally still run agents if requested
        if run_agents:
            csv_path = Path(run_csv) if run_csv else None
            res = load_and_run_agents(root, run_on_csv=csv_path, load_results=load_results, mapping_file=mapping_file)
            print("Run agents results:", res)
        return
    print(f"Found {len(candidates)} candidate(s) with empty methods:")
    for f, cname, empty_methods in candidates:
        print(f" - {f.name} :: class {cname} has empty methods {empty_methods}")

    # -----------------------------
    # Filter out base/interface files
    # -----------------------------
    filtered: List[Tuple[Path, str, List[str]]] = []
    for f, cname, empty_methods in candidates:
        fname = f.name.lower()
        # load text safely
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            txt = ""
        # skip files explicitly marked to skip
        if "# AUTOGEN_SKIP" in txt:
            print(f"Skipping (marked) {f.name} :: {cname}")
            continue
        # skip obvious base/interface modules
        if "base" in fname or cname.lower().endswith("base") or cname.lower() == "baseagent":
            print(f"Skipping interface/base file from autogen consideration: {f.name} :: {cname}")
            continue
        filtered.append((f, cname, empty_methods))
    candidates = filtered

    if not candidates:
        print("No empty agent classes detected (after filtering). Nothing to autogenerate.")
        # optionally still run agents if requested
        if run_agents:
            csv_path = Path(run_csv) if run_csv else None
            res = load_and_run_agents(root, run_on_csv=csv_path)
            print("Run agents results:", res)
        return

    # --------------------------------
    # Handle explicit forcing (override)
    # --------------------------------
    forced_list = [s.strip().lower() for s in (force or "").split(",") if s.strip()]
    if forced_list:
        forced_candidates: List[Tuple[Path, str, List[str]]] = []
        for f, cname, empty_methods in candidates:
            keynames = [f.name.lower(), cname.lower()]
            if any(fk in kn for fk in forced_list for kn in keynames):
                forced_candidates.append((f, cname, empty_methods))
        if not forced_candidates:
            print("Warning: --force requested but no matching candidates found in filtered set; proceeding with normal scoring.")
        else:
            candidates = forced_candidates
            print("FORCE: reduced candidate set to:", [c[0].name for c in candidates])

    chosen, scored_sorted = decide_generation(candidates, budget=budget, lam=lam)
    print("\nScored candidates (all):")
    for item in scored_sorted:
        fp, cname, family, sim, C, Delta, p_success, U, methods = item
        print(f"{fp.name}::{cname} -> family={family}, sim={sim:.2f}, cost={C:.2f}, priorΔ={Delta:.3f}, p={p_success:.2f}, U={U:.3f}")
    if not chosen:
        print("\nNo candidate selected under budget/utility threshold.")
        # optionally still run agents
        if run_agents:
            csv_path = Path(run_csv) if run_csv else None
            res = load_and_run_agents(root, run_on_csv=csv_path)
            print("Run agents results:", res)
        return
    print("\nCandidates selected for autogeneration (under budget):")
    for item in chosen:
        fp, cname, family, sim, C, Delta, p_success, U, methods = item
        print(f" * {fp.name}::{cname} -> family={family}, cost={C:.2f}, expectedU={U:.3f}")
    if not apply_changes:
        print("\nDry run mode: use --apply to write files. (Files will be backed up)")
        # optionally still run agents if requested
        if run_agents:
            csv_path = Path(run_csv) if run_csv else None
            res = load_and_run_agents(root, run_on_csv=csv_path)
            print("Run agents results:", res)
        return
    # apply changes
    for item in chosen:
        fp, cname, family, sim, C, Delta, p_success, U, methods = item
        print(f"\nGenerating code for {fp.name}::{cname} (family={family}) ...")
        template = generate_template_for_family(family)
        header = f"# AUTOGENERATED BY AetherOrchestrator at {time.strftime('%Y-%m-%d %H:%M:%S')}\n# Family: {family}\n# NOTE: review this file manually before production use\n\n"
        new_content = header + template
        bak = backup_and_write_file(fp, new_content)
        print(f"Backed up original to: {bak.name} and wrote new content to {fp.name}")
    print("\nAutogeneration complete. Please review generated files in your editor.")

    # optionally run agents after generation
    if run_agents:
        csv_path = Path(run_csv) if run_csv else None
        res = load_and_run_agents(root, run_on_csv=csv_path)
        print("Run agents results:", res)

# -------------------------- CLI ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root (containing src/agents)")
    parser.add_argument("--apply", action="store_true", help="Write files (default is dry-run)")
    parser.add_argument("--budget", type=float, default=3.0, help="Total budget units for generation")
    parser.add_argument("--lam", type=float, default=0.5, help="Cost penalty weight lambda")
    parser.add_argument("--force", type=str, default="", help="Comma-separated names (filename or classname fragment) to force-generate (overrides budget selection)")
    parser.add_argument("--run-agents", action="store_true", help="After generation (or on dry-run) attempt to import and run discovered agents")
    parser.add_argument("--run-csv", type=str, default="", help="Optional CSV path to pass to run-agents (absolute or relative to project root)")
    parser.add_argument("--load-results", action="store_true", help="Load existing agent models and results instead of running them")
    parser.add_argument("--mapping-file", type=str, default="results/model_mapping.json", help="Path to the model mapping JSON file")
    args = parser.parse_args()

    # normalize root path and pass into main
    proj_root = Path(args.root).resolve()
    run_csv = args.run_csv if args.run_csv else None
    main(str(proj_root), apply_changes=args.apply, budget=args.budget, lam=args.lam, force=args.force, run_agents=args.run_agents, run_csv=run_csv, load_results=args.load_results, mapping_file=args.mapping_file)
