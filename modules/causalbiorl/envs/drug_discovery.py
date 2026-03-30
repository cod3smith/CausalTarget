"""
DrugDiscovery-v0 — Gymnasium environment for RL-driven drug discovery.

This is the **integration point** where CausalBioRL meets the real
NeoRx pipeline.  Instead of toy ODE environments, the RL agent
interacts with real drug-discovery modules:

    - **NeoRx** — disease graph, causal targets, SCM
    - **GenMol** — VAE-based molecule generation (latent space navigation)
    - **MolScreen** — drug-likeness scoring (QED, SA, filters)
    - **DockBot** — binding affinity (via surrogate for speed)
    - **MirrorFold** — structural stability assessment

Architecture
------------
State:
    Graph embedding (128D from R-GCN) + current best molecule features (32D)
    + per-target summary (n_targets × 8D) → flattened to fixed size.

Action (hierarchical):
    Level 1: Target selection (discrete: which target to pursue)
    Level 2: Molecule generation (continuous 128D delta-z in GenMol latent)

Reward:
    Adaptively-weighted multi-objective (AdaptiveRewardLearner):
    binding, QED, SA, novelty, causal confidence, stability.

Episode:
    One disease campaign.  Agent has a budget of N generate-screen
    cycles.  It decides which targets to invest in and how to explore
    the chemical space for each.

Termination:
    Budget exhausted OR agent outputs "stop" action (early confidence).

Design Decisions
-----------------
    - Disease graph is built once at ``reset()`` and cached.
    - DockBot uses a surrogate model during training; real docking
      only at episode end for recalibration.
    - GenMol generates from ``z_base + delta_z`` where ``z_base``
      is sampled once per target and ``delta_z`` is the agent's action.
    - The agent learns to navigate GenMol's latent space, not just
      sample randomly.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
#  Constants                                                                 #
# ────────────────────────────────────────────────────────────────────────── #

GRAPH_EMBEDDING_DIM = 128
LATENT_DIM = 128          # GenMol VAE latent dimension
MOL_FEATURE_DIM = 32      # Molecular property features
TARGET_FEATURE_DIM = 8    # Per-target summary features
MAX_TARGETS = 10          # Maximum number of targets supported

# Total observation dim:
# graph_emb (128) + mol_features (32) + target_summaries (10*8) + meta (4)
OBS_DIM = GRAPH_EMBEDDING_DIM + MOL_FEATURE_DIM + (MAX_TARGETS * TARGET_FEATURE_DIM) + 4

# ────────────────────────────────────────────────────────────────────────── #
#  Target Tracker                                                            #
# ────────────────────────────────────────────────────────────────────────── #


class _TargetState:
    """Tracks the agent's progress on a single target."""

    def __init__(
        self,
        target_info: dict[str, Any],
        target_idx: int,
    ) -> None:
        self.target_info = target_info
        self.target_idx = target_idx
        self.gene_name: str = target_info.get("gene_name", f"target_{target_idx}")
        self.causal_confidence: float = target_info.get("causal_confidence", 0.5)
        self.pdb_ids: list[str] = target_info.get("pdb_ids", [])
        self.node_embedding: NDArray[np.floating] = np.zeros(
            GRAPH_EMBEDDING_DIM, dtype=np.float32,
        )

        # Best molecule found for this target
        self.best_smiles: str = ""
        self.best_score: float = -np.inf
        self.best_objectives: dict[str, float] = {}

        # Running z-vector for latent space navigation
        self.z_base: NDArray[np.floating] = np.random.randn(LATENT_DIM).astype(np.float32) * 0.5

        # Number of attempts on this target
        self.n_attempts: int = 0

    def summary_features(self) -> NDArray[np.floating]:
        """Return an 8-D summary vector for observation space."""
        return np.array([
            self.causal_confidence,
            min(self.n_attempts / 20.0, 1.0),  # normalised attempt count
            max(self.best_score, 0.0),           # best composite score
            self.best_objectives.get("binding", 0.0),
            self.best_objectives.get("qed", 0.0),
            self.best_objectives.get("sa", 0.0),
            self.best_objectives.get("novelty", 0.0),
            self.best_objectives.get("stability", 0.0),
        ], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────── #
#  DrugDiscoveryEnv                                                          #
# ────────────────────────────────────────────────────────────────────────── #


class DrugDiscoveryEnv(gym.Env):
    """Gymnasium environment for RL-driven drug discovery campaigns.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "Malaria", "HIV").
    max_steps : int
        Budget: maximum generate-screen cycles per episode.
    top_n_targets : int
        Number of causal targets to pursue (from NeoRx pipeline).
    latent_dim : int
        GenMol VAE latent space dimension.
    use_surrogate : bool
        Use surrogate docking model (fast) vs real DockBot (slow).
    recalibrate_interval : int
        Every N steps, run real docking to recalibrate surrogate.
    difficulty : str
        ``"easy"`` / ``"medium"`` / ``"hard"`` — affects noise and
        budget constraints.
    prebuilt_graph : nx.DiGraph | None
        Pre-built disease graph.  If provided, skips graph building.
    prebuilt_targets : list[dict] | None
        Pre-identified targets.  If provided, skips identification.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    _DIFFICULTY = {
        "easy":   {"budget_mult": 2.0, "noise": 0.0},
        "medium": {"budget_mult": 1.0, "noise": 0.02},
        "hard":   {"budget_mult": 0.5, "noise": 0.05},
    }

    def __init__(
        self,
        disease: str = "Malaria",
        max_steps: int = 50,
        top_n_targets: int = 5,
        latent_dim: int = LATENT_DIM,
        use_surrogate: bool = True,
        recalibrate_interval: int = 10,
        difficulty: str = "medium",
        prebuilt_graph: nx.DiGraph | None = None,
        prebuilt_targets: list[dict[str, Any]] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        assert difficulty in self._DIFFICULTY

        self.disease = disease
        self.latent_dim = latent_dim
        self.use_surrogate = use_surrogate
        self.recalibrate_interval = recalibrate_interval
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.top_n_targets = min(top_n_targets, MAX_TARGETS)

        diff = self._DIFFICULTY[difficulty]
        self.max_steps = int(max_steps * diff["budget_mult"])
        self._noise_std: float = diff["noise"]

        # Pre-built data (can be injected for testing / speed)
        self._prebuilt_graph = prebuilt_graph
        self._prebuilt_targets = prebuilt_targets

        # ── Observation Space ──────────────────────────────────
        # Fixed-size vector: graph_emb + mol_features + target_summaries + meta
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # ── Action Space (Hierarchical) ───────────────────────
        # [0]: target_selection ∈ [0,1] → discretised to target index
        # [1]: stop_signal ∈ [0,1] → if > 0.9, agent signals "done"
        # [2:2+latent_dim]: delta_z ∈ [-1,1]^latent_dim
        action_dim = 2 + latent_dim
        self.action_space = spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
        )

        # ── Internal State ─────────────────────────────────────
        self._graph: nx.DiGraph | None = None
        self._graph_embedding = np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)
        self._targets: list[_TargetState] = []
        self._n_targets: int = 0
        self._current_mol_features = np.zeros(MOL_FEATURE_DIM, dtype=np.float32)
        self._step_count: int = 0
        self._episode_best_score: float = -np.inf
        self._episode_rewards: list[float] = []

        # Lazy-loaded components
        self._graph_encoder: Any = None
        self._reward_learner: Any = None
        self._surrogate: Any = None
        self._genmol_model: Any = None
        self._genmol_tokenizer: Any = None

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Reset the environment for a new drug discovery campaign.

        1. Build/load disease graph
        2. Identify causal targets
        3. Encode graph with R-GCN
        4. Initialise per-target state
        """
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_best_score = -np.inf
        self._episode_rewards = []

        # ── Build disease graph ────────────────────────────────
        self._graph = self._get_disease_graph()

        # ── Identify targets ──────────────────────────────────
        target_dicts = self._get_targets()
        self._n_targets = min(len(target_dicts), MAX_TARGETS)

        # ── Encode graph ──────────────────────────────────────
        self._encode_graph()

        # ── Initialise target states ──────────────────────────
        self._targets = []
        for i, tdict in enumerate(target_dicts[: self._n_targets]):
            ts = _TargetState(tdict, i)
            # Assign node embedding if available
            ts.z_base = self.np_random.standard_normal(self.latent_dim).astype(np.float32) * 0.5
            self._targets.append(ts)

        # ── Assign node embeddings to targets ─────────────────
        self._assign_target_embeddings()

        self._current_mol_features = np.zeros(MOL_FEATURE_DIM, dtype=np.float32)

        obs = self._build_observation()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Execute one generate-screen cycle.

        Action interpretation:
            action[0] — target selector (continuous → discretised)
            action[1] — stop signal (> 0.9 = early termination)
            action[2:] — delta-z for GenMol latent space navigation
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # ── Parse hierarchical action ─────────────────────────
        target_idx = self._select_target(float(action[0]))
        stop_signal = float(action[1]) > 0.9
        delta_z = action[2: 2 + self.latent_dim].astype(np.float32)

        target = self._targets[target_idx]
        target.n_attempts += 1

        # ── Generate molecule ─────────────────────────────────
        z = target.z_base + delta_z * 0.3  # scale delta for stability
        smiles = self._generate_from_latent(z)

        # ── Screen molecule ───────────────────────────────────
        obj_scores = self._screen_molecule(smiles, target)

        # ── Compute reward ────────────────────────────────────
        state_vec = self._build_observation()
        reward = self._compute_reward(state_vec, obj_scores)

        # ── Update target state ───────────────────────────────
        composite = sum(obj_scores.values()) / max(len(obj_scores), 1)
        if composite > target.best_score:
            target.best_score = composite
            target.best_smiles = smiles
            target.best_objectives = dict(obj_scores)

        if composite > self._episode_best_score:
            self._episode_best_score = composite

        # ── Update latent base (drift toward good regions) ────
        if composite > 0.5:
            target.z_base = 0.8 * target.z_base + 0.2 * z

        # ── Update mol features for next observation ──────────
        self._current_mol_features = self._encode_molecule(smiles, obj_scores)

        self._step_count += 1
        self._episode_rewards.append(reward)

        # ── Periodic surrogate recalibration ──────────────────
        if (
            self.use_surrogate
            and self._step_count % self.recalibrate_interval == 0
            and self._step_count > 0
        ):
            self._recalibrate_surrogate()

        # ── Termination ───────────────────────────────────────
        terminated = stop_signal and self._step_count >= 5  # min 5 steps
        truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = self._get_info()
        info["smiles"] = smiles
        info["target"] = target.gene_name
        info["objectives"] = obj_scores

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Causal Interface (shared with toy envs)                             #
    # ------------------------------------------------------------------ #

    def get_causal_graph(self) -> nx.DiGraph:
        """Return the disease causal graph (or empty graph if not built)."""
        if self._graph is not None:
            return self._graph
        return nx.DiGraph()

    # ------------------------------------------------------------------ #
    #  Internal: Disease Graph                                             #
    # ------------------------------------------------------------------ #

    def _get_disease_graph(self) -> nx.DiGraph:
        """Build or retrieve the disease knowledge graph."""
        if self._prebuilt_graph is not None:
            return self._prebuilt_graph

        try:
            from modules.neorx.graph_builder import build_disease_graph
            from modules.causalbiorl.causal.graph_encoder import disease_graph_to_networkx

            dg = build_disease_graph(self.disease)
            return disease_graph_to_networkx(dg)
        except Exception as e:
            logger.warning("Could not build disease graph: %s — using synthetic", e)
            return self._synthetic_graph()

    def _synthetic_graph(self) -> nx.DiGraph:
        """Create a small synthetic graph for testing/fallback."""
        G = nx.DiGraph()
        targets = [f"gene_{i}" for i in range(self.top_n_targets)]
        disease_node = f"disease_{self.disease}"
        G.add_node(disease_node, node_type="disease", score=1.0)

        for i, gene in enumerate(targets):
            G.add_node(gene, node_type="gene", score=0.3 + 0.1 * i)
            G.add_edge(gene, disease_node, edge_type="causes", weight=0.5 + 0.05 * i)
            # Add some inter-gene edges
            if i > 0:
                G.add_edge(targets[i - 1], gene, edge_type="regulates", weight=0.3)

        return G

    def _get_targets(self) -> list[dict[str, Any]]:
        """Identify causal targets or use pre-built list."""
        if self._prebuilt_targets is not None:
            return self._prebuilt_targets

        try:
            from modules.neorx.identifier import identify_causal_targets
            from modules.neorx.graph_builder import build_disease_graph

            dg = build_disease_graph(self.disease)
            results = identify_causal_targets(dg, top_n=self.top_n_targets)
            return [
                {
                    "gene_name": r.gene_name,
                    "protein_id": r.protein_id,
                    "protein_name": r.protein_name,
                    "causal_confidence": r.causal_confidence,
                    "pdb_ids": r.pdb_ids,
                    "is_causal": r.is_causal_target,
                    "target_type": r.target_type,
                    "node_id": r.protein_id,
                }
                for r in results
                if r.is_causal_target
            ]
        except Exception as e:
            logger.warning("Could not identify targets: %s — using synthetic", e)
            return self._synthetic_targets()

    def _synthetic_targets(self) -> list[dict[str, Any]]:
        """Create synthetic target list for testing."""
        return [
            {
                "gene_name": f"Gene{i}",
                "protein_id": f"P{i:05d}",
                "protein_name": f"Protein_{i}",
                "causal_confidence": 0.4 + 0.1 * i,
                "pdb_ids": [],
                "is_causal": True,
                "target_type": "CAUSAL",
                "node_id": f"gene_{i}",
            }
            for i in range(self.top_n_targets)
        ]

    # ------------------------------------------------------------------ #
    #  Internal: Graph Encoding                                            #
    # ------------------------------------------------------------------ #

    def _encode_graph(self) -> None:
        """Encode the disease graph using R-GCN."""
        if self._graph is None or len(self._graph) == 0:
            self._graph_embedding = np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)
            return

        try:
            if self._graph_encoder is None:
                from modules.causalbiorl.causal.graph_encoder import DiseaseGraphEncoder
                self._graph_encoder = DiseaseGraphEncoder(
                    embedding_dim=GRAPH_EMBEDDING_DIM,
                )
                self._graph_encoder.eval()

            with __import__("torch").no_grad():
                graph_emb, node_embs, node_order = self._graph_encoder.encode_disease_graph(
                    self._graph,
                )
                self._graph_embedding = graph_emb.cpu().numpy()
                self._node_embeddings = node_embs.cpu().numpy()
                self._node_order = node_order

        except Exception as e:
            logger.debug("R-GCN encoding failed: %s — using degree-based features", e)
            self._graph_embedding = self._fallback_graph_features()
            self._node_embeddings = np.zeros((len(self._graph), GRAPH_EMBEDDING_DIM), dtype=np.float32)
            self._node_order = list(self._graph.nodes())

    def _fallback_graph_features(self) -> NDArray[np.floating]:
        """Simple graph statistics as fallback embedding."""
        G = self._graph
        if G is None or len(G) == 0:
            return np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)

        features = np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)
        features[0] = len(G.nodes()) / 100.0
        features[1] = len(G.edges()) / 500.0
        features[2] = nx.density(G)

        try:
            pr = nx.pagerank(G, max_iter=50)
            top_pr = sorted(pr.values(), reverse=True)[:10]
            for i, v in enumerate(top_pr):
                features[3 + i] = v
        except Exception:
            pass

        return features

    def _assign_target_embeddings(self) -> None:
        """Assign R-GCN node embeddings to targets."""
        if not hasattr(self, "_node_order"):
            return

        node_to_idx = {n: i for i, n in enumerate(self._node_order)}
        for target in self._targets:
            node_id = target.target_info.get("node_id", "")
            if node_id in node_to_idx:
                idx = node_to_idx[node_id]
                target.node_embedding = self._node_embeddings[idx].copy()

    # ------------------------------------------------------------------ #
    #  Internal: Molecule Generation                                       #
    # ------------------------------------------------------------------ #

    def _generate_from_latent(
        self,
        z: NDArray[np.floating],
    ) -> str:
        """Generate a molecule from a latent vector using GenMol.

        Falls back to sampling from known drug-like scaffolds if
        GenMol is unavailable.
        """
        try:
            if self._genmol_model is None:
                self._init_genmol()

            if self._genmol_model is not None and self._genmol_tokenizer is not None:
                import torch
                z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    smiles_list = self._genmol_model.decode_from_latent(
                        z_tensor, self._genmol_tokenizer,
                    )
                if smiles_list:
                    return smiles_list[0]
        except Exception as e:
            logger.debug("GenMol decode failed: %s", e)

        return self._fallback_generate(z)

    def _init_genmol(self) -> None:
        """Lazy-initialise GenMol model."""
        try:
            from modules.genmol.data import SmilesTokenizer
            from modules.genmol.models import MolVAE

            self._genmol_tokenizer = SmilesTokenizer()
            self._genmol_model = MolVAE(
                vocab_size=max(self._genmol_tokenizer.vocab_size, 64),
            )
            self._genmol_model.eval()
        except Exception as e:
            logger.debug("Could not init GenMol: %s", e)
            self._genmol_model = None
            self._genmol_tokenizer = None

    def _fallback_generate(self, z: NDArray[np.floating]) -> str:
        """Deterministic fallback: select from scaffold library using z."""
        scaffolds = [
            "c1ccc2[nH]c(-c3ccncc3)nc2c1",
            "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1",
            "Cc1nc2ccccc2n1Cc1ccc(F)cc1",
            "O=C(c1ccc(O)cc1)c1ccc(O)cc1O",
            "CC(=O)Nc1ccc(O)cc1",
            "c1ccc(-c2nc3ccccc3s2)cc1",
            "O=c1[nH]c2ccccc2c2ccccc12",
            "NC(=O)c1cccc(-c2cccnc2)c1",
            "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1",
            "CC1=NN(c2ccccc2)C(=O)C1",
            "c1ccc(CNc2ncnc3[nH]cnc23)cc1",
            "CC(C)c1nnc(C(C)C)n1C1CC1c1ccc(F)cc1",
        ]
        # Use z to index into scaffolds (hash-based selection)
        idx = int(np.abs(z[:4].sum()) * 1000) % len(scaffolds)
        return scaffolds[idx]

    # ------------------------------------------------------------------ #
    #  Internal: Molecule Screening                                        #
    # ------------------------------------------------------------------ #

    def _screen_molecule(
        self,
        smiles: str,
        target: _TargetState,
    ) -> dict[str, float]:
        """Screen a molecule and return per-objective scores in [0, 1]."""
        scores: dict[str, float] = {}

        # ── Binding affinity ──────────────────────────────────
        scores["binding"] = self._get_binding_score(smiles, target)

        # ── QED ───────────────────────────────────────────────
        scores["qed"] = self._get_qed(smiles)

        # ── SA ────────────────────────────────────────────────
        scores["sa"] = self._get_sa(smiles)

        # ── Novelty ───────────────────────────────────────────
        scores["novelty"] = self._get_novelty(smiles)

        # ── Causal confidence ─────────────────────────────────
        scores["causal"] = target.causal_confidence

        # ── Stability (MirrorFold) ────────────────────────────
        scores["stability"] = self._get_stability(smiles, target)

        # Add noise for difficulty
        if self._noise_std > 0:
            for key in scores:
                scores[key] = float(np.clip(
                    scores[key] + self.np_random.normal(0, self._noise_std),
                    0.0, 1.0,
                ))

        return scores

    def _get_binding_score(self, smiles: str, target: _TargetState) -> float:
        """Get binding score via surrogate or real docking."""
        if self.use_surrogate:
            return self._surrogate_binding(smiles, target)
        return self._real_binding(smiles, target)

    def _surrogate_binding(self, smiles: str, target: _TargetState) -> float:
        """Fast binding estimate via surrogate model."""
        try:
            if self._surrogate is None:
                from modules.causalbiorl.causal.surrogate_docker import SurrogateDockingModel
                self._surrogate = SurrogateDockingModel()

            from modules.causalbiorl.causal.surrogate_docker import smiles_to_fingerprint
            from modules.causalbiorl.causal.reward_learner import normalise_binding

            fp = smiles_to_fingerprint(smiles)
            if fp is None:
                return 0.3

            affinity = self._surrogate.predict(fp, target.node_embedding)
            return normalise_binding(affinity)

        except Exception:
            return 0.3  # neutral prior

    def _real_binding(self, smiles: str, target: _TargetState) -> float:
        """Real docking via DockBot (slow but accurate)."""
        try:
            from modules.neorx.pipeline import _run_docking
            from modules.causalbiorl.causal.reward_learner import normalise_binding

            if not target.pdb_ids:
                return 0.3

            affinity = _run_docking(smiles, target.pdb_ids[0])
            if affinity is not None:
                return normalise_binding(affinity)
            return 0.3

        except Exception:
            return 0.3

    def _get_qed(self, smiles: str) -> float:
        """Get QED score."""
        try:
            from modules.molscreen.accessibility import qed_score
            q = qed_score(smiles)
            return float(q) if q is not None else 0.5
        except Exception:
            try:
                from rdkit import Chem
                from rdkit.Chem import QED
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return QED.qed(mol)
            except Exception:
                pass
            return 0.5

    def _get_sa(self, smiles: str) -> float:
        """Get normalised SA score (higher = easier to synthesise)."""
        try:
            from modules.molscreen.accessibility import sa_score
            from modules.causalbiorl.causal.reward_learner import normalise_sa
            sa = sa_score(smiles)
            return normalise_sa(sa)
        except Exception:
            return 0.5

    def _get_novelty(self, smiles: str) -> float:
        """Get structural novelty vs known drugs."""
        try:
            from modules.molscreen.similarity import find_similar_drugs
            similar = find_similar_drugs(smiles, top_k=1, threshold=0.3)
            if similar:
                max_sim = max(s[1] for s in similar) if similar else 0.0
                return 1.0 - max_sim
            return 0.9
        except Exception:
            return 0.5

    def _get_stability(self, smiles: str, target: _TargetState) -> float:
        """Get structural stability via MirrorFold therapeutic assessment."""
        try:
            from modules.mirrorfold import therapeutic_assessment

            # MirrorFold operates on protein sequences, not SMILES.
            # For drug discovery, we assess the target protein's
            # stability as a drug target (is it structurally tractable?)
            if target.pdb_ids:
                from modules.mirrorfold import fetch_pdb_structure, pdb_to_sequence
                pdb_text = fetch_pdb_structure(target.pdb_ids[0])
                if pdb_text:
                    seq = pdb_to_sequence(pdb_text)
                    if seq and len(seq) > 10:
                        result = therapeutic_assessment(seq[:500])  # cap length
                        return float(result.get("therapeutic_score", 0.5))
        except Exception:
            pass
        return 0.5  # neutral prior

    # ------------------------------------------------------------------ #
    #  Internal: Reward                                                    #
    # ------------------------------------------------------------------ #

    def _compute_reward(
        self,
        state: NDArray[np.floating],
        obj_scores: dict[str, float],
    ) -> float:
        """Compute adaptively-weighted reward."""
        try:
            if self._reward_learner is None:
                from modules.causalbiorl.causal.reward_learner import AdaptiveRewardLearner
                self._reward_learner = AdaptiveRewardLearner(
                    state_dim=OBS_DIM,
                )

            # Compute reward
            reward = self._reward_learner.compute_reward(state, obj_scores)

            # Update critic
            self._reward_learner.update(state, obj_scores)

            return reward

        except Exception:
            # Fallback: equal-weight sum
            return sum(obj_scores.values()) / max(len(obj_scores), 1)

    # ------------------------------------------------------------------ #
    #  Internal: Observation Building                                      #
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> NDArray[np.floating]:
        """Build the full observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # [0:128] Graph embedding
        obs[:GRAPH_EMBEDDING_DIM] = self._graph_embedding[:GRAPH_EMBEDDING_DIM]

        # [128:160] Current molecule features
        offset = GRAPH_EMBEDDING_DIM
        obs[offset: offset + MOL_FEATURE_DIM] = self._current_mol_features

        # [160:240] Per-target summaries (10 targets × 8 features)
        offset += MOL_FEATURE_DIM
        for i, target in enumerate(self._targets):
            start = offset + i * TARGET_FEATURE_DIM
            end = start + TARGET_FEATURE_DIM
            obs[start:end] = target.summary_features()

        # [240:244] Meta features
        meta_offset = offset + MAX_TARGETS * TARGET_FEATURE_DIM
        obs[meta_offset] = self._step_count / max(self.max_steps, 1)
        obs[meta_offset + 1] = self._n_targets / MAX_TARGETS
        obs[meta_offset + 2] = max(self._episode_best_score, 0.0)
        obs[meta_offset + 3] = float(self.use_surrogate)

        return obs

    def _encode_molecule(
        self,
        smiles: str,
        obj_scores: dict[str, float],
    ) -> NDArray[np.floating]:
        """Encode a molecule into a 32-D feature vector."""
        features = np.zeros(MOL_FEATURE_DIM, dtype=np.float32)

        # Objective scores (6 values)
        for i, name in enumerate(["binding", "qed", "sa", "novelty", "causal", "stability"]):
            if i < MOL_FEATURE_DIM:
                features[i] = obj_scores.get(name, 0.0)

        # Molecular fingerprint summary (top 26 bits of Morgan fp)
        try:
            from modules.causalbiorl.causal.surrogate_docker import smiles_to_fingerprint
            fp = smiles_to_fingerprint(smiles, n_bits=256)
            if fp is not None:
                # Compress 256-bit fp to 26 values via chunked sums
                chunk_size = 256 // 26
                for i in range(26):
                    start = i * chunk_size
                    end = min(start + chunk_size, 256)
                    features[6 + i] = fp[start:end].sum() / chunk_size
        except Exception:
            pass

        return features

    # ------------------------------------------------------------------ #
    #  Internal: Target Selection                                          #
    # ------------------------------------------------------------------ #

    def _select_target(self, action_value: float) -> int:
        """Convert continuous action to discrete target index."""
        if self._n_targets == 0:
            return 0

        # Map [-1, 1] → [0, n_targets-1]
        normalised = (action_value + 1.0) / 2.0  # → [0, 1]
        idx = int(normalised * self._n_targets)
        return max(0, min(idx, self._n_targets - 1))

    # ------------------------------------------------------------------ #
    #  Internal: Surrogate Recalibration                                   #
    # ------------------------------------------------------------------ #

    def _recalibrate_surrogate(self) -> None:
        """Run real docking on recent molecules to recalibrate surrogate."""
        if self._surrogate is None:
            return

        try:
            from modules.causalbiorl.causal.surrogate_docker import smiles_to_fingerprint
            from modules.causalbiorl.causal.reward_learner import normalise_binding

            recal_count = 0
            for target in self._targets:
                if target.best_smiles and target.pdb_ids:
                    real_aff = self._real_binding(target.best_smiles, target)
                    fp = smiles_to_fingerprint(target.best_smiles)
                    if fp is not None:
                        # Convert normalised score back to approx affinity
                        self._surrogate.add_observation(
                            fp, target.node_embedding, -real_aff * 12.0,
                        )
                        recal_count += 1

            if recal_count > 0 and self._surrogate.buffer_size >= 10:
                self._surrogate.fit(epochs=50)
                logger.debug(
                    "Surrogate recalibrated with %d new observations.", recal_count,
                )

        except Exception as e:
            logger.debug("Surrogate recalibration failed: %s", e)

    # ------------------------------------------------------------------ #
    #  Info & Rendering                                                    #
    # ------------------------------------------------------------------ #

    def _get_info(self) -> dict[str, Any]:
        """Return episode info dict."""
        target_summary = {}
        for t in self._targets:
            target_summary[t.gene_name] = {
                "n_attempts": t.n_attempts,
                "best_score": float(t.best_score) if t.best_score > -np.inf else 0.0,
                "best_smiles": t.best_smiles,
            }

        return {
            "step": self._step_count,
            "n_targets": self._n_targets,
            "budget_remaining": self.max_steps - self._step_count,
            "episode_best_score": float(self._episode_best_score) if self._episode_best_score > -np.inf else 0.0,
            "targets": target_summary,
        }

    def render(self) -> None:
        """Print current episode status."""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"\n─── DrugDiscovery Step {info['step']}/{self.max_steps} ───")
            print(f"  Disease: {self.disease}")
            print(f"  Best score: {info['episode_best_score']:.3f}")
            print(f"  Budget remaining: {info['budget_remaining']}")
            for gene, data in info["targets"].items():
                print(
                    f"  {gene}: {data['n_attempts']} attempts, "
                    f"best={data['best_score']:.3f}"
                )

    def close(self) -> None:
        """Clean up resources."""
        self._graph = None
        self._graph_encoder = None
        self._surrogate = None
        self._genmol_model = None
        self._genmol_tokenizer = None
