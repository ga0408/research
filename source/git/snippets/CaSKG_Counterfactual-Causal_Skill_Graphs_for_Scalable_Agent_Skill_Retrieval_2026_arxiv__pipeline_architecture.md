# CaSKG Pipeline Architecture & Retrieval Diffusion Pseudo-Implementation

> 분석 문서: [report/[paper][git]_CaSKG_Counterfactual-Causal_Skill_Graphs_for_Scalable_Agent_Skill_Retrieval_2026_arxiv.md](../../report/[paper][git]_CaSKG_Counterfactual-Causal_Skill_Graphs_for_Scalable_Agent_Skill_Retrieval_2026_arxiv.md)
> 원본: [GitHub ZhiyuanLi218/Caskg](https://github.com/ZhiyuanLi218/Caskg) · [arXiv:2608.25500](https://arxiv.org/abs/2608.25500)

```python
"""
CaSKG: Counterfactual-Causal Skill Graphs for Scalable Agent Skill Retrieval
Core Offline Graph Construction & Task-Conditioned Diffusion Engine
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    role: str  # e.g., "preparation", "operation", "observation", "completion", "recovery"
    embedding: np.ndarray

class CaSKGPipeline:
    def __init__(
        self,
        skill_library: List[Skill],
        tau_str: float = 0.6,
        eta_str: float = 0.85,
        tau_c: float = 0.70,
        rho_unc: float = 0.35,
        rho_scaf: float = 0.10,
        eps_e: float = 0.05,
        eps_w: float = 0.02,
        restart_gamma: float = 0.15,
    ):
        self.skills = skill_library
        self.n = len(skill_library)
        self.skill_map = {s.skill_id: i for i, s in enumerate(skill_library)}
        
        # Hyperparameters
        self.tau_str = tau_str
        self.eta_str = eta_str
        self.tau_c = tau_c
        self.rho_unc = rho_unc
        self.rho_scaf = rho_scaf
        self.eps_e = eps_e
        self.eps_w = eps_w
        self.gamma = restart_gamma

    def stage1_candidate_induction(self, top_k_neighbors: int = 10) -> Tuple[Dict[Tuple[int, int], float], List[Tuple[int, int]]]:
        """
        Stage 1: Multi-source heterogeneous candidate induction with high recall.
        """
        candidate_weights: Dict[Tuple[int, int], float] = {}
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                s_i, s_j = self.skills[i], self.skills[j]
                
                # 1. Semantic cosine similarity
                phi_sem = float(np.dot(s_i.embedding, s_j.embedding) / (
                    np.linalg.norm(s_i.embedding) * np.linalg.norm(s_j.embedding) + 1e-9
                ))
                # 2. Input/Output interface match
                common_io = set(s_i.outputs).intersection(set(s_j.inputs))
                phi_io = len(common_io) / max(len(s_j.inputs), 1) if s_j.inputs else 0.0
                
                # 3. Structural role affinity (preparation -> operation -> observation -> completion)
                phi_struct = self._compute_role_affinity(s_i.role, s_j.role)
                
                # Active signal aggregation (Eq. 1)
                signals = [phi_sem, phi_io, phi_struct]
                weights = [0.4, 0.3, 0.3]
                
                a_tilde = np.clip(sum(w * s for w, s in zip(weights, signals)) / sum(weights), 0.0, 1.0)
                
                if phi_struct > self.tau_str:
                    a_ij = max(a_tilde, self.eta_str * phi_struct)
                else:
                    a_ij = a_tilde
                    
                if a_ij > self.eps_w:
                    candidate_weights[(i, j)] = a_ij
                    
        # Budgeted validation frontier selection (|F| = budgeted top pairs)
        sorted_candidates = sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True)
        validation_frontier = [pair for pair, score in sorted_candidates[:500]]
        return candidate_weights, validation_frontier

    def _compute_role_affinity(self, role_i: str, role_j: str) -> float:
        valid_transitions = {
            ("preparation", "operation"): 0.95,
            ("operation", "observation"): 0.90,
            ("observation", "completion"): 0.95,
            ("operation", "recovery"): 0.85,
            ("observation", "recovery"): 0.80,
        }
        return valid_transitions.get((role_i, role_j), 0.10)

    def stage3_bayesian_calibration(
        self,
        candidate_weights: Dict[Tuple[int, int], float],
        probe_results: Dict[Tuple[int, int], Dict[str, float]],
        scaffold_edges: Set[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Stage 3: Beta-form evidence accumulation & State-gated graph publication (Eq. 3-8).
        """
        W_pub = np.zeros((self.n, self.n), dtype=np.float32)
        
        for (i, j), a_ij in candidate_weights.items():
            if (i, j) in probe_results:
                probes = probe_results[(i, j)] # e_rem, e_sub, e_ord
                alpha_ij = 1.0
                beta_ij = 1.0
                
                for m in ["rem", "sub", "ord"]:
                    e_m = probes[m]
                    z_m = 1.0 if e_m > 0.5 else 0.0
                    delta_m = max(2.0 * abs(e_m - 0.5), self.eps_e)
                    alpha_ij += z_m * delta_m
                    beta_ij += (1.0 - z_m) * delta_m
                    
                c_ij = alpha_ij / (alpha_ij + beta_ij)
                
                # State assignment (Eq. 6)
                if c_ij > self.tau_c:
                    state = "confirmed"
                    rho_ij = 1.0
                elif c_ij < (1.0 - self.tau_c):
                    state = "rejected"
                    rho_ij = 0.0
                else:
                    state = "uncertain"
                    rho_ij = self.rho_unc
            else:
                c_ij = 0.0
                state = "unvalidated"
                rho_ij = self.rho_scaf if (i, j) in scaffold_edges else 0.0
                
            b_ij = max(a_ij, c_ij, self.eps_w)
            
            if rho_ij > 0.0:
                W_pub[i, j] = float(np.clip(rho_ij * b_ij, self.eps_w, 1.0))
            else:
                W_pub[i, j] = 0.0
                
        return W_pub

    def stage4_task_retrieval(
        self,
        query_embedding: np.ndarray,
        W_pub: np.ndarray,
        top_k: int = 5,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> List[int]:
        """
        Stage 4: Task-conditioned Personalized PageRank diffusion (Eq. 9).
        """
        # 1. Compute seed affinities
        sims = np.array([
            float(np.dot(query_embedding, s.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(s.embedding) + 1e-9
            ))
            for s in self.skills
        ])
        
        # Inverse-rank-weighted seed distribution pi_q
        ranks = np.argsort(np.argsort(-sims)) + 1
        raw_weights = 1.0 / ranks
        pi_q = raw_weights / np.sum(raw_weights)
        
        # 2. Row-normalize W_pub to obtain transition matrix T
        row_sums = W_pub.sum(axis=1, keepdims=True)
        T = np.divide(W_pub, row_sums, out=np.zeros_like(W_pub), where=row_sums != 0)
        # Dangling nodes transition uniformly to seed distribution pi_q
        dangling_mask = (row_sums.flatten() == 0)
        
        # 3. Power iteration for Personalized PageRank
        p = pi_q.copy()
        for _ in range(max_iter):
            p_next = self.gamma * pi_q + (1.0 - self.gamma) * (
                np.dot(T.T, p) + np.sum(p[dangling_mask]) * pi_q
            )
            if np.linalg.norm(p_next - p, 1) < tol:
                break
            p = p_next
            
        return list(np.argsort(-p)[:top_k])
```
