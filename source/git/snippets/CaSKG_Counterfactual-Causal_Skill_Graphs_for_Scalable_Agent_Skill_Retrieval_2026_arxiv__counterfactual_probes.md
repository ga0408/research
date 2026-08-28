# CaSKG Direction-Conditioned Textual Counterfactual Probes

> 분석 문서: [report/[paper][git]_CaSKG_Counterfactual-Causal_Skill_Graphs_for_Scalable_Agent_Skill_Retrieval_2026_arxiv.md](../../report/[paper][git]_CaSKG_Counterfactual-Causal_Skill_Graphs_for_Scalable_Agent_Skill_Retrieval_2026_arxiv.md)
> 원본: [GitHub ZhiyuanLi218/Caskg](https://github.com/ZhiyuanLi218/Caskg) · [arXiv:2608.25500](https://arxiv.org/abs/2608.25500)

```python
"""
Prompt Templates and Evaluator for Direction-Conditioned Counterfactual Probes
- Removal Probe: Necessity (필요성)
- Substitution Probe: Specificity (특이성)
- Reordering Probe: Directionality (순서 의존성)
"""

PROMPT_REMOVAL_PROBE = """
You are an expert evaluator assessing procedural dependencies between agent skills.
Evaluate whether Skill B strictly requires Skill A to be executed first.

Candidate Relation: Skill A -> Skill B
Skill A (Source): {skill_a_name}: {skill_a_desc}
Skill B (Target): {skill_b_name}: {skill_b_desc}

[Counterfactual Test: Source Removal]
Suppose Skill A is UNAVAILABLE in the environment or has NOT been executed.
Can the agent successfully and safely execute Skill B without prior execution of Skill A?

Provide your reasoning and a support score between 0.0 and 1.0:
- 1.0: Skill B is completely impossible / invalid without Skill A (Strong Necessity).
- 0.5: Skill B can partially function or has trivial alternatives.
- 0.0: Skill B is completely independent of Skill A.

Output format:
Reasoning: <explanation>
Score: <float between 0.0 and 1.0>
"""

PROMPT_SUBSTITUTION_PROBE = """
You are an expert evaluator assessing procedural dependencies between agent skills.
Evaluate whether Skill B specifically depends on Skill A rather than a generic substitute.

Candidate Relation: Skill A -> Skill B
Skill A (Source): {skill_a_name}: {skill_a_desc}
Skill B (Target): {skill_b_name}: {skill_b_desc}
Substitute Skill (A~): {substitute_a_name}: {substitute_a_desc}

[Counterfactual Test: Source Substitution]
Suppose Skill A is replaced by Substitute Skill A~.
Does the downstream execution of Skill B fail, degrade, or lose required preconditions?

Provide your reasoning and a support score between 0.0 and 1.0:
- 1.0: Replacing Skill A severely breaks Skill B (High Specificity).
- 0.5: Replacing Skill A causes moderate degradation.
- 0.0: Any generic skill or substitute works equally well.

Output format:
Reasoning: <explanation>
Score: <float between 0.0 and 1.0>
"""

PROMPT_REORDERING_PROBE = """
You are an expert evaluator assessing procedural dependencies between agent skills.
Evaluate whether the directional order (Skill A THEN Skill B) is mandatory.

Proposed Direction: Execute Skill A -> Then Execute Skill B
Skill A: {skill_a_name}: {skill_a_desc}
Skill B: {skill_b_name}: {skill_b_desc}

[Counterfactual Test: Order Reversal]
Suppose the execution order is REVERSED: Execute Skill B FIRST, then Skill A.
Does this reversed workflow violate physical laws, preconditions, or logical causal sequence?

Provide your reasoning and a support score between 0.0 and 1.0:
- 1.0: Executing B before A is completely invalid/impossible (Strict Order Dependence).
- 0.5: Executing B before A is suboptimal but partially coherent.
- 0.0: The execution order does not matter at all (Symmetric / Commutative).

Output format:
Reasoning: <explanation>
Score: <float between 0.0 and 1.0>
"""
```
