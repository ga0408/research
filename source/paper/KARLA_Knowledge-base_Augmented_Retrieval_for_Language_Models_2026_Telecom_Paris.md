# KARLA: Knowledge-base Augmented Retrieval for Language Models — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_KARLA_Knowledge-base_Augmented_Retrieval_for_Language_Models_2026_Telecom_Paris.md) / 원본: [arXiv:2606.26807](https://arxiv.org/abs/2606.26807)

---

## 1. Problem Formulation & Core Concept

> "We are given a pre-trained language model and a knowledge base (KB). Formally, a KB is a set of triples of the form $\langle s, r, o \rangle$, where $ is a subject entity, $ is a relation (or predicate) from a closed set $\mathcal{R}$, and $ is the corresponding object, as in $\langle \text{Paris}, \text{populationTotal}, \text{2,047,602} \rangle$. Our objective is to fine-tune the model so that it interleaves natural-language generation with inline queries of the form $\langle s, r, ? \rangle$. These inline queries are then replaced by an object $ for which $\langle s, r, o \rangle$ is in the KB." (§3.1)

> "Our goal is to separate linguistic competence from factual knowledge: the language model handles interpretation and generation, while the KB provides the atomic factual values. This brings several advantages:
> 1. Factual knowledge can be updated in the KB at virtually no cost, without retraining the model.
> 2. Facts in the output can be traced back to the KB, providing explainability and provenance.
> 3. Smaller models can match the factual precision of larger models, as factual capacity is externalized." (§1)

---

## 2. Inline Query Syntax & Predicate Embedding Initialization

> "In practice, the inline query $\langle s, r, ? \rangle$ with result $ is expressed as
> 3526\langle r \rangle \langle \text{subj} \rangle s \langle /\text{subj} \rangle \langle \text{KB} \rangle o : o_{\text{desc}} \langle /\text{KB} \rangle3526
> In this sequence, $\langle r \rangle$ is a relation-specific trigger token. The tag $\langle \text{subj} \rangle$ marks the subject, which is used to query the KB. The model will be trained to identify the subject here from the preceding part of the sentence. The tag $\langle \text{KB} \rangle$ marks the answer to the query that was retrieved from the KB. It is included in the output sequence of tokens as soon as the inline query is generated, so as to allow the model to condition the following tokens on $. The text {\text{desc}}$ is a short description of the object as retrieved from the KB (e.g., by help of the relation schema:description)." (§3.1)

> "We represent each relation as an atomic special token $\langle r \rangle$ rather than as a plain-text label. This has the advantage that the model cannot generate relations that do not exist in the KB. It also reduces per-call inference cost, because each retrieval action emits one token rather than the several sub-tokens of a plain-text predicate label. To speed up convergence, we initialize the new embeddings from the base tokenizer. Let  \in \mathbb{R}^{|\mathcal{V}| \times d}$ denote the embedding matrix of the original tokenizer where $\mathcal{V}$ is the vocabulary set and $ is the embedding dimension. For a relation $, we define the embedding of the corresponding predicate token {\langle r \rangle}$ as the mean of its constituent sub-token embeddings:
> 3526E_{\langle r \rangle} = \frac{1}{|t(r)|} \sum_{w \in t(r)} E_w3526
> where (r)$ is the sub-token decomposition of predicate label $." (§3.1)

---

## 3. Subgraph Sampling Algorithm & Proposition 1

> "Algorithm 1: Subgraph sampling algorithm.
> Input: Knowledge base $\text{KB}$, relation set $\mathcal{R}$, per-relation budget $, relations per sample $.
> Output: Training set of entity subgraphs $\mathcal{S}$.
> 1: counts 0 \leftarrow 0$ for all  \in \mathcal{R}$
> 2: $\mathcal{S} \leftarrow \emptyset$
> 3: while $\exists r \in \mathcal{R}$ such that counts 0 < T$ do
> 4:   Sample entity  \sim \text{Entities}(\text{KB})$ uniformly with replacement
> 5:   $\mathcal{F}(e) \leftarrow \{\langle e, r, o \rangle \in \text{KB} \mid \text{counts}[r] < 2T\}$
> 6:   $\mathcal{F}_{\text{valid}}(e) \leftarrow \text{deduplicate}(\mathcal{F}(e))$
> 7:   if $\mathcal{F}_{\text{valid}}(e) \neq \emptyset$ then
> 8:     $\mathcal{F}_{\text{sample}} \leftarrow \text{sample } \min(k, |\mathcal{F}_{\text{valid}}(e)|) \text{ facts from } \mathcal{F}_{\text{valid}}(e)$
> 9:     $\mathcal{S} \leftarrow \mathcal{S} \cup \{(e, \mathcal{F}_{\text{sample}})\}$
> 10:    for $\langle e, r, o \rangle \in \mathcal{F}_{\text{sample}}$ do
> 11:      counts 0 \leftarrow \text{counts}[r] + 1$
> 12: return $\mathcal{S}$" (Appendix B)

> **Proposition 1 (Relation Budget Bounds)**:
> "Let $\mathcal{S}$ be the output of Algorithm 1. For every relation  \in \mathcal{R}$, the final count satisfies:
> 3526T \le \text{counts}[r] < 2T3526
> Proof: Since $\mathcal{R}$ is constructed from the KB, every relation has at least one supporting triple, so $\mathcal{E}(r) \neq \emptyset$ for all  \in \mathcal{R}$.
> Lower bound: The while loop terminates only when counts 0 \ge T$ for all  \in \mathcal{R}$. Thus counts 0 \ge T$.
> Upper bound: Line 5 excludes any relation with counts 0 \ge 2T$ from candidate selection. Since counts 0$ is incremented by at most 1 per sample step, the condition counts 0 < 2T$ ensures that counts 0$ cannot reach or exceed T$ before exclusion." (Appendix B)

---

## 4. Masked Loss Formulation

> "To enforce the separation of natural language and KB knowledge, we employ a masked next-token prediction loss. The loss is defined as:
> 3526\mathcal{L}(\theta) = - \sum_{t=1}^M m_t \cdot \log p_\theta (\tilde{x}_t \mid \tilde{x}_{<t})3526
> The binary mask  \in \{0, 1\}$ is constructed to gate the gradient flow:
> 3526m_t = \begin{cases} 0 & \text{if } \tilde{x}_t \in \text{span}(\langle \text{KB} \rangle, \dots, \langle /\text{KB} \rangle) \\ 1 & \text{otherwise} \end{cases}3526
> By setting  = 0$ for tokens within the retrieval-result span, the model is not penalized for failing to predict the returned object $ from the context alone. Instead, supervision is concentrated on the surrounding natural language and on the retrieval query that obtains the relevant KB evidence. This reduces direct supervision pressure to store KB facts in the model parameters and encourages the model to condition factual generation on the retrieved KB object." (§3.3)

> "Since training covers only a small subset of the KB, the model must learn to handle cases where the KB contains no matching fact. In 10% of inline queries (sampled independently), we replace the successful lookup result with the failure token $\langle \text{KB\_FAIL} \rangle$. This teaches the model to recover from empty or unresolved KB queries by falling back to its parametric knowledge." (§3.3)

---

## 5. Target-Normalized Masked Perplexity ({aug}$)

> "For KARLA models, we report target-normalized masked perplexity. Let  \in \{0, 1\}$ denote a scoring mask over the augmented sequence. The mask is set to zero for KB-returned objects, since these tokens are supplied by the external executor rather than predicted by the model. It is set to one for all tokens generated by the model, including text tokens and query tokens. We define:
> 3526\text{PPL}_{\text{aug}}(\tilde{x}) = \exp\left(-\frac{1}{N} \sum_{j=1}^M m_j \log p_\theta(\tilde{x}_j \mid \tilde{x}_{<j})\right)3526
> This is not the ordinary perplexity of the augmented sequence: the denominator is the number of tokens $ in the non-augmented sequence, not the number of scored tokens nor the full augmented length $. The metric thus measures the description length of the original passage while charging the model for the additional inline query tokens. A lower score means that the evidence obtained through KB execution reduces uncertainty enough to outweigh the overhead of producing the KB query." (§4.2)

---

## 6. Key Empirical Results

### Table 1: Perplexity on Synthetic Held-out Set
| Model | Setup | YAGO ($) | PrimeKG ($) |
| :--- | :--- | :---: | :---: |
| Qwen 0.6B | KARLA | **7.09** | **3.96** |
| Qwen 0.6B | KARLA-empty-KB | 9.27 | 5.61 |
| Qwen 0.6B | KARLA-raw | 11.17 | 6.38 |
| Qwen 0.6B | Raw-text SFT | 8.79 | 5.08 |
| Qwen 1.7B | KARLA | **6.05** | **3.36** |
| Qwen 1.7B | Raw-text SFT | 7.28 | 4.28 |
| Qwen 4B | KARLA | **5.32** | **2.96** |
| Qwen 4B | Raw-text SFT | 6.27 | 3.75 |
| Qwen 8B | KARLA | **5.08** | **2.84** |
| Qwen 8B | Raw-text SFT | 5.77 | 3.51 |

### Table 2: Inline-Query Exact-Match Accuracy (%)
| Setup | Model | YAGO Subj | YAGO Rel | YAGO Both | PrimeKG Subj | PrimeKG Rel | PrimeKG Both |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| KARLA | Qwen 0.6B | 99.4 | 87.2 | 86.7 | 96.9 | 94.7 | 91.8 |
| KARLA | Qwen 1.7B | 99.6 | 89.9 | 89.5 | 98.1 | 96.6 | 94.9 |
| KARLA | Qwen 4B | 99.7 | 91.2 | 91.0 | 99.2 | 97.8 | 97.0 |
| KARLA | Qwen 8B | 99.7 | 88.1 | 87.9 | 99.4 | 95.3 | 94.8 |

### Table 3: Short-Form & Long-Form Factuality
| Model | Setup | PopQA (Acc %) | FActScore (%) |
| :--- | :--- | :---: | :---: |
| Qwen 0.6B | KARLA | **78.56** | 53.0 |
| Qwen 0.6B | Base LM | 16.37 | 22.78 |
| Qwen 0.6B | 1-hop graph RAG | 54.45 | 53.1 |
| Qwen 0.6B | Tool-schema prompt | 15.37 | 24.4 |
| Qwen 4B | KARLA | **80.91** | **58.9** |
| Qwen 4B | Base LM | 23.41 | 24.16 |
| Qwen 4B | 1-hop graph RAG | 56.17 | 56.8 |
| Qwen 4B | Tool-schema prompt | 41.74 | 30.0 |
| Qwen 8B | KARLA | **80.63** | 57.3 |
| Qwen 8B | 1-hop graph RAG | 58.68 | 58.2 |
| LLAMA2-382M | LMLM | 52.00 | 23.9 |
| GPT2-774M | LMLM | 50.80 | 31.9 |

### Table 4: COUNTERFACTUAL YAGO Factual Overriding by Popularity Quartile
| Setup | Q1 (Least Popular) | Q2 | Q3 | Q4 (Most Popular) | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KARLA (Qwen3 4B)** | **97.3%** | **95.9%** | **95.6%** | **95.6%** | **96.11%** (Zero-shot KB Swap) |
| 1-hop graph RAG (Qwen3 4B) | 89.1% | 93.2% | 92.3% | 77.8% | 87.30% (Parametric override) |
| Parametric LoRA SFT (Qwen3 4B) | Requires 1000+ gradient update steps to converge (Fig. 3) |

---

## 7. Qualitative Failure Modes in Baseline Graph RAG

> "Table 14: Example of parametric override by the 1-hop graph RAG baseline on COUNTERFACTUAL YAGO.
> Bulgaria’s facts have been replaced by Slovakia’s in the rewritten KB, so the ground-truth demonym is the value present in the retrieved context (Slūfākiyyūn, 'Slovaks'). The model instead emits its memorized answer for the original entity, ignoring the retrieved Demonym field.
> Expected answer: Slūfākiyyūn
> Model output: Bulgarians." (Appendix H)

> "Table 15: Example of confident fabrication by the 1-hop graph RAG baseline on COUNTERFACTUAL YAGO.
> Bulgaria’s Date Created field is explicitly present in the retrieved context as 1 January 1993, yet the model generates 1 January 1946 — a date matching neither the retrieved value nor the real-world creation of modern Bulgaria, indicating that the relevant field is not consulted before generation.
> Expected answer: 1 January 1993
> Model output: 1 January 1946." (Appendix H)
