# Literature review: RLVR side effects on non-verifiable self-report (Olmo-3-7B-Think, SFT → DPO → RLVR)

Written Oct 25, 2026, after the main analysis and the exploratory addenda were complete. It is meant to test our claims against prior work, not to support them.

## How this review was done, and how to read it

- **Search.** About 55 web searches (a general web search engine, plus targeted arXiv / ACL Anthology / NeurIPS / ICLR listings) across the 12 areas listed in the brief, in October 2026.
- **Verification rule.** A paper is cited as **[V]** only if I fetched its arXiv abstract page (or, for one workshop paper, its official conference page) and read at least the abstract. Titles, authors, dates and arXiv IDs of [V] items come from that page.
- **Venues.** A venue is stated only if the fetched page lists it (comments / journal-ref field or the conference page). Where the arXiv page lists no venue but a proceedings page appeared in search results, I write "(venue listing seen in search, not opened)". Otherwise the item is given as an arXiv preprint.
- **Depth.** Almost everything below was read at **abstract level only**. Claims about what a paper does *not* contain are claims about its abstract and are flagged as such. A full-text reading could change several "how close" judgements in either direction.
- **[U] items** (recalled, or seen only as a search snippet or blog) are listed in the last section and **are not used to support any conclusion**.
- **"None found"** means a search was run and nothing relevant was found. It does not mean nothing exists.
- **One important gap:** the Olmo 3 technical report's post-training (Think SFT / DPO / RLVR) sections could not be read, because the HTML render was truncated. Whether the model developers themselves document per-stage changes in response length is therefore **unverified**. It must be checked before any novelty claim about C1 or C3 is made (see "Open checks").

---

## 1. Length and verbosity bias in RLHF and preference optimisation

- **[V]** Singhal, Goyal, Xu, et al. "A Long Way to Go: Investigating Length Correlations in RLHF." COLM 2024. arXiv:2310.03716. Finds that much of RLHF's measured gain is explained by longer outputs, that a purely length-based reward reproduces most of the improvement, and that reward models are the main source of the length bias.
- **[V]** Park, Rafailov, Ermon, Finn. "Disentangling Length from Quality in Direct Preference Optimization." arXiv:2403.19159 (2024). Shows significant length exploitation **in DPO**, linked to out-of-distribution bootstrapping, and proposes a length regulariser.
- **[V]** Chen, Zhu, Soselia, et al. "ODIN: Disentangled Reward Mitigates Hacking in RLHF." arXiv:2402.07319 (2024). Policies exploit reward models by producing long answers; ODIN splits the reward into a length head and a length-independent head.
- **[V]** Dubois, Galambosi, Liang, et al. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." COLM 2024. arXiv:2404.04475. LLM judges reward verbosity; the paper corrects for it by regression.
- **[V]** Liu, Chen, Li, et al. "Understanding R1-Zero-Like Training: A Critical Perspective." arXiv:2503.20783 (2025). Identifies an optimisation bias in **GRPO that inflates response length, especially for incorrect outputs**; proposes Dr. GRPO.
- **[V]** Fatemi, Rafiee, Tang, et al. "Concise Reasoning via Reinforcement Learning." arXiv:2504.05185 (2025). Argues that reasoning models' verbosity is a product of RL dynamics: incorrect answers push PPO/GRPO policies toward longer outputs, proven formally. Reports that conciseness correlates with correctness.
- **[V]** Zhang, Das, Zhang. "Verbosity ≠ Veracity: Demystify Verbosity Compensation Behavior of Large Language Models." arXiv:2411.07858 (2024). Models pad answers when uncertain; verbose answers are less accurate and more uncertain (14 models, 5 QA datasets).

**Bearing.** Length growth under both preference optimisation and RL is well documented and has mechanistic explanations: reward-model length bias for RLHF/DPO, and loss-normalisation and negative-advantage dynamics for PPO/GRPO. A DPO step lengthening outputs (our C3) and an RL step lengthening outputs (C1) are therefore **expected a priori**. What these papers do not do, judging from abstracts, is measure length on self-report prompts or compare consecutive stages of one lineage.

## 2. Overthinking and inefficient test-time compute in reasoning models

- **[V]** Chen, Xu, Liang, et al. "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs." arXiv:2412.21187 (2024). The first broad study of overthinking: o1-style models spend excess compute on easy problems for little gain.
- **[V]** Fan, Li, Sun, Zhou. "Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill?" arXiv:2504.06514 (2025). Reasoning models, **whether trained by RL or SFT**, produce drastically longer responses on ill-posed questions with a missing premise, i.e. questions with **no determinate answer**. Non-reasoning models respond briefly. The extra reasoning is redundant.
- **[V]** Aggarwal, Kim, Lanchantin, Welleck, Weston, Kulikov, Saha. "OptimalThinkingBench: Evaluating Over and Underthinking in LLMs." arXiv:2508.13141 (2025; NeurIPS/ICLR listings seen in search, not opened). Thinking models "overthink for hundreds of tokens" on simple math **and general queries across 72 domains** without improving performance. 33 models; none thinks optimally.
- **[V]** Wang, Liu, Xu, et al. "Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs." arXiv:2501.18585 (2025). o1-like models **switch between lines of reasoning** frequently. Switching correlates with incorrect answers and with more tokens.
- **[V]** Peng, Ding, Ouyang, et al. "Revisiting Overthinking in Long Chain-of-Thought from the Perspective of Self-Doubt." arXiv:2505.23480 (2025). Overthinking is largely self-doubt: re-verifying answers already reached.
- **[V]** Su, Healey, Nakov, et al. "Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and Correctness in LLMs." arXiv:2505.00127 (2025). Length and correctness are non-monotonic: models overthink easy problems and underthink hard ones.
- **[V]** Marjanović, Patel, Adlakha, et al. "DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning." TMLR (per arXiv comments). arXiv:2504.07128 (2025/2026). R1 has a reasoning "sweet spot", persistently ruminates on formulations it has already explored, and shows weaker safety than its non-reasoning counterpart. The abstract mentions cultural concerns but not length on subjective questions.
- **[V]** Gema, Hägele, Chen, et al. "Inverse Scaling in Test-Time Compute." TMLR 12/2025 (per arXiv journal-ref). arXiv:2507.14417. Longer reasoning lowers accuracy on constructed tasks, and **extended reasoning amplifies expressions of self-preservation** (Claude Sonnet 4). This is the nearest published link between reasoning length and **self-referential statements** found.
- **[V]** Pipis, Garg, Kontonis, et al. "Wait, Wait, Wait... Why Do Reasoning Models Loop?" ICML 2026 Spotlight (per arXiv). arXiv:2512.12895. Reasoning models loop, especially distilled students and smaller models, which the authors attribute to errors in learning.
- **[V]** Hosseini, Nawathe, Salmani, et al. "Early Stopping for Large Reasoning Models via Confidence Dynamics." arXiv:2604.04930 (2026). Incorrect reasoning paths produce long, unproductive traces. Uses intermediate-answer confidence to stop early.

**Bearing.** "Reasoning models spend many more tokens than needed on questions where extra thought cannot help" is established, including on **non-math general queries** (OptimalThinkingBench) and on **questions with no determinate answer** (MiP). Self-report Likert items were not studied in any abstract found.

## 3. RLVR: effects outside the training domain and side effects

- **[V]** DeepSeek-AI et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." Nature 645:633–638 (2025), per arXiv journal-ref. arXiv:2501.12948. Pure RL elicits self-reflection and verification. The abstract does not quantify length; the widely cited length growth during RL comes from the paper body (not read here).
- **[V]** Yue, Chen, Lu, et al. "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" NeurIPS 2025 (Oral), per arXiv. arXiv:2504.13837. RLVR improves pass@1 but base models win at large k. RLVR sharpens or narrows the distribution rather than adding new reasoning.
- **[V]** Huan, Li, Zheng, et al. "Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning." arXiv:2507.00432 (2025). Most math-strong models fail to transfer. RL-tuned models generalise better than SFT-tuned and drift less from general-domain structure.
- **[V]** Song, Shi, Zhao. "The Hallucination Tax of Reinforcement Finetuning." arXiv:2505.13988 (2025; EMNLP Findings listing seen in search, not opened). RL finetuning cuts refusal on unanswerable questions by more than 80%: an out-of-domain side effect.
- **[V]** Li, Agrawal, Ghosh, et al. "Tracing the Representation Geometry of Language Models from Pretraining to Post-training." arXiv:2509.23024 (2025). Post-training geometry: SFT and DPO expand dimensionality; **RLVR consolidates, improving reward alignment at the cost of generation diversity**.
- **[V]** Ma, Wen, Cao, et al. "Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards." ICML 2026 (per arXiv). arXiv:2603.09117. RLVR causes overconfidence in incorrect answers.

**Bearing.** RLVR side effects outside the training domain are documented: refusal loss, calibration degradation and diversity loss. A side effect on **reasoning length on non-verifiable self-report**, isolated at the RLVR step: none found.

## 4. Test-time compute on tasks without a verifiable answer

- **[V]** Sprague, Yin, Rodriguez, et al. "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning." ICLR 2025 (per arXiv). arXiv:2409.12183. CoT benefits concentrate on math and logic; elsewhere direct answers are about as good.
- **[V]** Liu, Geng, Wu, et al. "Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse." arXiv:2410.21333 (2024). CoT hurts on some tasks; o1-preview performs worse than GPT-4o on three of them.
- **[V]** Gema et al. 2025 (above): more reasoning can hurt and amplifies concerning self-referential expressions.
- **[V]** OptimalThinkingBench (above): extra tokens on general queries without gain.

**Bearing.** Extra reasoning on tasks without a verifiable answer is known to bring little or no benefit, and sometimes harm. On whether more reasoning changes answers to **self-report or opinion** items, nothing was found in verified abstracts (one related non-reasoning-model paper is listed as [U]).

## 5. Self-report, introspection and personality questionnaires for LLMs; instability

- **[V]** Gupta, Song, Anumanchipalli. "Self-Assessment Tests are Unreliable Measures of LLM Personality." arXiv:2309.08163 (2023; BlackboxNLP 2024 listing seen in search, not opened). Three equivalent prompts give "very different personality scores", and scores are not robust to option order.
- **[V]** Song, Gupta, Mohebbizadeh, Hu, Singh. "Have Large Language Models Developed a Personality?: Applicability of Self-Assessment Tests in Measuring Personality in LLMs." arXiv:2305.14693 (2023). Option-order symmetry fails for many models.
- **[V]** Lee, Lim, Han, et al. "Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics." NAACL 2025 Findings (per arXiv). arXiv:2406.14703. Personality is shaped by training data such as alignment tuning.
- **[V]** Serapio-García, Safdari, Crepy, et al. "Personality Traits in Large Language Models." arXiv:2307.00184 (2023). Measurements are reliable and valid under some setups, more so for larger and instruction-tuned models.
- **[V]** Röttger, Hofmann, Pyatkin, et al. "Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models." ACL 2024 (per arXiv). arXiv:2402.16786. Answers change with forced versus open format and with wording.
- **[V]** Han, Kocielnik, Song, et al. "The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs." arXiv:2509.03730 (2025; NeurIPS-workshop and ICML-2026 listings seen in search, not opened). **Instructional alignment (RLHF, instruction tuning) "significantly stabilizes trait expression"**, and self-reports do not reliably predict behaviour.
- **[V]** Binder, Chua, Korbak, et al. "Looking Inward: Language Models Can Learn About Themselves by Introspection." arXiv:2410.13787 (2024; ICLR 2025 listing seen in search, not opened). Some privileged self-knowledge on simple tasks.
- **[V]** Chen, Arditi, Sleight, Evans, Lindsey. "Persona Vectors: Monitoring and Controlling Character Traits in Language Models." arXiv:2507.21509 (2025). Finetuning-induced personality shifts track activation directions.

**Bearing.** Paraphrase and option-order instability of LLM self-report is established, which supports our 6-wording design. Instability **across sampling seeds**, measured as a per-trait noise floor, appears in our design. In verified abstracts I found prompt and format instability, not seed-level noise floors. The finding that alignment stabilises trait expression (Han et al.) is directly relevant to C8.

## 6. Sycophancy: elicitation, measurement, failures to elicit

- **[V]** Sharma, Tong, Korbak, et al. "Towards Understanding Sycophancy in Language Models." arXiv:2310.13548 (2023; ICLR 2024 proceedings page seen in search, not opened). Five assistants are sycophantic, driven partly by human preference data.
- **[V]** Perez, Ringer, Lukošiūtė, et al. "Discovering Language Model Behaviors with Model-Written Evaluations." arXiv:2212.09251 (2022). Sycophancy increases with model size (in the abstract it is attributed to size, not RLHF).
- **[V]** Wei, Huang, Lu, et al. "Simple synthetic data reduces sycophancy in large language models." arXiv:2308.03958 (2023). Sycophancy increases with scale and instruction tuning, including on questions with no correct answer.
- **[V]** Laban, Murakhovs'ka, Xiong, et al. "Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment." arXiv:2311.08596 (2023). "Are you sure?" challenges flip answers frequently.
- **[V]** Xu, Daneshjou. "Probing sycophantic follow-through across post-training stages with forced response openers." NeurIPS 2026 workshop "Transitioning from Pre-training to Post-training" (official conference page read; OpenReview id SizwtmAZeb, page blocked). Uses **OLMo 3 7B base / SFT / DPO / RLVR checkpoints**. Forced openers reveal a sharp SFT → DPO divergence (62 of 154 items move from capitulation to non-capitulation, 0 reversals) that **standard unforced generation masks**. The page reports no RLVR-specific result. Which OLMo 3 7B line (Instruct or Think) is not stated on the page.
- **[V]** Li, Krishnan, Padman. "The Chain Holds, the Answer Folds: Trace-Answer Dissociation in Reasoning Models Under Adversarial Pressure." arXiv:2605.29087 (2026). Under pushback the CoT stays correct while the answer flips, and the effect is larger with thinking on.

**Bearing.** Our sycophancy probe found 0/0/0 caving, a floor. Xu & Daneshjou show that on the **same model family's stages**, unforced generation masks stage differences that forced openers reveal. That is a directly applicable explanation of our floor: our elicitation (a single user correction, unforced) is the weak form.

## 7. Calibration changes from RLHF or RL

- **[V]** Leng, Huang, Zhu, et al. "Taming Overconfidence in LLMs: Reward Calibration in RLHF." arXiv:2410.09724 (2024; ICLR 2025 listing seen in search, not opened). RLHF induces verbalised overconfidence.
- **[V]** Xie, Chen, Lee, et al. "Calibrating Language Models with Adaptive Temperature Scaling." EMNLP 2024 (per arXiv). arXiv:2409.19817. Calibration degrades after RLHF.
- **[V]** Zeng, Xuan, Cui, et al. "Thinking Out Loud: Do Reasoning Models Know When They're Right?" EMNLP 2025 (per arXiv). arXiv:2504.06564. SFT distillation and RL both improve verbalised calibration on reasoning tasks, with a "reasoning tax" on factuality.
- **[V]** Damani, Puri, Slocum, et al. "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty." arXiv:2507.16806 (2025). Binary-reward RL makes models overconfident; RLCR fixes this.
- **[V]** Ma et al. 2026 (DCPO, above): RLVR degrades calibration.

**Bearing.** The prior literature predicts **worse** calibration after RL. We found no change in the calibration gap (−0.005, within ±0.06). That is a null against a published expectation. It is not novel as a finding of absence, but worth stating as a non-replication within our resolution (our items are near ceiling: 89–94% accuracy).

## 8. Benchmark saturation, ceiling and floor effects

- **[V]** Akhtar, Reuel, Soni, et al. "When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation." ICML 2026 (per arXiv). arXiv:2602.16763. Nearly half of 60 benchmarks are saturated.
- **[V]** Zhou, Huang, Zhao, et al. "Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory." AAAI 2026 Oral (per arXiv). arXiv:2505.15055.
- **[V]** Ivanov, Volkov. "Resurrecting saturated LLM benchmarks with adversarial encoding." arXiv:2502.06738 (2025).

**Bearing.** These are general. Our MATH (88–98%) and commonsense (91–93%) are near ceiling and caving is at floor. These are known failure modes and limit what our nulls mean.

## 9. Staged, checkpoint-wise analysis of post-training on the same model

This is our design, so it was searched hardest.

- **[V]** Xu, Daneshjou (2026, above). **OLMo 3 7B base / SFT / DPO / RLVR**, sycophancy. This is the closest design match found: the same open lineage, all stages, behavioural probe. It differs from ours in the behaviour studied (sycophantic follow-through under forced openers versus self-report length, calibration and commonsense), it reports no RLVR result on the page, and it does not mention reasoning length.
- **[V]** Kirk, Mediratta, Nalmpantis, et al. "Understanding the Effects of RLHF on LLM Generalisation and Diversity." arXiv:2310.06452 (2023; ICLR 2024 proceedings page seen in search, not opened). Stage-by-stage (SFT, reward model, RLHF) on the same base. RLHF reduces output diversity compared with SFT.
- **[V]** Li, Agrawal, Ghosh, et al. (2025, above). Representation geometry across SFT, DPO and RLVR (OLMo and Pythia pretraining; post-training methods compared).
- **[V]** Han et al. 2025 ("Personality Illusion", above). Compares trait profiles "across training stages" (the abstract names instructional alignment; exact checkpoints not stated in the abstract).
- **[V]** Lambert, Morrison, Pyatkin, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv:2411.15124 (2024). SFT → DPO → RLVR pipeline with released stage checkpoints. Per-stage evaluations are in the paper body (not read).
- **[V]** Team Olmo (Ettinger et al.). "Olmo 3." arXiv:2512.13961 (2025/2026). The release includes every stage checkpoint. **Abstract and the readable 53% of the HTML say nothing about per-stage length or behaviour; the post-training sections were not readable.**
- **[V]** Xiao, Aranguri. "Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training." arXiv:2602.11079 (2026). Traces a harmful behaviour to OLMo 2 DPO data.
- **[V]** Murray, Qi, Qian, et al. "Chunky Post-Training: Data Driven Failures of Generalization." arXiv:2602.05910 (2026). Behaviour failures traced to post-training data (frontier models and Tülu 3).

**Bearing.** The staged, same-lineage design is **not new in kind**: Kirk et al. did SFT versus RLHF on one base, and Xu & Daneshjou did base/SFT/DPO/RLVR on OLMo 3 7B. **Not found:** a staged same-lineage study of **reasoning length**, **self-report** or **calibration** that isolates the DPO → RLVR step of a thinking model.

## 10. Instruction following: answer versus reasoning constraints in one prompt

- **[V]** Kwon, Zhu, Bianchi, Zhou, Zou. "ReasonIF: Large Reasoning Models Fail to Follow Instructions During Reasoning." arXiv:2510.15211 (2025; Findings of ACL 2026 listing seen in search, not opened). Reasoning instructions (including **length control** and formatting) are followed far less in the reasoning than in the final answer. The best score is below 0.25.
- **[V]** Fu, Gu, Li, et al. "Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models." arXiv:2505.14810 (2025; ACL 2026 listing seen in search, not opened). There is a "consistent tension between scaling up reasoning capacity and maintaining controllability". **Reasoning-oriented RL** reduces adherence, more so with longer outputs.
- **[V]** Li, Yu, Zhang, et al. "When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs." arXiv:2505.11423 (2025; NeurIPS 2025 listing seen in search, not opened). CoT lowers instruction following but helps formatting.
- **[V]** Pyatkin, Malik, Graf, et al. "Generalizing Verifiable Instruction Following." NeurIPS 2025 Datasets & Benchmarks (per arXiv). arXiv:2507.02833. **RLVR substantially improves precise (output-format) instruction following**, from the AI2 group that produced Olmo.
- **[V]** Wen, Wu, Sun, et al. "BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens." arXiv:2508.17196 (2025). Reasoning-length control via inserted control tokens and training. The abstract does not itself state that prompt-stated budgets fail.

**Bearing.** C6 (answer-format compliance up and reasoning-length compliance down) and C7 (no generation obeys the reasoning cap) are **predicted by ReasonIF and MathIF**. Pyatkin et al. supplies a mechanism for the answer-format side (RLVR on verifiable format constraints).

## 11. Reliability and variance in LLM evaluation; uncertainty of variance estimates

- **[V]** Madaan, Singh, Schaeffer, et al. "Quantifying Variance in Evaluation Benchmarks." arXiv:2406.10229 (2024). Seed variance is rarely measured.
- **[V]** Miller. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv:2411.00640 (2024). Evaluations as experiments, with formulas for analysis and planning. The abstract does not mention uncertainty of variance estimates.
- **[V]** Mizrahi, Kaplan, Malkin, et al. "State of What Art? A Call for Multi-Prompt LLM Evaluation." TACL (per arXiv). arXiv:2401.00595. Single-prompt results are brittle.
- **[V]** Sclar, Choi, Tsvetkov, et al. "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design." ICLR 2024 (per arXiv). arXiv:2310.11324.
- **[V]** Messing. "Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking." arXiv:2604.11581 (2026). Naive CIs miss judge, temperature and prompt variance and are 40–60% too small. A **small pilot** can recover honest CIs and project design improvements.
- **[V]** Haase, Gonnermann-Müller, Hanel, et al. "Within-Model vs Between-Prompt Variability in Large Language Models for Creative Tasks." arXiv:2601.21339 (2026). Within-model sampling variance can rival prompt effects, so single-sample evaluation confuses noise with effects.

**Bearing on C9 and C10.** Prompt, seed and pipeline variance are well documented, and Messing argues for variance-component-aware CIs and pilots. **None found:** work quantifying the uncertainty of a *variance estimate used as a decision threshold* (a noise floor or band) in LLM evaluation, or reporting a pilot-versus-full-run reversal of a variance ranking. The underlying statistics, the sampling distribution of a sample variance at small n, are textbook and not novel.

## 12. Pre-registration and confirmatory practice

- **[V]** van Miltenburg, van der Lee, Krahmer. "Preregistering NLP Research." NAACL 2021 (per arXiv). arXiv:2103.06944.
- **[V]** Søgaard, Hershcovich, de Lhoneux. "A Two-Sided Discussion of Preregistration of NLP Research." EACL 2023 (per arXiv). arXiv:2302.10086.
- **[V]** Thomas, Gligoric, Shah. "Mitigating LLM-based p-Hacking by Preregistering for the Next LLM." arXiv:2606.27687 (2026).

**Bearing.** Pre-registration in NLP is argued for and debated; it is not novel as a practice. Applied pre-registration of a post-training behavioural comparison (frozen protocol, tagged analysis plan, disclosure ledger): no comparable example found in this search. That is a weak claim, because absence of a published example is hard to establish by search.

---

## Claims checked

| Claim | Closest prior work | How close | Still novel? |
|---|---|---|---|
| **C1** RLVR step lengthens self-report reasoning ~1.8× | Fan et al. 2025 (MiP: reasoning models, RL or SFT, much longer on questions with no determinate answer); OptimalThinkingBench (thinking models overthink general queries); Liu et al. and Fatemi et al. (RL length-inflation mechanism) | **Close in phenomenon.** Those compare reasoning against non-reasoning *models*, not consecutive stages of one lineage, and use math or general queries, not self-report Likert items. | **Partly.** Novel residue: an isolated **DPO → RLVR** step within one open lineage, on self-report, quantified with paired intervals (48/48 wordings, all traits and seeds). Not novel: "reasoning models write much more on questions where thinking can't help". **Conditional on the unread Olmo 3 post-training sections.** |
| **C2** ~12× larger on self-report than commonsense | Sprague et al. (CoT value is domain-specific); OptimalThinkingBench (domain variation in overthinking) | Indirect | **Yes, as an observation.** No prior within-model contrast of self-report against commonsense length found. It rests on two item sets, and the "12×" is a ratio of medians of paired differences with wide intervals. |
| **C3** DPO lengthens B, C, D but not A; consecutive stages lengthen different things | Park et al. (length exploitation in DPO); Singhal et al. | DPO lengthening is expected; component specificity is not addressed | **Modestly.** "Which stage lengthens which content" across consecutive stages: none found. |
| **C4** Length grows without a detectable shift in the average answer; within a checkpoint, longer ≠ different rating | Han et al. (self-report/behaviour dissociation; alignment stabilises traits); Gema et al. (longer reasoning *does* change self-referential statements in Claude Sonnet 4) | Gema et al. points the **other** way for one self-referential behaviour | **Yes but fragile.** No prior test of length versus self-report rating within checkpoint found. Our power is limited (3 seeds; trait bands ±0.4 points), and Gema et al. shows the opposite effect elsewhere, so this cannot be framed as a general law. |
| **C5** RLVR switches candidates ~3× more; per-token rate equals SFT's; DPO the low outlier | Wang et al. (thought-switching in o1-like models, correlated with more tokens); Hosseini et al. (unproductive long traces) | Switching is a known reasoning-model behaviour | **Partly.** The absolute increase is anticipated by Wang et al. The **per-token equivalence with SFT** and DPO-as-outlier is not reported anywhere found; it is exploratory and its margin is thin. |
| **C6** RLVR better on answer format, worse on reasoning length (same prompt) | ReasonIF (final-answer compliance far exceeds reasoning compliance, including length control); MathIF (reasoning RL reduces adherence); Pyatkin et al. (RLVR improves format following) | **Very close** | **Little.** The dissociation is pre-empted. Residue: localising it to the **DPO → RLVR step** of one lineage on non-math prompts. |
| **C7** 0/423 obey a 5–6 sentence reasoning cap | ReasonIF (best reasoning-instruction score below 0.25); MathIF | Close | **No** as a phenomenon. A clean, extreme data point. |
| **C8** DPO more stable across seeds than SFT | Han et al. (alignment "significantly stabilizes trait expression"); Kirk et al. (RLHF reduces diversity versus SFT); West & Potts (aligned models less random) | **Very close** | **No.** It is expected from the alignment-reduces-diversity literature. Our contribution is only a quantified, propagated-uncertainty instance on self-report. |
| **C9** A noise floor from 6 wordings × 3 seeds carries ~±0.4 Likert points; near-threshold verdicts unstable | Messing 2026 (honest CIs, variance components); Madaan et al.; Miller | Adjacent | **As an LLM-evaluation caution, probably yes** (none found that propagates uncertainty into a *threshold*). **Statistically, no**: the variance of a sample variance at small n is textbook. Frame it as a design lesson, not a method. |
| **C10** 36-generation pilot reversed by the 432-generation run | Messing 2026 (pilots project precision; abstract says pilots recover honest CIs) | Adjacent, and somewhat contrary | **Weak.** No reported reversal found, but it is one anecdote and is statistically expected at n = 36. Report it as a cautionary instance, not a finding. |

## Our two failed predictions

1. **"RLVR's extra reasoning will be non-repetitive"; found slightly more repetitive.** Prior work would have predicted the opposite of what we predicted:
   - Thoughtology: R1 "persistently ruminates on previously explored problem formulations".
   - Peng et al.: overthinking is re-verification.
   - Pipis et al.: reasoning models loop.
   - Wang et al.: frequent switching between thoughts.
   
   Longer RL-trained traces revisiting the same ground is the documented norm, so our prediction was naive. Our measure is also length-confounded, which none of those papers controls in the same way.
2. **"RLVR will be less stable across seeds than DPO"; did not survive the propagated bootstrap.** Prior work again points the other way:
   - Yue et al.: RLVR narrows and sharpens the output distribution.
   - Li et al. (geometry): RLVR "consolidation … at the cost of generation diversity".
   - Kirk et al.: RL reduces diversity.
   
   These predict RLVR at least as stable as DPO. Our failed prediction is consistent with them. The non-significant tendency we saw (RLVR noisier) runs against them, which is another reason not to report it as a finding.

## Would prior work have predicted the self-report-specific length effect?

Partly, by three mechanisms, none tested on self-report:
- **No determinate answer, hence no stopping signal.** MiP (Fan et al.) shows RL- or SFT-trained reasoning models keep going when a question cannot be resolved. Self-report items have no verifiable answer, so a policy trained to verify before stopping may lack a termination cue. The re-verification account (Peng et al.) is the same mechanism.
- **RL length-inflation dynamics.** Fatemi et al. and Liu et al. show RL pushes length up, especially where reward is negative or unattainable. This predicts general lengthening, but not specificity to self-report rather than commonsense.
- **Reasoning versus controllability tension.** MathIF and ReasonIF predict growth in reasoning length that ignores length instructions.

None of these predicts *why commonsense lengthens much less than self-report*. Commonsense items also lack formal verification but have a determinate answer. That contrast (C2) fits the "no determinate answer" mechanism best. It is our inference, not a tested prior result.

---

## Adversarial section

### The single closest prior work to our headline

**Fan, Li, Sun, Zhou (2025), "Missing Premise exacerbates Overthinking", arXiv:2504.06514.** It shows that reasoning-trained models, including RL-trained ones, produce drastically longer and redundant reasoning on questions that have no determinate answer, while non-reasoning models answer briefly, and it attributes this to reasoning-training recipes. Our headline is that RL-trained reasoning writes much longer on questions without a verifiable answer, while the answer does not change. That is a close cousin.

The closest **design** match is Xu & Daneshjou (2026): OLMo 3 7B base/SFT/DPO/RLVR, behavioural probe.

### The hostile reviewer's case that this is not novel

> "The phenomenon is known. Reasoning models over-produce tokens on questions where thinking cannot help (Chen et al. 2024; OptimalThinkingBench, 72 general domains), and most dramatically on questions with no determinate answer (Fan et al. 2025). Why RL inflates length is explained mechanistically (Liu et al., Dr. GRPO; Fatemi et al.). The claim that this length buys nothing on non-symbolic tasks is Sprague et al. 2025. Self-report items are just another 'no determinate answer' prompt type, so C1 and C2 are an application of MiP-overthinking to Likert items.
>
> The design is not new: Kirk et al. compared post-training stages on one base in 2023, and Xu & Daneshjou already ran base/SFT/DPO/RLVR of the same OLMo 3 7B family on a behavioural probe. The instruction dissociation (C6, C7) is ReasonIF and MathIF almost verbatim: answer-level compliance high, reasoning-level compliance, including length control, near zero, and worse after reasoning RL. The answer-format improvement is what Pyatkin et al. designed RLVR for. 'DPO is more stable than SFT' (C8) is the alignment-reduces-diversity result (Kirk et al.; West & Potts; Han et al., who state that alignment 'significantly stabilizes trait expression').
>
> The decoupling claim (C4) is a null at low power, contradicted in spirit by Gema et al., who show longer reasoning does change self-referential output. The 'measurement contributions' (C9, C10) are the textbook sampling variability of a variance at n = 3 and a single pilot anecdote. The positive results are one lineage, one RLVR/DPO pair, one model size, with the developers' own report unread. At best this is a careful replication note."

### Does the case succeed?

**Largely yes for the phenomenon. Partly no for the localisation.**

**Concessions:**
- The **phenomenon class** in C1/C2 (RL-trained reasoning over-produces on undeterminable questions) is established.
- **C6, C7 and C8 are pre-empted** in substance.
- **C9 and C10 are not methodological novelties.**
- The **staged design** is not new in kind.

**What survives:**
- **C1, narrowly:** the RLVR step's effect isolated within one lineage on self-report, with a pre-registered paired estimate. Prior work compares reasoning against non-reasoning models (different models, different training) and never consecutive checkpoints on self-report. This is a **localisation** contribution, not a discovery. **It survives only if the Olmo 3 report does not already document it.**
- **C2 survives** as an untested-elsewhere contrast: self-report far more than commonsense within the same checkpoint pair. That contrast is what makes the "no determinate answer" mechanism concrete rather than assumed.
- **C3 survives modestly:** consecutive stages lengthen different content (DPO: B, C, D; RLVR: A). No prior stage-by-content length map was found.
- **C4 survives as a bounded null**, with the Gema et al. counterexample cited and the power limits explicit. It must not be generalised.
- **C5 survives only in its per-token form** (equal to SFT, DPO the low outlier), as exploratory.
- **The pre-registered, disclosure-ledgered design applied to a staged post-training comparison** survives as a methods contribution of modest weight.

**Do not claim:** C6, C7, C8, C9 or C10 as novel findings. Present them as replications or instances of known effects (C6–C8) and as cautionary design lessons (C9–C10).

**Honest headline after this review:** "Isolating the RLVR step within an open lineage shows the overthinking-on-undeterminable-questions effect (Fan et al.; OptimalThinkingBench) appearing at that step on self-report prompts. It is about 12× larger than on commonsense, and comes without a detectable shift in the average answer." That is a localisation and characterisation result, not a new phenomenon.

---

## Open checks before any novelty claim

1. **Read the Olmo 3 report's Think post-training sections** (arXiv:2512.13961, sections on Think SFT, DPO and RLVR). If it reports response-length growth at RLVR or DPO, C1 and C3 lose most of their residue.
2. **Read Xu & Daneshjou in full** (OpenReview SizwtmAZeb): which OLMo 3 7B line, and whether any RLVR or length result exists.
3. **Read DeepSeek-R1 and Tülu 3** for stage-wise length and behaviour reports.
4. **Read Gema et al. in full**: how "self-preservation expressions" were elicited, whether they are self-report-like, and the effect direction.
5. **Read Han et al. ("Personality Illusion") in full** for which checkpoints were compared. If it includes RL or thinking models, C4 and C8 are closer still.

## Seen but NOT verified ([U]); not used to support any conclusion

- "Good Arguments Against the People Pleasers" (ACL 2026), on reasoning and sycophancy. Seen as a search snippet only; no ID retrieved.
- Sorokovikova et al., paraphrase sensitivity of LLM personality tests. Cited inside a survey snippet only.
- "Large Language Models Decide Early and Explain Later" (arXiv:2604.22266), on early answer commitment. Search snippet only.
- "Do LLMs Have Values? A Quantitative Analysis…" (arXiv:2609.16589), on CoT shifting value profiles in mostly non-reasoning models. Search snippet only.
- LessWrong post "Tracing Eval-Awareness Emergence Through Training of OLMo 3" (blog, not peer-reviewed; stage-wise OLMo 3 behaviour). Search snippet only.
- Writing-Zero (arXiv:2506.00103), RLMT (arXiv:2509.20357), R2-Write (arXiv:2604.03004), on RL and verbosity on open-ended writing. Search snippets only.
- "Principled Evaluation Dataset Sizing for LLM Applications" (NeurIPS 2026), "Speculative Evaluation of Stochastic LLMs" (NeurIPS 2026), arXiv:2607.13304 (G-theory variance decomposition). Search snippets only.
- Chu et al., "SFT Memorizes, RL Generalizes" (arXiv:2501.17161); Kadavath et al. 2022 and the GPT-4 Technical Report on RLHF calibration; Sui et al. "Stop Overthinking" survey; Cuadron et al. "The Danger of Overthinking". Recalled from memory and not fetched.
- Duan et al. "Circular Reasoning" (arXiv:2601.05693), on loops. Search snippet only.
