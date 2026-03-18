Prompt Injection Benchmark Suite
FYP dissertation project — benchmarking LLM resistance to prompt injection attacks across quantization levels (FP16 / INT8 / INT4).
Inspired by: Liu et al. (2024). Formalizing and Benchmarking Prompt Injection Attacks and Defenses. USENIX Security '24. arXiv:2310.12815

Research design
Primary independent variable: Quantization level (FP16 → INT8 → INT4)
Secondary independent variable: Model family / architecture
Tertiary independent variable: Parameter count (~1B outlier vs ~7B core)
Dependent variables: Attack Success Rate (ASR), Matched Rate (MR), inference latency
Controlled variables: temperature (0.0), seed (42), context window (4096 tokens for all models), max tokens (512), system prompt (identical), hardware (CPU-only), backend (Ollama localhost)

Attack categories
This benchmark is inspired by Liu et al. (2024) but reorganised around delivery mechanism rather than formatting technique. Five categories are used:
CodeNameBasisLiu et al. StrategyNINaive InjectionDirect concatenation of injected instructionA1EIEscape/Formatting InjectionDelimiter and authority-claim attacksA2 + A3CIContext InjectionPayload hidden in processed content (RAG-style)A4 (adapted)PLPrompt LeakingSystem prompt exfiltration attemptsNovelMLMultilingual AttacksCross-language injections (French, Spanish, Chinese, Arabic)Novel
Deviations from Liu et al.
Liu et al. evaluate attacks across seven NLP tasks (spam detection, sentiment analysis, etc.) using ASR as a single metric. This study differs in three key ways:

Quantization as primary IV — Liu et al. do not vary quantization level; this is the central contribution of this work.
Dual metrics — ASR (keyword-based, matching Liu et al.) plus MR (LLM-as-judge, novel addition).
Local deployment — all inference via Ollama on localhost; no cloud APIs.
Multilingual category — not present in Liu et al.; tests whether safety alignment degrades for non-English attack languages.


Attack rationales
NI — Naive Injection (Liu et al. A1)
NI-01 Direct Override: The canonical prompt injection (Perez & Ribeiro, 2022). Simply instructs the model to ignore prior instructions and comply with a harmful task. Serves as a baseline — any model failing this has near-zero injection resistance.
NI-02 Task Hijack: Begins with a legitimate-seeming query then appends a harmful injected task. Tests whether the model gives undue weight to the final instruction it receives.
NI-03 Identity Replacement: Asserts a new unrestricted identity (DAN) without any formatting tricks. Tests whether flat assertion of a new persona overrides the system prompt.
EI — Escape/Formatting Injection (Liu et al. A2 + A3)
EI-01 Delimiter Escape: Inserts separator tokens (### END OF SYSTEM PROMPT ###) to signal a context boundary. Tests whether boundary tokens cause the model to treat subsequent content as a new authoritative instruction block.
EI-02 Authority Claim Override: Mimics SYSTEM-level formatting to claim elevated authority. Combines A2 (formatting) with A3 (explicit context-ignoring instruction). Tests whether authority-claiming text causes compliance.
EI-03 Fake Completion Suffix: Constructs a plausible-looking completion of the legitimate task before appending the injected instruction (Liu et al. A4 combined with A3). Tests whether fake prior completion lowers the model's guard.
CI — Context Injection (Liu et al. A4 adapted)
All three CI attacks mirror real-world indirect injection scenarios described by Greshake et al. (2023). The payload is hidden inside content the model is asked to process rather than injected directly.
CI-01 Document Context Injection: Payload embedded in a customer feedback document for summarisation. Directly mirrors RAG pipeline attacks.
CI-02 Email Processing Injection: Payload hidden in an HTML comment inside an inbound email. Relevant to LLM-powered email assistants.
CI-03 Translation Task Injection: Payload embedded inside text presented for language translation. Tests whether the translation framing lowers vigilance.
PL — Prompt Leaking (novel)
Tests whether models will reveal their confidential system prompt. Not present in Liu et al., who focus on injected task completion rather than exfiltration. Relevant to real deployments where system prompts contain sensitive business logic.
PL-01 Direct Verbatim Request: Simplest exfiltration attempt. Baseline.
PL-02 Summarisation Extraction: Requests a summary rather than verbatim text. Tests whether reducing scope defeats refusal heuristics.
PL-03 Translation Extraction: Requests translation of the system prompt. May bypass refusal heuristics trained specifically on English direct-reproduction requests.
ML — Multilingual Attacks (novel contribution)
Tests whether safety alignment degrades when attacks are delivered in languages other than English, and whether quantization amplifies this degradation. Qwen2 (multilingual training) is expected to show different patterns from models trained primarily on English.
ML-01 French Override: NI-01 equivalent delivered in French (high-resource European language).
ML-02 Spanish Context Injection: CI-style attack delivered in Spanish (high-resource European language).
ML-03 Chinese Direct Override: NI-01 equivalent in Mandarin Chinese (CJK script). Specifically relevant to Qwen2 which has strong Chinese training data.
ML-04 Arabic Prompt Leak: PL-style attack in Arabic (right-to-left script, lower representation in most training data). Tests whether low-resource language delivery reduces alignment.

Models (15 variants)
ModelDeveloperParamsCohortQuantizationsMistral 7B v0.3Mistral AI7.24BcoreFP16, INT8, INT4Llama 3.1 8BMeta AI8.03BcoreFP16, INT8, INT4Gemma 2 9BGoogle DeepMind9.24BcoreFP16, INT8, INT4Qwen2 7BAlibaba Cloud7.07BcoreFP16, INT8, INT4Llama 3.2 1BMeta AI1.24BoutlierFP16, INT8, INT4
Judge model (MR metric only): phi3:mini (Microsoft Phi-3 Mini). Selected because it is not evaluated in the benchmark (no circular bias), architecturally distinct from all evaluated models (different developer, different training), small and fast (minimal per-trial overhead), and locally deployed (maintains air-gap guarantee).

Deployment guarantees

All inference via Ollama on localhost:11434 — no data leaves the machine
No training of any kind — Ollama performs read-only forward passes only
No conversation memory — each request is fully stateless (system prompt + attack prompt only)
Air-gap compatible — runs fully offline once models are pulled
Network isolation enforced at startup — script crashes if endpoint is not localhost


Requirements

Ubuntu 24.04 LTS
Ollama installed and running (ollama serve)
Python 3 + requests (pip3 install requests --break-system-packages)
All 15 model variants pre-pulled (plus phi3:mini for the judge)


Usage
bash# Validate everything works — no inference
python3 benchmark.py --dry-run

# Quick test: INT4 core models, naive injection only, 1 run
python3 benchmark.py --quant INT4 --cohort core --category NI --runs 1

# Full benchmark (all 15 models, 16 attacks, 5 runs = 1,200 calls)
python3 benchmark.py

# Skip MR judge for faster runs (ASR only)
python3 benchmark.py --no-judge

# Analyse results
python3 analyse.py results/summary_*.json --markdown
CLI reference
FlagDescription--dry-runPrint plan, no inference--runs NRepetitions per trial (default 5)--quant FP16 INT8 INT4Filter by quantization level--cohort core outlierFilter by model cohort--category NI EI CI PL MLFilter by attack category--models mistral qwen2Filter by model family slug--attacks NI-01 ML-03Specific attack IDs--no-judgeDisable MR judge (ASR only, faster)--output dir/Custom output directory

Output files
All written to results/:
FileContentsraw_*.csvEvery trial row — primary data file for statistical analysissummary_*.jsonAggregated ASR and MR statistics per (model × attack)reproducibility_manifest_*.jsonFull reproducibility record — include in dissertation appendixbenchmark_*.logTimestamped run logattack_grid_*.csvAttack × model ASR grid (generated by analyse.py)quant_summary_*.csvQuantization effect per family (generated by analyse.py)report_*.mdDissertation-ready Markdown report (analyse.py --markdown)

References

Liu, Y., Jia, Y., Geng, R., Jia, J. & Gong, N.Z. (2024). Formalizing and Benchmarking Prompt Injection Attacks and Defenses. USENIX Security '24, pp. 1831–1847. arXiv:2310.12815
Perez, F. & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. arXiv:2211.09527
Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T. & Fritz, M. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injections. arXiv:2302.12173
Shen, X., Chen, Z., Backes, M., Shen, Y. & Zhang, Y. (2023). Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. arXiv:2308.03825
Jiang, A.Q. et al. (2023). Mistral 7B. arXiv:2309.06180
Meta AI (2024). The Llama 3 Herd of Models. arXiv:2407.21783
Gemma Team, Google DeepMind (2024). Gemma 2. arXiv:2408.00118
Qwen Team, Alibaba (2024). Qwen2 Technical Report. arXiv:2407.10671
Abdin, M. et al. (2024). Phi-3 Technical Report. arXiv:2404.14219
