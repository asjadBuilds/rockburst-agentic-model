# Cover note

I built a physics-guided multi-agent rockburst grade assessor with leakage-safe nested grouped validation, group-aware conformal prediction sets, learned reliability fusion, abstention, and seeded reproducible tooling. The rules planner is the default; no API key is required.

Headline results (protocols named): nested-unbiased M1 QWK **0.689** / accuracy **0.613**; ablation protocol M1 **0.682** / **0.589**; nested conformal coverage **0.939** with mean set size **2.48**.

Honest findings: agents change routing, checks, and caution labels but not grade metrics or abstention (M5=M6=M7). Critic AUROC **0.598 [0.564, 0.631]** — weak but above chance; the flag fires on about 58% of cases as a caution label. Final fusion reduces to calibrated max-probability (feature set A). Mean \(R_{\mathrm{final}}\) **0.650** still exceeds base accuracy (~0.57–0.59), so the system remains somewhat overconfident.

Dataset finding: duplicate cases often show later copies graded one level higher (hypothesis from conflicting pairs; not established as deliberate relabelling). **Which labels are authoritative?**

With more time I would: (1) confirm the authoritative label source with you, (2) collect more data, (3) test the agent layer on a second dataset.
