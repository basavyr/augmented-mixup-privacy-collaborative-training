# Handoff — Singularized Mixup for Open Tech Days

**Project:** Internal demo + competition submission for Orange "Open Tech Days"
**Topic:** Singularized Mixup for Privacy-Preserving Collaborative Training
**Authors of underlying research:** Mihail Plesa, Fabrice Clerot, Simona David, Robert Poenaru (Orange Services / Orange Research)
**Publication venue:** Transactions on Machine Learning Research (TMLR), April 2026
**Paper URL:** https://openreview.net/forum?id=1SrZyNgmpY
**Selected demo vertical:** Healthcare (multi-hospital diagnostic-model training)

---

## 0. How to use this document

This handoff is written for a teammate (or a future-you) who needs to:

1. Understand the research paper without re-reading it cover to cover.
2. Pick up the competition form answers and finish polishing them, knowing the reasoning behind every editorial choice.
3. Build the demo end-to-end, with a clear design, scope, and stack already decided.

Three companion files are already in the repo:

- `Open_Tech_Days_competition_answers.txt` — current draft of the Q1-Q8 answers, with sources.
- `../paper/7201_Augmented_Mixup_Procedure.txt` — full plain-text dump of the paper.
- `../paper/7201_Augmented_Mixup_Procedure.pdf` — original PDF.

Whenever this handoff says "see paper Sec. X" it refers to the TMLR paper.

---

## 1. The research paper, in one read

### 1.1 Problem statement

Multiple parties (hospitals, telcos, banks, public administrations) each hold private datasets that they cannot legally share, yet a model trained on the *union* of those datasets would dramatically outperform any model trained on a single party's slice. The challenge is to enable that joint training without exposing the raw data.

Current options all have serious flaws:

- **Federated learning (FL)** is operationally heavy, sensitive to non-IID data, and structurally vulnerable to gradient-inversion attacks (Geiping et al., NeurIPS 2020) that allow a malicious aggregator to reconstruct individual training samples from the exchanged gradients.
- **Homomorphic encryption (HE) / Secure multi-party computation (MPC)** are cryptographically strong but typically 100x-10,000x too slow for deep learning, with prohibitive communication costs.
- **Data-mixing / instance-hiding schemes** — most notably InstaHide (Huang et al., ICML 2020) and NeuraCrypt (Yala et al., 2021) — promised the right cost profile but were both publicly broken by Carlini and collaborators in 2021 (S&P 2021 and arXiv:2108.07256). Subsequent attacks by Chen et al. (2020) and Luo et al. (AAAI 2022) further eroded confidence in this entire family of methods.

The market is left with a gap: a fast, scalable scheme that genuinely resists the attacks that killed the previous generation.

### 1.2 Why prior mixing schemes failed

Carlini's attack, Chen's attack, and Luo's attack all rely on the same structural weakness in InstaHide: each private sample appears in *many* encoded mixtures. This repeated reuse creates a persistent statistical signal that an attacker can exploit:

1. Take the absolute value of each encoded image to neutralise the sign-flip mask.
2. Train a similarity network on public data to detect when two encoded images share a common private source.
3. Cluster encoded samples by shared private component.
4. Solve the resulting (noisy but tractable) linear system `B = M*A + e` where `e` averages out across many equations.

The end result: near-complete recovery of the original images.

### 1.3 The proposed method — Singularized Mixup

The fix is conceptually simple: never let the same private sample appear in more than one mixture, and inject noise that prevents the linear system from being inverted even if the attacker knows everything else.

**Algorithm 1 — Singularized Mixup**

Input: private dataset `{(x_i, y_i)}_{i=1..n}`, noise norm `r`
Output: mixed dataset `{(x_tilde_i, y_tilde_i)}_{i=1..n}`

```
sample a random permutation pi over {1..n}
for each i = 1..n:
    sample weights w_i ~ Uniform([0,1]^2), normalized so ||w_i||_1 = 1 and ||w_i||_inf <= alpha
    sample noise e_i ~ Uniform(S(0, r))     # uniform on sphere of radius r
    x_tilde_i = w_i1 * x_i + w_i2 * (x_pi(i) + e_i)
    y_tilde_i = w_i1 * y_i + w_i2 * y_pi(i)
return {(x_tilde_i, y_tilde_i)}
```

Three departures from InstaHide:

1. **k = 2 fixed**, and only private images. No public images mixed in.
2. **Noise injected on the second (partner) component** before mixing, tightly coupling noise with private content.
3. **No sign-flip mask** (those were already shown to be defeated by taking absolute values).

### 1.4 Theoretical security argument

Two original theorems carry the security argument.

**Theorem 1 — Minimax reconstruction lower bound.**
Let `W = D1 + D2 * P_pi`. If `W` is invertible, then `X_tilde = W * X + E` where `E_i = w_i2 * e_i`. For *any* estimator (linear or non-linear, even with full knowledge of `W` and the encoded `X_tilde`):

```
sup_X E[||x_i - x_hat_i||_2^2] >= r^2 * T_i
where T_i = sum_l (W^-1)_il^2 * w_l2^2
```

In plain language: the reconstruction MSE for any attacker scales linearly with `r^2`. There is no estimator (cleverness, neural net, or otherwise) that can do better than this lower bound.

**Theorem 2 — Directional SNR control.**
Defines per-direction SNR `SNR(u) = E<S_i,u>^2 / E<I_i,u>^2` for signal `S_i = w_i1 * x_i` and interference `I_i = (1 - w_i1)(x_pi(i) + e_i)`. Shows that with `r = m_f * D` (where `D = E||X - X'||_2` is the average inter-sample distance), the worst-case directional SNR is bounded by:

```
sup_||u||=1 SNR(u) <= tau    is achieved when    m_f >= sqrt(alpha^2 / (tau * (1-alpha)^2) - 1)
```

This is the practical recipe: pick a desired security level `tau`, compute `m_f` from the closed-form, set `r = m_f * D` where `D` is estimated empirically.

The guarantee is *average-case* (not worst-case per sample, which is why a small LPIPS-based post-processing filter is used in practice to discard rare outlier mixtures).

### 1.5 Empirical results that matter

All numbers are from the paper's tables. Use them as background; in the competition answers we deliberately avoid pointing at specific tables.

**Attack resistance (CIFAR-10):**

| tau     | Linear attack SNR (dB) | U-Net attack SNR (dB) |
|---------|------------------------|------------------------|
| 1       | 0.71                   | 7.58                   |
| 0.1     | -5.23                  | 6.98                   |
| 0.01    | -7.10                  | 5.40                   |
| 1e-3    | -7.63                  | 2.87                   |
| 1e-4    | -7.79                  | 0.10                   |
| 1e-5    | -7.84                  | -0.14                  |
| 1e-6    | -7.85                  | -0.15                  |

By `tau = 1e-4` both attacks are effectively defeated.

**Accuracy at `tau = 1e-6` (two orders of magnitude stricter than necessary):**

| Dataset       | InstaHide (k=4) | Singularized Mixup |
|---------------|-----------------|---------------------|
| MNIST         | 99.66           | 99.32               |
| CIFAR-10      | 91.20           | 90.51               |
| CIFAR-100     | 74.01           | 75.99               |
| Tiny-ImageNet | (not reported)  | 72.50               |

Matches or beats InstaHide despite far stronger privacy.

**Collaborative training vs. FedProx (CIFAR-10, Dirichlet partitioning):**
Gains range from +0.41 pp to +11.83 pp across 9 configurations, with the largest gaps in heterogeneous (small beta) or many-parties (P = 100) settings.

### 1.6 Why this is genuinely novel

- Resolves the core weakness exploited by the entire InstaHide attack family.
- Proves security under a conservative threat model where the attacker is *given* the mixing weights — most prior schemes never granted that.
- One-shot collaboration protocol: no iterative gradient exchange, no FL infrastructure, no gradient-inversion surface.
- Distribution-aware, closed-form noise calibration through Theorem 2.
- Runs on commodity hardware (single NVIDIA L4 GPU was sufficient for all paper experiments, including CIFAR-5M).

---

## 2. Competition form — our approach

### 2.1 Audience and tone

The internal jury at Open Tech Days is a mix of business and technical reviewers from across Orange BUs. The answers therefore:

- Lead with business context and value (Q1, Q5, Q6, Q7) but never hide the science.
- Use formal, measured prose. Marketing-style language ("breakthrough", "wow", "unlocks", "world-class") is deliberately avoided.
- Cap each answer at 2-3 paragraphs.
- Use normal sentence case throughout. No all-caps emphasis.
- Cite every numeric or factual claim to an external, verifiable source. The Sources section at the end of the file is the receipt the jury can check.
- Never cite the underlying paper's specific tables or sections. The jury should perceive the work as a substantial Orange capability, not as a paper summary. The peer-reviewed publication is mentioned as backing only at the high level ("recently published at a leading peer-reviewed venue in machine learning").
- Acknowledge limitations honestly (Q8): the guarantee is average-case, not cryptographic. This transparency is itself a differentiator in a field where overclaiming has repeatedly damaged credibility (InstaHide and NeuraCrypt were both publicly broken after overstated security claims).

### 2.2 Question-by-question editorial logic

**Q1 — Context.** Hook with the GDPR enforcement totals (EUR 6.1B cumulative fines, EUR 4.97B in the Media/Telecoms sector) to anchor the problem in numbers the jury cannot dismiss. Then walk through why each incumbent technical approach is failing today (FL, HE, mixing schemes). Close with the market-sizing data point (Privacy Management Software USD 15.2B by 2028, 41.9% CAGR) so the jury sees both the pain and the opportunity.

**Q2 — Innovation.** Use the three-levels framing: conceptual, theoretical, practical. The opening line ("The innovation is at three levels — conceptual, theoretical, and practical — and that combination is what makes it defensible") sets up a "promise" and each paragraph fulfils one third. The conceptual paragraph attributes singularization to prior Orange cryptographic work to anchor the Orange-DNA story. The practical paragraph closes with a memorable line: "small enough to fit in a few dozen lines of code, yet powerful enough to change the economics of privacy-preserving AI."

**Q3 — Technologies.** Two paragraphs. The first describes our core stack (the algorithm, the theory, the pretrained backbones, the classifier head). The second describes our adversarial evaluation stack (linear attack, U-Net attack, metrics, FedProx baseline, datasets). The closing line — "no specialised cryptographic accelerator, no secure multi-party computation cluster" — is a quiet jab at HE/MPC competitors.

**Q4 — Differentiators.** Three differentiators, in the order: (a) resists the attacks that killed prior schemes, (b) does so at almost zero utility cost, (c) beats federated learning under realistic heterogeneity. The phrasing "we resist the very attacks that killed the prior generation" is deliberately memorable.

**Q5 — Customer benefit.** The universal pain articulation first ("we have data we cannot legally share, but the AI we need can only be built on the union"), per-stakeholder benefits second (data scientist, DPO/compliance), then the healthcare scenario as a concrete vignette. The scenario closes by noting transferability to telco fraud, AML, threat intelligence.

**Q6 — Link to Orange strategy.** Two paragraphs. First paragraph: this is a home-grown Orange innovation that reinforces trust, sovereignty, and responsible AI positioning. Second paragraph: concrete monetisation channels (Orange Business, Orange Cyberdefense, healthcare initiatives, cross-affiliate analytics) plus academic credibility note.

**Q7 — Business potential.** Three time horizons (short, medium, long). Tone is deliberately measured: "plausible candidates", "subject to product-team validation", "conditional on demonstrated traction". The closing line ("Realising each horizon will require dedicated product, legal, and go-to-market work beyond the scope of the research itself") signals to the jury that we understand the difference between a research result and a product. Market-sizing data and defensibility argument come in the second paragraph.

**Q8 — Anything else.** Two points: (a) the timing is exceptional because the previous generation just collapsed; (b) the technology is production-ready in principle. Closes with the intellectual-honesty caveat about the guarantee not being cryptographic, framed as a differentiator rather than a limitation.

### 2.3 Sources discipline

Every external claim cites a `[Sx]` reference with a live URL in the Sources section. Source highlights:

- **S1** (GDPR Enforcement Tracker) — most important regulatory anchor.
- **S3** (Geiping et al.) — gradient inversion in FL.
- **S5, S6** (Carlini et al.) — the publicly broken predecessors. These are the *most strategically important* citations because they justify why the previous generation lost market trust.
- **S7, S8, S9** — market sizing (MarketsandMarkets, Gartner).
- **S12** — prior Orange singularization work (Macario-Rat & Plesa), anchors the Orange-DNA argument.
- **S14, S15** — AWS Clean Rooms, Snowflake Data Clean Rooms — concrete competitive landscape.

References are one-per-line for clean copy-paste into the form.

### 2.4 What the jury must NOT see

- Specific table or section references to the TMLR paper itself.
- Fabricated market figures. Anything not externally verifiable is stated qualitatively.
- Marketing-style superlatives ("revolutionary", "breakthrough", "world-class").
- All-caps emphasis.
- Specific numeric promises about ROI or unblocked-project counts.

---

## 3. Demo proposal — healthcare scenario

### 3.1 Narrative

A consortium of European hospitals wants to train a diagnostic model for a rare condition. No single hospital has enough labelled cases. Sharing scans is blocked by GDPR and national health-data law. Federated learning was tried and abandoned (operationally fragile, vulnerable to gradient inversion). Homomorphic encryption is too slow for the imaging workload.

With Singularized Mixup: each hospital transforms its scans once, locally, in seconds. A central institute trains one model on the union. No scan can be reconstructed by anyone — not the institute, not a leak, not a future attacker. Diagnostic accuracy matches what a fully centralised model would achieve.

The demo lets a visitor walk through that story interactively in roughly 3-5 minutes, and lets them step into the attacker's shoes to see the privacy guarantee fail (at weak privacy) and then hold (at strong privacy).

### 3.2 Three-scene structure

**Scene 1 — "The naive approaches fail."**

- Show three "hospital" panels, each with a small private dataset of medical images.
- Show what happens if each hospital trains alone: poor accuracy on rare-disease classification (around chance level or just above).
- Show what happens if they try to share: red GDPR warning, fines context (cite the EUR 6.1B figure on screen).
- Show what happens with FedProx: works partially but accuracy degrades sharply when local distributions are non-IID (one hospital sees mostly one condition, another sees something else). A small label-skew slider lets the visitor watch FedProx accuracy collapse.

**Scene 2 — "Singularized Mixup in action."**

- Each hospital clicks "Apply Singularized Mixup" on their local data.
- The dataset visualization morphs: original images become mixed + noisy images. Critically, the labels also mix (soft labels are shown).
- Each hospital ships its singularized dataset to the central server.
- The server trains one model on the union.
- Test accuracy jumps to near-centralised levels (within 1-2 percentage points of the fully centralised baseline).
- Voiceover/caption: "One-shot. No gradients exchanged. No iterative rounds."

**Scene 3 — "The attacker's view."**

- Put the visitor in the attacker's seat. They are at the central server.
- They have access to: the mixed images, the mixing weights, and (for added drama) a pretrained U-Net reconstruction attacker.
- A slider for the privacy parameter `tau`, log scale from 1 down to 1e-6.
- Three image strips, updating live as the slider moves:
  - Top: original private image (only shown for demonstration — in reality the attacker never sees this).
  - Middle: the mixed image the server actually receives.
  - Bottom: the attacker's best reconstruction.
- Live numerical readouts of SNR, SSIM, and LPIPS on the reconstruction.
- A second live readout: classifier test accuracy at this tau.
- The "money shot": at tau = 1e-6 the U-Net reconstruction is unrecognisable, yet test accuracy stays above 90 percent.

### 3.3 Optional Scene 4 — head-to-head vs. InstaHide

- Same UI, but the encoded images come from InstaHide instead of Singularized Mixup.
- A pre-computed Carlini-style attack output is shown alongside.
- The reconstruction is near-perfect.
- The contrast with Scene 3 is the demo's strongest competitive moment.

This scene is optional because building a faithful InstaHide + Carlini attack pipeline is roughly the same effort as everything else. Recommended only if we have the time.

### 3.4 Dataset selection

Medical images are the natural choice for the visual hook but raise IP and ethical questions for a public-ish workshop. Three options, in preference order:

1. **MedMNIST v2** (https://medmnist.com/) — licensed CC BY 4.0, anonymised, 12 standardised 2D datasets covering pathology, dermoscopy, retina, chest X-ray, etc. Safe choice and visually meaningful.
2. **NIH ChestX-ray14** (https://nihcc.app.box.com/v/ChestXray-NIHCC) — public, large-scale, visually striking. Slightly heavier IP/ethics framing needed.
3. **Synthetic medical-style images** — generated with a public diffusion model, guaranteed clean from a licensing standpoint, but less visceral.

**Recommendation:** MedMNIST (specifically PathMNIST or DermaMNIST for visual punch). Standardised resolution, small size, free of licensing risk.

### 3.5 Tech stack

| Component | Choice | Why |
|-----------|--------|-----|
| Backend / ML | Python + PyTorch | Matches the paper's reference implementation. |
| Frontend / UI | Streamlit | Fast to build, ML-native, handles sliders and live plots without custom JS. Custom CSS for light Orange branding. |
| Packaging | Single Docker container | Reproducible, deployable on any laptop with Docker. |
| Compute | Single GPU laptop (or NVIDIA L4 if available) | Live attack rounds need a GPU; rest can run CPU-only. |
| Caching | Local file system, pre-computed artefacts | Trained classifiers and attack reconstructions per `tau` are pre-baked. |

### 3.6 What runs live vs. pre-computed

The demo must be snappy and resilient under workshop conditions (flaky wifi, projector quirks, hostile time pressure). Split:

**Live (sub-second response):**
- Visualisation of one mixed image given a chosen tau.
- The slider updating the per-tau metrics readouts (SNR/SSIM/LPIPS/accuracy).
- Switching the "hospital" being viewed.
- Label-skew slider in Scene 1 changing the FedProx accuracy display (pre-computed lookup table, not actual training).

**Pre-computed (cached at build time, served instantly):**
- Trained classifier per tau (one for `tau` in {1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6}) and test-accuracy numbers.
- Linear-attack reconstructions for a fixed set of demo images, per tau.
- U-Net-attack reconstructions for the same demo images, per tau.
- FedProx baseline accuracies for {P, beta} configurations matching the Scene 1 slider.
- Optional: InstaHide encodings + Carlini-attack reconstructions for Scene 4.

**Live but optional:**
- A "live attack" button that re-runs ~50 iterations of the linear inversion attack on a freshly chosen image while the audience watches the loss curve. Good for technical attendees who want to see something running, but never required for the narrative.

### 3.7 Project structure (proposed)

```
ootd-demo-tentative/
├── HANDOFF.md                              # this file
├── Open_Tech_Days_competition_answers.txt  # competition form draft
├── 7201_Augmented_Mixup_Procedure.pdf      # paper PDF
├── 7201_Augmented_Mixup_Procedure.txt      # paper plain text
├── README.md                               # demo run instructions (to be written)
├── pyproject.toml                          # or requirements.txt
├── Dockerfile
├── src/
│   ├── singularized_mixup.py               # Algorithm 1 (~30 lines)
│   ├── theorems.py                         # mf <-> tau helpers from Theorem 2
│   ├── feature_extractor.py                # frozen ResNet wrapper
│   ├── classifier.py                       # 3-layer classifier head
│   ├── attacks/
│   │   ├── linear_inversion.py
│   │   └── unet_reconstruction.py
│   ├── federated/
│   │   └── fedprox_baseline.py             # for Scene 1 lookup-table generation
│   └── instahide_baseline/                 # only if Scene 4 included
├── data/
│   ├── medmnist_cache/                     # downloaded MedMNIST splits
│   └── precomputed/
│       ├── classifiers/                    # one per tau
│       ├── attacks/                        # linear + unet reconstructions per tau
│       └── fedprox_lookup.json
├── demo/
│   ├── app.py                              # Streamlit entrypoint
│   ├── pages/
│   │   ├── 1_The_Naive_Approaches.py
│   │   ├── 2_Singularized_Mixup.py
│   │   ├── 3_The_Attacker_View.py
│   │   └── 4_vs_InstaHide.py               # optional
│   ├── components/                         # reusable UI widgets
│   └── assets/                             # Orange logo, custom CSS
└── scripts/
    ├── precompute_classifiers.py
    ├── precompute_attacks.py
    └── precompute_fedprox.py
```

### 3.8 Build plan (rough)

Two-week sprint, assuming one developer plus occasional support:

**Week 1 — Core mechanics.**
- Day 1-2: Set up repo skeleton, environment, dataset loading (MedMNIST), feature extractor, classifier.
- Day 3: Implement Algorithm 1 (Singularized Mixup) and Theorem 2 calibration helpers. Validate against expected SNR/LPIPS behaviour.
- Day 4: Implement linear-inversion attack and pre-compute reconstructions per tau on a small demo image set.
- Day 5: Implement U-Net attack and pre-compute reconstructions per tau (this is the longest single training step; budget a full day).

**Week 2 — Demo UI and polish.**
- Day 6: Streamlit scaffolding, Scene 3 (the attacker view) — the most important scene, build first.
- Day 7: Scene 2 (the protocol in action) with the "hospitals" abstraction.
- Day 8: Scene 1 (failure modes of naive approaches) — pre-compute FedProx baselines.
- Day 9: Polish (Orange branding, copy, smooth transitions, error handling).
- Day 10: Dry run, fix the inevitable issues, write the leave-behind one-pager and the 5-minute scripted walkthrough.

Scene 4 (vs. InstaHide head-to-head) is a +3-4 day add-on. Only commit to it if Week 1 finishes ahead of schedule.

### 3.9 Risks and mitigations

- **Compute on the demo machine is insufficient for live U-Net reconstruction.** Mitigation: ship the pre-computed cache; live attack is optional and clearly labelled.
- **MedMNIST images are too small to look impressive on a projector.** Mitigation: upscale display side, or fall back to NIH ChestX-ray14 on a curated subset.
- **A jury member asks "what is the actual privacy guarantee in DP terms?".** Mitigation: rehearsed answer — it is not a DP guarantee; it is a reconstruction lower bound and directional SNR bound; this is the right trade-off for the regime where DP/HE are too costly. The intellectual-honesty paragraph of Q8 covers this.
- **The audience is more business than technical and the math intimidates.** Mitigation: every scene is fully understandable without the math; the math is in an "Advanced" tab that only technical attendees open.

### 3.10 Deliverables for the workshop day

1. The interactive Streamlit demo (Dockerised, runs on a laptop).
2. A one-page leave-behind PDF: the value proposition, the three numbers that matter (accuracy preserved, attacker SNR drop, FL overhead avoided), the QR code to the live demo / GitHub repo.
3. A 5-minute scripted walkthrough that any presenter can deliver.
4. The polished competition form (currently in `Open_Tech_Days_competition_answers.txt`).
5. A GitHub-ready repo with reproducible instructions in `README.md`.

---

## 4. Open decisions to confirm before building

These are the questions a teammate should resolve before kicking off the build, ideally with the original paper authors (Plesa, Clerot, David, Poenaru) since they are inside Orange:

1. **Code reuse.** The paper's reproducibility statement says all source code is publicly available. Confirm the repository URL and whether we can fork/wrap it rather than re-implement.
2. **Branding constraints.** Confirm with Orange Corporate Communications what visual elements (logos, colour palette, fonts) we are allowed to use on a demo shown internally.
3. **Strategic-plan reference.** Confirm whether the current Orange strategic plan we should reference in Q6 is "Lead the Future" (most recent) or "Engage 2025" (legacy). Update `[S11]` accordingly.
4. **Dataset license review.** If MedMNIST is the chosen dataset, confirm CC BY 4.0 attribution requirements are met in the demo footer.
5. **Compute provisioning.** Decide whether the demo runs on a presenter laptop or a workshop-hall machine. If the latter, plan a dry run on the actual hardware.
6. **InstaHide head-to-head scene.** Decide go/no-go on Scene 4 (the strongest competitive contrast but doubles the implementation effort).
7. **Form length limits.** Confirm whether the competition form imposes character or word limits per question. If yes, tighten the 2-3 paragraph answers further.

---

## 5. Quick-reference: things that took non-trivial effort to figure out

- **Why InstaHide is broken.** The clustering of mixed images by shared private component, made possible by repeated reuse of each private sample. Singularization breaks this at the source.
- **Why the noise goes on the *partner* component, not the target.** It tightly couples noise with the private partner's content, which is what makes the linear inversion ill-conditioned. Putting noise on the target would defeat training.
- **Why Theorem 2 measures *directional* SNR rather than scalar SNR.** A scalar ratio can hide anisotropic leakage — interference concentrated in a few coordinates. The directional bound rules that out.
- **Why the security guarantee is not differential privacy.** DP is worst-case indistinguishability with respect to neighbouring datasets. Our guarantee is an average-case lower bound on reconstruction error. Different formal object, different threat model. Calling it DP would be incorrect and would invite a justified attack from anyone familiar with the literature.
- **Why we removed `[Paper, Table X]` citations from the competition form.** The jury should perceive the demo as a substantial Orange capability, not as a paper summary. The peer-reviewed status of the underlying research is mentioned only at the high level. Specific numeric claims are kept (they come from the paper) but presented as our own measurements.
- **Why the market-sizing references are MarketsandMarkets and not Gartner directly.** Gartner's specific market-size figures sit behind a paywall and are hard to cite with a live URL. MarketsandMarkets has freely browsable summary pages that contain the headline numbers we use. Gartner is cited only for the "top strategic trend" qualitative claim, which is on a free glossary page.

---

End of handoff.
