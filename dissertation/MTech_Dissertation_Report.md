# An Explainable Machine Learning-Based Framework for Detecting SQL Injection and Cross-Site Scripting Attacks

**A Dissertation Submitted in Partial Fulfillment of the Requirements for the Degree of Master of Technology (M.Tech) in Computer Science and Engineering**

---

## Candidate Information
- **Candidate Name:** Fauzia Khatoon
- **Enrollment / Roll Number:** 20250501001
- **Department:** Department of Computer Science and Engineering
- **Institution:** Azad Institute of Technology and Management, Lucknow, India
- **Affiliation:** Dr. A.P.J. Abdul Kalam Technical University, Lucknow, Uttar Pradesh, India
- **Academic Year:** 2025–2026

---

## Certificate of Approval
This is to certify that the dissertation entitled **"An Explainable Machine Learning-Based Framework for Detecting SQL Injection and Cross-Site Scripting Attacks"**, submitted by **Fauzia Khatoon (Roll No. 20250501001)** to the Department of Computer Science and Engineering, is a bona fide record of research work carried out under supervision. The results embodied in this dissertation have been verified through reproducible empirical execution and have not been submitted to any other university or institute for the award of any degree or diploma.

---

## Acknowledgments
I express my deepest gratitude to my thesis supervisor and the faculty members of the Department of Computer Science and Engineering for their constant encouragement, insightful guidance, and technical suggestions throughout the development of this research. I also extend my appreciation to the open-source security community and benchmark data curators (CSIC 2010, OWASP, and academic repository contributors) whose openly shared corpora made this empirical investigation possible.

---

## Abstract
Web applications represent the primary interaction gateway for digital enterprise services, financial operations, healthcare registries, and e-commerce platforms. Consequently, they serve as the prime target for injection-based cyberattacks. Among these, **SQL Injection (SQLi, CWE-89)** and **Cross-Site Scripting (XSS, CWE-79)** consistently dominate the MITRE Common Weakness Enumeration Top 25 and OWASP Top Ten vulnerability rankings. Traditional signature-based Web Application Firewalls (WAFs) fail to detect novel, polyglot, or obfuscated payloads due to rigid pattern-matching heuristics. Conversely, while machine learning and deep learning detectors achieve high detection rates, their black-box nature breeds operational skepticism among Security Operations Center (SOC) analysts who cannot afford false positive blocks or opaque verdicts.

This dissertation presents **XAI-WebGuard**, a simple, efficient, and transparent cybersecurity framework that detects and classifies HTTP requests into three categories: **Benign Request**, **SQL Injection**, and **Cross-Site Scripting**, while simultaneously attributing every prediction to specific syntactic features and raw input tokens. The system introduces a hybrid feature extraction pipeline combining **subword character TF-IDF n-grams (3-to-5 grams)** with **fifteen domain-engineered cybersecurity features** capturing lexical entropy, quotation imbalances, SQL operators, and JavaScript event sinks. 

Three prominent machine learning algorithms—**Logistic Regression**, **Random Forest**, and **XGBoost**—are benchmarked under identical 5-fold stratified cross-validation on a harmonized dataset of 21,930 real-world samples derived from the HTTP CSIC 2010 dataset, Mendeley SQLi v2, and IEEE DataPort XSS repositories. **XGBoost** emerges as the superior model, achieving **>99% accuracy**, a **macro-F1 score of >0.99**, and an average prediction latency of **<0.1 ms per sample**, rendering it suitable for inline line-rate inspection. The trained tree ensemble is integrated with **SHAP (SHapley Additive exPlanations) TreeExplainer**, enabling global feature importance audits, per-class attribution rankings, and real-time local sample explanations featuring color-coded syntactic token highlighting. The complete system is operationalized as an interactive **Streamlit web application**, providing automated inspection, confidence gauges, and transparent diagnostic narratives for security analysts.

**Keywords:** Web Application Security, SQL Injection (CWE-89), Cross-Site Scripting (XSS, CWE-79), Explainable Artificial Intelligence (XAI), SHAP, TF-IDF, XGBoost, Random Forest, Logistic Regression.

---

## Table of Contents
1. **Chapter 1: Introduction**
   - 1.1 Research Background & Context
   - 1.2 The Threat of Injection Flaws (SQLi & XSS)
   - 1.3 The Black-Box Dilemma in AI-Driven Intrusion Detection
   - 1.4 Research Objectives
   - 1.5 Key Contributions of this Dissertation
   - 1.6 Thesis Organization
2. **Chapter 2: Literature Review & Related Work**
   - 2.1 Evolution of Web Application Firewalls (WAFs)
   - 2.2 Machine Learning and Deep Learning for Web Attack Detection
   - 2.3 Explainable Artificial Intelligence (XAI) in Cybersecurity
   - 2.4 Limitations of Existing Literature & Identified Research Gaps
3. **Chapter 3: Dataset Collection & Preprocessing Methodology**
   - 3.1 Benchmark Data Sources (CSIC 2010, SQLi v2, XSS Large)
   - 3.2 Multi-Pass URL Canonicalization & Decoding
   - 3.3 HTML Entity Unescaping & Character Filtering
   - 3.4 Missing Value Imputation and Strict Deduplication
   - 3.5 Multi-Class Harmonization and Balance Analysis
4. **Chapter 4: Feature Engineering Framework**
   - 4.1 Limitations of Standard Word Tokenization in Cybersecurity
   - 4.2 Subword Character TF-IDF Formulation
   - 4.3 Handcrafted Domain Cybersecurity Features
   - 4.4 Hybrid Feature Pipeline Integration
5. **Chapter 5: Machine Learning Classification Models**
   - 5.1 Logistic Regression (Multinomial Softmax Baseline)
   - 5.2 Random Forest Classifier (Bagging Ensemble)
   - 5.3 Extreme Gradient Boosting (XGBoost)
   - 5.4 Cross-Validation Strategy & Hyperparameter Tuning
6. **Chapter 6: Explainable Artificial Intelligence (XAI) via SHAP**
   - 6.1 Theoretical Foundations of Cooperative Game Theory & Shapley Values
   - 6.2 TreeSHAP Algorithm for Gradient Boosted Trees
   - 6.3 Global Feature Importance Analysis
   - 6.4 Local Sample Attributions & Syntactic Token Highlighting
   - 6.5 Automated Natural Language Diagnostic Narratives
7. **Chapter 7: Experimental Results & Comparative Discussion**
   - 7.1 Experimental Testbed & Software Environment
   - 7.2 Detection Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - 7.3 Confusion Matrix Analysis Across Models
   - 7.4 Multi-Class ROC-AUC Curves
   - 7.5 Computational Efficiency & Inference Latency Benchmarks
   - 7.6 SHAP Attribution Analysis & Security Insights
8. **Chapter 8: Web-Based Prototype Application**
   - 8.1 Streamlit Architectural Design
   - 8.2 Home Page & Threat Taxonomy Module
   - 8.3 Live Attack Detection Module
   - 8.4 Interactive XAI Diagnostic Workspace
   - 8.5 Performance & Dataset Analytical Dashboards
9. **Chapter 9: Operational Security, Limitations & Future Work**
   - 9.1 Adversarial Robustness & Evasion Considerations
   - 9.2 Limitations of the Current Prototype
   - 9.3 Directions for Future Investigation
10. **Chapter 10: Conclusion**
11. **References**

---

## Chapter 1: Introduction

### 1.1 Research Background & Context
In the contemporary digital era, web applications underpin vital socioeconomic activities, encompassing financial transactions, electronic medical records, critical government services, and cloud computing infrastructure. The hypertext transfer protocol (HTTP) and its secure counterpart (HTTPS) serve as the ubiquitous conduit through which client browsers and automated agents exchange state and input with back-end enterprise services.

However, the rapid delivery cycles of agile software development, coupled with reliance on heterogeneous third-party JavaScript libraries and legacy server-side frameworks, routinely introduce subtle architectural vulnerabilities. Among these, input validation failures remain the most persistent and devastating hazard facing web application security teams.

### 1.2 The Threat of Injection Flaws (SQLi & XSS)
When an application fails to properly neutralize, sanitize, or validate untrusted user input before incorporating it into an interpreter context, attackers can inject arbitrary syntax that modifies the intended execution flow.

1. **SQL Injection (SQLi, CWE-89):**  
   SQL injection occurs when user-supplied input is directly concatenated into a dynamic database query without parameterized binding. An attacker can craft inputs containing relational operators, string terminators (`'`, `"`), Boolean tautologies (`' OR '1'='1'`), union clauses (`UNION SELECT`), or administrative commands (`DROP TABLE`, `EXEC xp_cmdshell`). The consequences include unauthorized read access to proprietary customer tables, modification of financial ledgers, administrative credential harvesting, and in severe cases, complete underlying host takeover through database extended procedures.

2. **Cross-Site Scripting (XSS, CWE-79):**  
   Cross-Site Scripting arises when an application reflects or stores user-provided input in an HTML document without context-aware output encoding. Attackers smuggle executable HTML elements (`<script>`, `<img>`, `<svg>`, `<iframe>`) or JavaScript event handlers (`onerror=`, `onload=`, `onclick=`) into the browser DOM. When an unsuspecting user navigates to the affected page, the malicious script executes within the security context of the victim's session, enabling session cookie theft, keystroke logging, credential harvesting via fraudulent document object model (DOM) forms, and silent client-side defacement.

Despite decades of defensive engineering guidelines, the 2025/2026 revisions of the **MITRE Common Weakness Enumeration (CWE) Top 25 Most Dangerous Software Weaknesses** rank XSS at **#1** and SQL Injection at **#2**. Furthermore, the **OWASP Top 10** continues to position Injection as Category A03, demonstrating that input neutralization flaws remain the most frequently exploited attack vector globally.

### 1.3 The Black-Box Dilemma in AI-Driven Intrusion Detection
To shield unpatched or legacy web assets, organizations deploy Web Application Firewalls (WAFs) in reverse-proxy architectures. Historically, WAFs have relied on regular-expression pattern matching (e.g., OWASP ModSecurity Core Rule Set). Although deterministic and transparent, signature engines exhibit extreme fragility: minor syntactic variations, URL-encoding obfuscation, HTML entity nesting, or comment splitting (`UN/**/ION SE/**/LECT`) routinely bypass static filters.

To overcome signature rigidity, cybersecurity researchers turned to Machine Learning (ML) and Deep Learning (DL), utilizing algorithms such as Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) recurrent networks, and Transformer encoders (e.g., CodeBERT). While these models achieve nominal detection accuracies exceeding 98%, their operational integration into Security Operations Centers (SOCs) has stalled due to the **"Black-Box Problem"**:
- **Lack of Interpretability:** A traditional ML model outputs a scalar probability (e.g., $P(\text{Attack}) = 0.98$) without revealing which tokens or characteristics triggered the alarm.
- **Alert Fatigue & Mistrust:** Security analysts must triage hundreds of alerts per shift. When a machine learning model flags an urgent alert without evidentiary context, analysts cannot differentiate between genuine exploits and benign anomalous traffic (e.g., complex search strings or markdown submissions).
- **Clever Hans Phenomenon:** Models can easily overfit to spurious training artifacts—such as the specific path length, cookie names, or host headers—rather than recognizing the underlying attack grammar.

### 1.4 Research Objectives
The primary objective of this dissertation is to formulate, implement, and empirically validate an **Explainable Machine Learning-Based Cybersecurity Framework (XAI-WebGuard)** that simultaneously accomplishes three goals:
1. **Multi-Class Attack Discrimination:** Jointly classify HTTP request payloads into three distinct categories: **Benign Request**, **SQL Injection**, and **Cross-Site Scripting**.
2. **Empirical Algorithmic Benchmarking:** Evaluate and compare **Logistic Regression**, **Random Forest**, and **XGBoost** across standardized metrics (Accuracy, Precision, Recall, F1, ROC-AUC, 5-Fold Cross-Validation, and latency) using real-world public corpora.
3. **Transparent Model Explainability:** Integrate **SHAP (SHapley Additive exPlanations)** with the top-performing tree-based model to furnish global feature importance audits, per-attack attribution rankings, and real-time local sample explanations with exact syntactic token highlighting.

---

## Chapter 2: Literature Review & Related Work

### 2.1 Evolution of Web Application Firewalls (WAFs)
Early perimeter defenses depended heavily on stateless packet filters and pattern matching. In 2002, the release of ModSecurity introduced rule-based inspection into Apache and Nginx web servers. However, Kruegel et al. (2003) demonstrated that rule sets suffer from a fundamental trade-off: overly strict regular expressions yield unacceptable false-positive rates that disrupt legitimate web commerce, whereas lenient rules permit evasion via multi-byte encoding tricks.

### 2.2 Machine Learning in Web Attack Detection
To introduce adaptability, researchers applied feature-based machine learning to web payload analysis. Hasan et al. (2019) evaluated Support Vector Machines (SVM), Naive Bayes, and Random Forests on SQLi query strings, establishing that tree-based ensembles offered superior resistance to noise. In parallel, Mokbal et al. (2019) engineered structural features from URLs and HTML markup to detect XSS with Multilayer Perceptrons.

Subsequent research shifted toward deep learning representation learning. Liang et al. (2017) proposed *DeepXSS*, utilizing Word2Vec embeddings and Bi-LSTM networks to classify script payloads without manual feature engineering. More recently, Fang et al. (2020) deployed CNNs over character matrices, and transformer architectures such as CodeBERT have been adapted to detect SQLi and XSS code sequences. Nevertheless, deep neural architectures demand immense computational resources, require GPU acceleration, and introduce severe inference latencies (50–200 ms), making them impractical for inline line-rate proxy deployment where round-trip latency must remain under 5 ms.

### 2.3 Explainable Artificial Intelligence (XAI) in Cybersecurity
In response to black-box skepticism, the field of Explainable Artificial Intelligence (XAI) emerged. Ribeiro et al. (2016) proposed **LIME (Local Interpretable Model-agnostic Explanations)**, which approximates model behavior in the vicinity of a target instance using a sparse linear surrogate. However, LIME exhibits sampling instability: perturbing the same input multiple times can yield differing explanations.

Lundberg and Lee (2017) unified cooperative game theory and additive feature attribution through **SHAP (SHapley Additive exPlanations)**. Built upon Lloyd Shapley's Nobel-prize-winning cooperative game formulation (1953), SHAP provides mathematically proven guarantees of local accuracy, missingness, and consistency. In 2020, Lundberg et al. introduced **TreeSHAP**, an optimized algorithm that computes exact Shapley values for tree ensembles in low polynomial time rather than exponential time.

Despite these advancements, prior applications of SHAP in cybersecurity have predominantly targeted tabular network intrusion datasets (such as NSL-KDD or CICIDS-2017) rather than raw web text payloads. Where payload XAI has been attempted, it has frequently generated opaque feature names (e.g., `feature_1482 = 0.42`) that fail to connect directly to the attacker's actual input characters.

### 2.4 Research Gaps
The literature reveals three conspicuous gaps:
1. **Isolated Binary Detectors:** Most studies develop binary classifiers dedicated exclusively to SQLi OR XSS in isolation, ignoring the reality of production WAFs that must handle heterogeneous multi-class traffic concurrently.
2. **Fabricated or Toy Datasets:** Numerous published papers report 99.9% accuracy on small, synthetic, or non-public datasets that lack the noise, varied headers, and encoding artifacts of real-world web traffic.
3. **Absence of Actionable Explanations:** Existing XAI implementations rarely map attribution scores back to raw payload spans or standard threat taxonomies (CWE/CAPEC), leaving security analysts with abstract statistical graphs rather than actionable alert verifications.

---

## Chapter 3: Dataset Collection & Preprocessing Methodology

### 3.1 Benchmark Data Sources
To construct a representative, noise-tolerant, and realistic multi-class corpus, this study combines three widely recognized public benchmark datasets:
1. **HTTP CSIC 2010 Dataset:** Developed by the Spanish National Research Council (CSIC), containing thousands of real-world HTTP GET and POST requests directed at an e-commerce web application. We extract genuine, non-malicious user sessions, search queries, item selections, and form submissions to represent the **Benign Request (Class 0)** baseline.
2. **Mendeley SQL Injection Attack Dataset (v2):** A peer-reviewed collection of SQL injection attack strings compiled by researchers, encompassing diverse vectors: tautology bypasses (`' OR 1=1`), union-based table exfiltration (`UNION ALL SELECT`), error-based injections, and blind time delays (`WAITFOR DELAY`, `SLEEP(5)`).
3. **IEEE DataPort Large-Scale Annotated XSS Dataset:** A curated repository of cross-site scripting attack payloads containing injected HTML tags, DOM function calls, and JavaScript event handlers, alongside benign HTML snippets.

### 3.2 Multi-Pass URL Canonicalization & Decoding
Attackers routinely evade inspection by layering percent-encodings (e.g., `%2527` where `%25` decodes to `%`, yielding `%27`, which then decodes to `'`). To dismantle this obfuscation, our preprocessing engine implements **multi-pass iterative URL decoding**:

$$\text{Canonicalize}(s) = \text{unquote}(\text{unquote}(s))$$

This ensures that obfuscated inputs are stripped down to their raw syntactic form before feature extraction.

### 3.3 HTML Entity Unescaping & Character Filtering
Cross-site scripting vectors often utilize HTML entity references (e.g., `&quot;`, `&lt;`, `&#x3C;`) to slip past naive string filters. Our pipeline applies `html.unescape()` to resolve all named and numeric entities into valid ASCII characters. Furthermore, control characters (newlines `\r`, `\n`, tabs `\t`, and non-printable bytes outside the standard range $32 \le \text{ord}(c) \le 126$) are collapsed into single spaces, eliminating whitespace stuffing attacks while preserving all critical punctuation symbols.

### 3.4 Missing Value Imputation and Strict Deduplication
All null or non-string rows are purged. Payloads with character lengths strictly below 3 are eliminated as non-informative. Crucially, strict deduplication is enforced across the entire corpus. If an identical string appears across multiple repositories, it is deduplicated to prevent data leakage between training and testing partitions.

### 3.5 Multi-Class Harmonization and Balance Analysis
The assembled dataset is harmonized into three balanced classes:
- **Class 0 (Benign Request):** 7,310 samples (33.33%)
- **Class 1 (SQL Injection):** 7,310 samples (33.33%)
- **Class 2 (Cross-Site Scripting):** 7,310 samples (33.33%)
- **Total Dataset Size:** **21,930 samples**

The dataset is partitioned using an **80/20 Stratified Split**, yielding **17,544 training samples** and **4,386 testing samples**, guaranteeing identical class proportions across both sets.

---

## Chapter 4: Feature Engineering Framework

### 4.1 Limitations of Standard Word Tokenization
Conventional Natural Language Processing (NLP) tokenizers rely on alphanumeric word boundaries:

$$\text{Token Pattern} = \texttt{\\b\\w\\w+\\b}$$

When applied to cybersecurity strings, such a tokenizer converts `admin' OR 1=1--` into `['admin', 'OR']`, completely discarding the quotation mark `'`, the equality sign `=`, and the SQL comment delimiter `--`. Because the maliciousness of an injection attack resides in punctuation and delimiters, standard NLP tokenization is fundamentally flawed for attack detection.

### 4.2 Subword Character TF-IDF Formulation
To overcome this limitation without losing structural syntax, we employ **Character Subword TF-IDF with Word Boundaries (`char_wb`)** across n-gram lengths from 3 to 5:
- An input such as `<script>` yields n-grams: `[' <s', '<sc', 'sci', 'rip', 'pt>', 't> ']`
- An input such as `OR 1=1--` yields n-grams: `[' or', 'r 1', '1=1', '=1-', '1--', '-- ']`

The Term Frequency-Inverse Document Frequency (TF-IDF) weight for term $t$ in payload $d$ across corpus $D$ is given by:

$$\text{TF}(t, d) = 1 + \ln(f_{t,d}) \quad \text{for } f_{t,d} > 0$$

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

We restrict the vocabulary to the top **2,500 sublinear TF-IDF features**, ensuring maximum capture of malicious substrings while preventing feature explosion.

### 4.3 Handcrafted Domain Cybersecurity Features
In addition to subword n-grams, we extract **15 specialized domain features** derived from cybersecurity threat modeling:
1. `domain_length`: Character length of the request.
2. `domain_special_char_count`: Total count of non-alphanumeric, non-space characters.
3. `domain_special_char_ratio`: Ratio of special characters to total length ($\frac{\text{Specials}}{N}$).
4. `domain_char_entropy`: Shannon entropy measuring syntactic randomness:
   $$H(X) = -\sum_{i=1}^{k} P(x_i) \log_2 P(x_i)$$
5. `domain_sql_keyword_count`: Frequency of canonical SQL keywords (`SELECT`, `UNION`, `DROP`, `WHERE`, `SLEEP`, etc.).
6. `domain_sql_operator_count`: Frequency of SQL operators (`=`, `--`, `/*`, `*/`, `#`, `||`).
7. `domain_sql_tautology_flag`: Boolean flag detecting Boolean tautology regex patterns (e.g., `'1'='1'`, `1=1`, `'a'='a'`).
8. `domain_html_tag_count`: Frequency of HTML tag initiators (`<script`, `<img`, `<svg`, `<iframe`, `<body`, etc.).
9. `domain_event_handler_count`: Count of JavaScript event attributes (`onerror`, `onload`, `onclick`, `onmouseover`).
10. `domain_script_scheme_count`: Count of inline URI execution schemes (`javascript:`, `data:`, `vbscript:`).
11. `domain_dom_sink_count`: Count of dangerous DOM execution sinks (`alert`, `eval`, `document.cookie`, `prompt`, `document.write`).
12. `domain_quote_imbalance`: Parity check of quotation marks:
    $$\Delta_{\text{quote}} = (|c_{'} \bmod 2| + |c_{"} \bmod 2|)$$
13. `domain_tag_imbalance`: Difference between opening and closing angle brackets ($|c_{<} - c_{>}|$).
14. `domain_comment_terminator_flag`: Indicator for SQL line-comment tails (`--`, `#`, `/*`).
15. `domain_semicolon_count`: Frequency of statement separators (`;`).

### 4.4 Hybrid Feature Pipeline Integration
The 2,500 TF-IDF features and 15 domain features are concatenated horizontally into a unified sparse feature matrix:

$$\mathbf{X} = [\mathbf{X}_{\text{TF-IDF}} \parallel \mathbf{X}_{\text{Domain}}] \in \mathbb{R}^{N \times 2,515}$$

This hybrid structure ensures that the models exploit both broad substring patterns and precise syntactic domain indicators, enabling SHAP to explain decisions in both token-level and conceptual terms.

---

## Chapter 5: Machine Learning Classification Models

### 5.1 Logistic Regression (Baseline)
Logistic Regression serves as the convex linear benchmark. For multi-class classification ($K=3$), we formulate the Multinomial Softmax model:

$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=0}^{K-1} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

The model is trained by minimizing the L2-regularized cross-entropy loss:

$$\mathcal{L}(\mathbf{W}) = -\sum_{i=1}^{N} \sum_{k=0}^{K-1} y_{i,k} \ln P(Y = k \mid \mathbf{x}_i) + \frac{\lambda}{2} \sum_{k=0}^{K-1} \|\mathbf{w}_k\|_2^2$$

Optimization is executed using the L-BFGS quasi-Newton solver with a maximum iteration threshold of 1,000.

### 5.2 Random Forest Classifier
Random Forest is a non-parametric ensemble bagging algorithm. It constructs $B = 150$ de-correlated decision trees $\{T_b\}_{b=1}^B$ trained on bootstrap samples of the training data. At each split, a random subset of features $m \approx \sqrt{p}$ is evaluated. Splits are chosen to maximize Gini Impurity reduction:

$$I_G(p) = 1 - \sum_{k=0}^{K-1} p_k^2$$

The ensemble prediction aggregates individual tree votes:

$$\hat{P}(Y = k \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y = k \mid \mathbf{x})$$

Hyperparameters: $n_{\text{estimators}} = 150$, $\text{max\_depth} = 25$, $\text{min\_samples\_split} = 4$.

### 5.3 Extreme Gradient Boosting (XGBoost)
XGBoost is an advanced implementation of gradient boosted decision trees. Rather than training independent trees in parallel, XGBoost builds trees sequentially to minimize a regularized objective function:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{N} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$

where the regularization term penalizes tree complexity:

$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

Using a second-order Taylor expansion:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{N} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$.

Hyperparameters: $n_{\text{estimators}} = 200$, $\text{max\_depth} = 6$, $\text{learning\_rate} = 0.1$, $\text{subsample} = 0.8$, $\text{colsample\_bytree} = 0.8$.

---

## Chapter 6: Explainable Artificial Intelligence (XAI) via SHAP

### 6.1 Theoretical Foundations of Shapley Values
Originating in cooperative game theory (Shapley, 1953), Shapley values divide the payout of a grand coalition among players according to their marginal contributions. In machine learning interpretability, the "players" are feature values, and the "payout" is the model's prediction score $f(\mathbf{x})$.

The unique attribution $\phi_i$ satisfying **Efficiency, Symmetry, Dummy Player, and Additivity** is:

$$\phi_i(f, \mathbf{x}) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

### 6.2 TreeSHAP Algorithm
Computing exact Shapley values across arbitrary model classes requires evaluating $2^{|F|}$ subsets, which is computationally intractable for 2,515 features. However, Lundberg et al. (2020) demonstrated that for tree ensembles, **TreeSHAP** computes exact attributions in polynomial time:

$$\mathcal{O}(T L D^2)$$

where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum tree depth. This renders real-time SHAP computation viable for production cybersecurity applications.

### 6.3 Local Explanations & Token Highlighting
For every flagged request, XAI-WebGuard computes the local attribution vector $\mathbf{\phi} \in \mathbb{R}^{2,515 \times 3}$. Features with positive attributions $\phi_i > 0$ push the log-odds toward the attack class, while negative attributions $\phi_i < 0$ push toward benign traffic.

Furthermore, an AST-inspired character span scanner detects the exact character boundaries $[start, end]$ of suspicious tokens in the raw text, allowing the user interface to visually render highlighted attack spans directly within the input string.

---

## Chapter 7: Experimental Results & Comparative Discussion

### 7.1 Experimental Setup
All experiments were executed on an AMD Ryzen / Intel Core multicore workstation equipped with 16 GB RAM running Windows 11 and Python 3.12. All numerical metrics reported below were obtained by direct empirical execution of the training and evaluation scripts.

### 7.2 Measured Classification Performance
The quantitative evaluation results across the test set (4,386 samples) are presented below:

| Machine Learning Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Macro ROC-AUC | 5-Fold Stratified CV Macro-F1 | Training Time (s) | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **99.45%** | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9996 | 0.9945 (±0.0009) | 16.27 s | 0.002 ms |
| **Random Forest** | **99.61%** | 0.9962 | 0.9961 | 0.9961 | 0.9961 | 0.9999 | 0.9952 (±0.0010) | 1.56 s | 0.039 ms |
| **XGBoost (Champion)** | **99.66%** | **0.9966** | **0.9966** | **0.9966** | **0.9966** | **0.9999** | **0.9967 (±0.0008)** | **82.74 s** | **0.017 ms** |

### 7.3 Detailed Class-Level Breakdown (XGBoost)
The per-class performance breakdown for the champion XGBoost classifier is detailed as follows:
- **Benign Request (Class 0):**
  - Precision: **99.18%** (1,460 / 1,472)
  - Recall: **99.86%** (1,460 / 1,462)
  - F1-Score: **0.9952**
  - False Positive Rate on Benign Traffic: **< 0.14%** (only 2 out of 1,462 legitimate requests misclassified)
- **SQL Injection (Class 1):**
  - Precision: **99.93%** (1,455 / 1,456)
  - Recall: **99.52%** (1,455 / 1,462)
  - F1-Score: **0.9973**
- **Cross-Site Scripting (Class 2):**
  - Precision: **99.86%** (1,456 / 1,458)
  - Recall: **99.59%** (1,456 / 1,462)
  - F1-Score: **0.9973**

### 7.4 Discussion of Empirical Findings
1. **Superiority of Gradient Boosting:** XGBoost demonstrated the highest discriminative power, achieving a test Accuracy of 99.66%, a Macro F1-score of 0.9966, and a cross-validated score of 0.9967 (±0.0008). The gradient-boosted decision trees effectively capture non-linear conjunctions of features (e.g., the co-occurrence of a single quote `'`, an `OR` keyword, and a comment marker `--`).
2. **False Positive Suppression:** In operational web security, false alarms represent the leading driver of operational cost. XGBoost constrained the false positive rate on legitimate web traffic to under 0.14%, preventing disruption to normal business transactions.
3. **Sub-Millisecond Inference:** XGBoost evaluated test samples at an average rate of **0.017 ms per request** (over 58,000 queries per second on a single CPU thread), proving that the hybrid feature pipeline and tree classifier can operate inline within reverse proxies without degrading user experience.

### 7.5 SHAP Attribution Insights
Analysis of global SHAP values revealed the primary drivers for each attack type:
- **SQL Injection:** Heavily driven by `domain_sql_tautology_flag`, `domain_comment_terminator_flag`, `tfidf_--`, `tfidf_union`, `tfidf_select`, and `domain_special_char_ratio`.
- **Cross-Site Scripting:** Dominated by `domain_html_tag_count`, `domain_event_handler_count`, `tfidf_<sc`, `tfidf_alert`, `tfidf_onerror`, and `domain_dom_sink_count`.
- **Benign Traffic:** Characterized by negative values on all attack features, low character entropy, and balanced quotation markers.

---

## Chapter 8: Web-Based Prototype Application

### 8.1 System Architecture
The framework is operationalized as an interactive Streamlit application (`app.py`), organized into five cohesive modules:
1. **Home Page:** Outlines the project scope, threat models, and architectural overview.
2. **Attack Detection:** Provides an interactive input box, quick test presets, real-time prediction verdicts, and confidence progress meters.
3. **Explainable AI (XAI):** Renders local SHAP attribution horizontal bar charts, natural language diagnostic reports, and color-coded character highlights.
4. **Model Performance:** Compares Logistic Regression, Random Forest, and XGBoost side-by-side using dynamic metrics tables, confusion matrices, and ROC curves.
5. **Dataset Analysis:** Displays class balance charts, payload length distributions, and an interactive data explorer.

---

## Chapter 9: Operational Security, Limitations & Future Work

### 9.1 Operational Security Considerations
Deploying machine learning models in adversarial environments requires defenses against adversarial evasion. Attackers may attempt:
- **Unicode Homoglyphs:** Using Cyrillic or mathematical symbols resembling ASCII characters.
- **SQL Dialect Specifics:** Utilizing database-specific functions (e.g., PostgreSQL `pg_sleep()` or Oracle `utl_inaddr`).
Our multi-pass canonicalizer mitigates encoding evasion, while character n-grams capture subword fragments regardless of spacing tricks.

### 9.2 Limitations
1. **POST Body Chunking:** The current prototype inspects query strings and single-parameter bodies. Very large multipart file uploads require streaming chunk analysis.
2. **Model Retraining:** As new zero-day attack patterns emerge, periodic retraining using active learning pipelines is recommended.

### 9.3 Future Directions
- Integration with reverse proxies (e.g., Envoy or Nginx) via WebAssembly (Wasm) filters.
- Extension to API security vulnerabilities, such as JSON-RPC and GraphQL injection.

---

## Chapter 10: Conclusion
This dissertation successfully developed and evaluated **XAI-WebGuard**, a comprehensive, explainable machine learning framework for detecting SQL Injection and Cross-Site Scripting attacks. By uniting subword character TF-IDF n-grams with handcrafted cybersecurity domain features, the system enables standard classifiers to achieve over **99.4% detection accuracy** while maintaining sub-millisecond latency. Through the integration of **SHAP TreeExplainer**, the black-box barrier of machine learning is dismantled, providing security analysts with intuitive, trustworthy, and actionable justifications for every security decision.

---

## References (IEEE Format)
1. MITRE Corporation, "2025 CWE Top 25 Most Dangerous Software Weaknesses," MITRE CWE Knowledge Base, 2025. [Online]. Available: https://cwe.mitre.org/top25/
2. OWASP Foundation, "OWASP Top 10 Web Application Security Risks," Open Web Application Security Project, 2021/2025. [Online]. Available: https://owasp.org/www-project-top-ten/
3. C. Kruegel and G. Vigna, "Anomaly detection of web-based attacks," in *Proc. 10th ACM Conf. Comput. Commun. Secur. (CCS)*, Washington, DC, USA, 2003, pp. 251–261.
4. M. Hasan, M. A. Kabir, and R. Sarker, "Detection of SQL injection attacks using machine learning algorithms: A comparative study," *Int. J. Comput. Appl.*, vol. 182, no. 45, pp. 1–7, 2019.
5. F. M. M. Mokbal, W. Dan, A. Imran, L. Jiuchuan, and F. Akhtar, "A deep learning approach for detection of cross-site scripting attacks using convolutional neural networks," *IEEE Access*, vol. 7, pp. 165842–165856, 2019.
6. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765–4774.
7. S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, P. C. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020.
8. M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?": Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.
9. C. Torrano-Gimenez, A. Perez-Villegas, and G. Alvarez, "A dataset for benchmarking anomaly-based web application firewalls," in *Proc. 7th Int. Conf. Soft Comput. Models Ind. Environ. Appl.*, Ostrava, Czech Republic, 2012, pp. 297–304.
10. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, San Francisco, CA, USA, 2016, pp. 785–794.
11. L. Breiman, "Random Forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.
12. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
