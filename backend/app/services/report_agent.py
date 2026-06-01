"""
Report Agent service.

Generates simulation-based prediction reports using a ReACT loop with
graph-backed retrieval tools.
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)
from .evidence_evaluator import EvidenceEvaluator

# New imports from report package
from .report import (
    ReportStatus,
    ReportSection,
    ReportOutline,
    Report,
    AnalyticalReport,
    FreshnessClass,
    TemporalAnnotation,
    GroundingReport,
    CredibilityReport,
    ContradictionReport,
    NumericalSanityReport,
    SpecificityReport,
    ConfidenceReport,
    SkepticismReport,
    EvidenceAlignmentReport,
    AnchoringDistributionReport,
    CausalCompletenessReport,
    NumericalGroundingCoverageReport,
    ThesisBalanceReport,
    EpistemicReviewReport,
    ReportLogger,
    ReportConsoleLogger,
    TemporalRelevanceFilter,
    GroundingVerifier,
    SourceCredibilityScorer,
    ContradictionDetector,
    NumericalSanityChecker,
    SpecificityDetector,
    ConfidenceCoverageChecker,
    SkepticismChecker,
    EvidenceAlignmentChecker,
    AnchoringDistributionChecker,
    CausalCompletenessChecker,
    NumericalGroundingCoverageChecker,
    ThesisBalanceChecker,
    EpistemicReviewChecker,
)

logger = get_logger('mirofish.report_agent')

# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# ── Tool descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep Insight Retrieval — powerful retrieval tool]
A powerful retrieval function designed for deep analysis. It will:
1. Automatically decompose your question into multiple sub-questions
2. Retrieve information from the simulation graph across multiple dimensions
3. Integrate semantic search, entity analysis, and relationship chain tracking results
4. Return the most comprehensive and in-depth retrieval content

[When to use]
- Need to deeply analyze a topic
- Need to understand multiple facets of an event
- Need rich supporting material for a report section

[Returns]
- Relevant original facts (can be quoted directly)
- Core entity insights
- Relationship chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Broad Search — full panorama view]
This tool retrieves the complete picture of simulation results, especially suited for understanding how events evolved. It will:
1. Retrieve all relevant nodes and relationships
2. Distinguish current (valid) facts from historical/expired facts
3. Help you understand how sentiment evolved over time

[When to use]
- Need to understand the full timeline of an event
- Need to compare sentiment changes across different phases
- Need comprehensive entity and relationship information

[Returns]
- Current valid facts (latest simulation results)
- Historical/expired facts (evolution record)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\
[Simple Search — quick retrieval]
A lightweight, fast retrieval tool for simple and direct information queries.

[When to use]
- Need to quickly look up a specific piece of information
- Need to verify a fact
- Simple information retrieval

[Returns]
- A list of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[In-Depth Interview — real Agent interviews (dual-platform)]
Calls the OASIS simulation environment's interview API to conduct real interviews with running simulated Agents!
This is not an LLM simulation — it calls the actual interview endpoint to get original Agent responses.
By default it interviews on both Twitter and Reddit to gather broader perspectives.

Workflow:
1. Automatically reads the persona file to learn about all simulated Agents
2. Intelligently selects Agents most relevant to the interview topic (e.g. students, media, officials)
3. Automatically generates interview questions
4. Calls the /api/simulation/interview/batch endpoint for dual-platform interviews
5. Consolidates all interview results for multi-perspective analysis

[When to use]
- Need perspectives from different roles (What do students think? Media? Officials?)
- Need to collect opinions and stances from multiple parties
- Need original answers from simulated Agents (from the OASIS environment)
- Want to make the report vivid with "interview transcripts"

[Returns]
- Identity information of interviewed Agents
- Each Agent's responses on both Twitter and Reddit
- Key quotes (can be cited directly)
- Interview summary and viewpoint comparison

[Important] The OASIS simulation environment must be running to use this feature!"""

TOOL_DESC_WEB_SEARCH = """\
[Web Search — get up-to-date real-time external information]
Searches the web for the latest, real-time news, developments, facts, or data that may not exist in the simulation database.
This is critical for ground-truth external facts, comparisons with real-world events, or locating missing companies and targets.

[When to use]
- Need to look up real-time information or news on the web
- Need external context or up-to-date baseline information (e.g. standard procedures, real-world events)
- When the query is about public facts beyond the simulation graph
- VERY IMPORTANT: Use this tool to find and identify specific under-the-radar companies, micro-cap stocks, components, technical innovators, or bottleneck vendors when the simulation requirement asks for them but they are not fully populated in the simulation graph data.

[Returns]
- Text corpus extracted from highly relevant search results and web pages"""

# ── Outline planning prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert author of "Future Prediction Reports" with a god's-eye view of the simulated world — you can observe every Agent's behavior, statements, and interactions.

[Core concept]
We built a simulated world and injected a specific "simulation requirement" as a variable. The evolution of this simulated world is a prediction of what could happen in the future. You are not observing "experimental data" — you are watching a "rehearsal of the future."

[Your task]
Write a "Future Prediction Report" that answers:
1. Under the conditions we set, what happened in the future?
2. How did the various Agents (populations) react and act?
3. What noteworthy future trends and risks does this simulation reveal?

[Report positioning]
- This is a simulation-based future prediction report revealing "if this happens, what could the future look like"
- Focus on prediction outcomes: event trajectories, group reactions, emergent phenomena, potential risks
- Agent behavior in the simulated world is itself a prediction of future human behavior
- This is NOT an analysis of the current real world
- This is NOT a generic sentiment overview
- CRUCIAL (Epistemic rigor): The report structure MUST NOT be a single unbroken narrative of confirmation. You must plan sections that explicitly explore competing explanations, substitution risks, adaptation effects, or downside/bear scenarios.

[Section count constraints]
- Minimum 2 sections, maximum 5 sections
- No sub-sections needed; each section contains complete content
- Content should be concise, focused on core predictive findings
- You design the section structure based on the prediction results

Output the report outline in JSON format as follows:
{
    "title": "Report title",
    "summary": "Report summary (one sentence summarizing the core predictive finding)",
    "sections": [
        {
            "title": "Section title",
            "description": "Section content description"
        }
    ]
}

Note: the sections array must have at least 2 and at most 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\
[Prediction scenario]
Variable injected into the simulated world (simulation requirement): {simulation_requirement}

[Simulation world scale]
- Number of entities in the simulation: {total_nodes}
- Number of relationships generated: {total_edges}
- Entity type distribution: {entity_types}
- Number of active Agents: {total_entities}

[Sample future facts predicted by the simulation]
{related_facts_json}

From a god's-eye view, examine this future rehearsal:
1. Under the conditions we set, what state does the future present?
2. How did the various populations (Agents) react and act?
3. What noteworthy future trends does this simulation reveal?

Based on the prediction results, design the most suitable report section structure.

[Reminder] Section count: minimum 2, maximum 5. Content should be concise and focused on core predictive findings.
CRUCIAL: Be sure to include at least one section or theme that explicitly explores adversarial challenges, competing explanations, substitution risks, or bear/downside scenarios."""

# ── Section generation prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert author of "Future Prediction Reports," currently writing one section of the report.

Report title: {report_title}
Report summary: {report_summary}
Prediction scenario (simulation requirement): {simulation_requirement}

Current section to write: {section_title}

═══════════════════════════════════════════════════════════════
[Core concept]
═══════════════════════════════════════════════════════════════

The simulated world is a rehearsal of the future. We injected specific conditions
(the simulation requirement) into it, and Agent behavior and interactions are
predictions of future human behavior.

Your task is to:
- Reveal what happened under the set conditions
- Predict how various populations (Agents) reacted and acted
- Identify noteworthy future trends, risks, and opportunities

Do NOT write this as an analysis of the current real world.
Focus on "what the future looks like" — simulation results ARE the predicted future.

═══════════════════════════════════════════════════════════════
[Most important rules — must follow]
═══════════════════════════════════════════════════════════════

1. [You MUST call tools to observe the simulated world]
   - You are observing a future rehearsal from a god's-eye view
   - All content must come from events and Agent behavior in the simulation
   - Do NOT use your own knowledge to write report content
   - Each section must call tools at least 3 times (max 5) to observe the simulated world
   - EXCEPTION FOR EXTERNAL TARGETS: If the "Prediction scenario (simulation requirement)" explicitly asks you to explore, analyze, or identify specific real-world targets (such as specific companies, bottleneck providers, or micro-cap stocks) that are missing or underrepresented in the simulation graph, you are explicitly allowed (and encouraged!) to call `web_search` and utilize the `[Recent external facts (auto-fetched, dated)]` to incorporate these external real-world entities and facts into the report.

2. [You MUST quote Agents' original words and actions]
   - Agent statements and actions are predictions of future human behavior
   - Use block-quote format to present these predictions, e.g.:
     > "A certain group would say: original content..."
   - These quotes are the core evidence of the simulation's predictions

3. [Language consistency — quoted content must match the report language]
   - Tool results may contain text in a language different from the report
   - The entire report must be written in the language specified by the user
   - When quoting tool-returned content in another language, translate it into the report language first
   - Keep the original meaning intact; ensure natural phrasing
   - This rule applies to both body text and block quotes (> format)

4. [Faithfully present prediction results]
   - Report content must reflect the simulation results that represent the future
   - Do not add information that does not exist in the simulation
   - EXCEPTION FOR EXTERNAL TARGETS: If specific companies, stocks, or bottleneck providers are requested by the user's simulation requirement but are not fully populated in the simulation graph, you MUST use `web_search` to find real-world players/companies fitting those descriptions, and present them as the predicted/simulated targets under those conditions.
   - If information is insufficient in some area, state that honestly

═══════════════════════════════════════════════════════════════
[Analytical Scaffold and Epistemic Control Layer]
═══════════════════════════════════════════════════════════════

To ensure rigorous, defensible, and analytical prediction reasoning instead of retail SEO storytelling, you MUST structure your analysis according to the following strict analytical frameworks:

1. [Precision without Provenance Guardrail]
   - All numerical estimates, metrics, and sizing claims MUST explicitly state their derivation.
   - Distinguish strictly between:
     - "Measured/Observed" (e.g., specific agent votes, concrete data points retrieved)
     - "Estimated" (e.g., extrapolated from partial data)
     - "Modeled" (e.g., theoretical impacts derived from causal chains)
   - Do NOT present modeled or estimated figures as exact factual measurements.

2. [Confidence-Governed Synthesis]
   - Your conclusions MUST NOT exceed the strength of your supporting evidence.
   - If the evidence is weak, mixed, or purely speculative, the conclusion must be explicitly weak, caveated, or framed as a hypothesis.
   - Do not use confident, definitive language unless backed by overwhelming, high-epistemic-weight evidence.
   - Follow the [PRE-SYNTHESIS EPISTEMIC MANDATE] and [NARRATIVE WEIGHT ALLOCATION] blocks when present.

3. [Adversarial Reasoning & Counter-Analysis]
   - You MUST subject every primary thesis to adversarial challenge.
   - Explicitly include a counter-analysis block that explores:
     - Alternative explanations for the observed data
     - Substitution risks or adaptation effects (how agents might bypass or mitigate the core event)
     - The Null Hypothesis / reasons the projected impact may not materialize or may not matter
   - Do NOT center the entire section on one salient retrieved theme (Thematic Overfitting). Balance structural drivers, cyclical events, and actively seek disconfirming evidence.

4. [Epistemic Discrimination & Institutional Inference Leakage]
   - Each observation includes an Epistemic tier (SIMULATION_PRIMARY → INSTITUTIONAL → PROFESSIONAL → ANALYST_ACTION → RETAIL_COMMENTARY → SOCIAL_OPINION → SPECULATIVE).
   - NEVER blend tiers at equal narrative weight.
   - Do NOT attribute vague claims to institutional actors, hedge funds, or professional rotation unless directly and specifically observed in the evidence.
   - Strict separation: What is *observed* vs what is *inferred* must be clearly demarcated.

5. [Causal reasoning chain expansion]
   - Every major finding or claim MUST follow a complete causal-chain scaffold:
     `Event / Observation → Operational Impact → Financial/Economic Impact → Strategic Implications → Vulnerability/Risk Invalidation`
   - If a link in the chain is not direct or is missing in the simulation data, explicitly mark it as "unobserved/missing link" or deductively inferred.

6. [Probabilistic Scenario Mapping (Base/Bear/Bull)]
   - You MUST conclude the section with a clear scenario mapping block:
     - **Base Scenario** (the most likely trajectory with rough probability band)
     - **Bear/Risk Scenario** (downside trajectory, probability band)
     - **Bull/Upside Scenario** (upside trajectory, probability band)
     - For each scenario, state the EXACT falsifiable trigger condition (what specific agent actions or metrics would invalidate your thesis and shift the system into a different scenario).

═══════════════════════════════════════════════════════════════
[Temporal Relevance Rules]
═══════════════════════════════════════════════════════════════

Each tool result you receive begins with a [Temporal Context] block generated by the system. You MUST apply the following rules:

1. [Read the Temporal Context header]
   - Check the "Freshness requirement" (HIGH / MEDIUM / LOW) and the "Source freshness score".
   - Note the "Dates found" and "Most recent" date for the retrieved facts.

2. [Cite key claims with a date]
   - Whenever a date is available for a claim, append "as of [date]" — e.g.:
     > "Agent X reported 65% support as of 2024-03-15."
   - If no date is found, omit the tag rather than fabricating one.

3. [Resolve conflicts between old and new sources]
   - If two retrieved sources make contradictory claims, prefer the one with the newer date.
   - Explicitly note the conflict: "Earlier data (as of [old date]) suggested X, but more recent data (as of [new date]) indicates Y."
   - Do NOT silently merge contradictory claims into a single coherent narrative.

4. [Flag stale sources for high-freshness topics]
   - If a [Temporal Context] block shows a staleness warning (⚠️), explicitly acknowledge it.
   - Use language such as: "Note: the available data for [topic] dates to [date]; current conditions may differ."
   - Do NOT present stale data as a current, confirmed fact.

5. [Display "as of [date]" in the final answer]
   - Key statistics, prices, sentiment figures, regulatory status, and other time-sensitive claims
     must carry an "as of [date]" qualifier in the final text visible to the reader.

═══════════════════════════════════════════════════════════════
[Grounding Rules — source citation is mandatory]
═══════════════════════════════════════════════════════════════

Every tool result you receive is labelled [S1], [S2], … [SN] in the observation header.
You MUST ground your Final Answer to these numbered sources using the following rules:

1. [Cite every factual claim]
   - After each specific fact, statistic, quote, or event you take from a source, append the
     source tag immediately: e.g. "Support reached 72% [S2]" or "Agent X said '…' [S1]."
   - Do not group all citations at the end — tag each claim inline.

2. [Do not invent facts beyond the sources]
   - Only assert as fact what appears in [S1]–[SN].
   - If you draw a logical inference beyond the sources, mark it as inference:
     e.g. "This suggests … (inferred, not directly stated in sources)."
   - If you make a speculative projection, mark it explicitly:
     e.g. "Speculatively, … (not evidenced in retrieved sources)."

3. [Handle conflicting sources explicitly]
   - If [S2] contradicts [S1], do not pick one silently. Write:
     "[S1] indicates X, while [S2] — which is more recent — indicates Y. The discrepancy
     may reflect …"

4. [Note unused sources]
   - If a retrieved source did not contribute to this section, briefly note why in your
     `<self_critique>`: e.g. "S3 was retrieved but contained no relevant data for this angle."

5. [Source tags are stripped from the published report]
   - [SN] tags are for internal grounding traceability only; they will be stripped before
     the report is published. Write naturally, as if the tags are footnotes.

═══════════════════════════════════════════════════════════════
[Format rules — extremely important!]
═══════════════════════════════════════════════════════════════

[One section = smallest content unit]
- Each section is the smallest unit of the report
- Do NOT use any Markdown headings (#, ##, ###, ####, etc.) inside a section
- Do NOT add the section title at the beginning of the content
- The section title is added automatically by the system; you only write body text
- Use **bold text**, paragraph breaks, block quotes, and lists to organize content — no headings

[Correct example]
```
This section analyzes the event's public-opinion dynamics. Through deep analysis of the simulation data, we found...

**Initial explosion phase**

Weibo, as the front line of public opinion, served as the core channel for breaking news:

> "Weibo contributed 68% of the initial buzz..."

**Emotion amplification phase**

Short-video platforms further amplified the event's impact:

- Strong visual impact
- High emotional resonance
```

[Incorrect example]
```
## Executive Summary          <- Wrong! Do not add any headings
### 1. Initial Phase          <- Wrong! Do not use ### for sub-sections
#### 1.1 Detailed Analysis    <- Wrong! Do not use #### for sub-sub-sections

This section analyzes...
```

═══════════════════════════════════════════════════════════════
[Available retrieval tools] (call 3–5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool usage tips — mix different tools, do not rely on just one]
- insight_forge: Deep insight analysis; auto-decomposes questions and retrieves facts and relationships across dimensions
- panorama_search: Wide-angle panoramic search; understand the full picture, timeline, and evolution
- quick_search: Quickly verify a specific data point
- interview_agents: Interview simulated Agents to get first-person perspectives and real reactions from different roles
- web_search: Searches the web for the latest, real-time news, developments, facts, or data (highly recommended to use at most 1–2 times to preserve your tool budget)

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

Each reply, you may do only ONE of the following (never both):

Option A — Call a tool:
Output your thinking, then call a tool using this format:
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return the result. You cannot and should not write tool results yourself.

Option B — Output final content:
When you have gathered enough information via tools, you MUST first perform a post-generation analytical self-critique pass.
First, output your self-critique inside a `<self_critique>` block assessing your assumptions, analytical gaps, counterarguments/alternative scenarios, and confidence calibration.
Then, immediately output your polished, analytical, and calibrated section content starting with "Final Answer:".

Example format:
<self_critique>
- Adversarial Challenge (Alternative explanations / Null Hypothesis): ...
- Provenance Check (Measured vs Estimated vs Modeled): ...
- Epistemic Leakage Check (Are we attributing to institutions without direct proof?): ...
- Confidence Calibration (Is our certainty matched by evidence weight?): ...
</self_critique>

Final Answer:
[Your analytical section content here...]

Strictly prohibited:
- Do NOT include both a tool call and a Final Answer in the same reply
- Do NOT fabricate tool results (Observations) — all tool results are injected by the system
- At most one tool call per reply

═══════════════════════════════════════════════════════════════
[Section content requirements]
═══════════════════════════════════════════════════════════════

1. Content must be based on simulation data retrieved via tools
2. Extensively quote original text to demonstrate simulation outcomes
3. Use Markdown formatting (but NO headings):
   - Use **bold text** to mark key points (instead of sub-headings)
   - Use lists (- or 1. 2. 3.) to organize points
   - Use blank lines to separate paragraphs
   - Do NOT use #, ##, ###, #### or any heading syntax
4. [Block-quote format — must be standalone paragraphs]
   Quotes must be standalone paragraphs with a blank line before and after:

   Correct format:
   ```
   The institution's response was seen as lacking substance.

   > "The institution's response pattern appeared rigid and slow in the fast-moving social media landscape."

   This assessment reflects widespread public dissatisfaction.
   ```

   Incorrect format:
   ```
   The institution's response was seen as lacking substance. > "The institution's response pattern..." This assessment reflects...
   ```
5. Maintain logical coherence with other sections
6. [Avoid repetition] Read the completed sections below carefully; do not repeat the same information
7. [Emphasis] Do NOT add any headings! Use **bold text** instead of sub-headings"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed sections so far (read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important reminders]
1. Read the completed sections above carefully to avoid repeating the same content!
2. You must call tools to retrieve simulation data before writing
3. Mix different tools; do not rely on just one
4. Report content must come from retrieval results; do not use your own knowledge

[Format warning — must follow]
- Do NOT write any headings (#, ##, ###, #### are all prohibited)
- Do NOT write "{section_title}" as the opening line
- The section title is added automatically by the system
- Write body text directly; use **bold text** instead of sub-headings

Begin:
1. First think (Thought) about what information this section needs
2. Then call a tool (Action) to retrieve simulation data
3. Once you have enough information, write a `<self_critique>` block assessing your assumptions, adversarial challenges, provenance of numbers, and confidence calibration, and then output your "Final Answer:" (pure body text, no headings)"""

COMPOSER_USER_TEMPLATE = """\
You are writing a section of a "Future Prediction Report".
Your primary task is to write polished, publication-ready Markdown text in English for the section: {section_title}.

[Simulation Requirement / Scenario Condition]
{simulation_requirement}

[Observations / Ground Truth Facts]
Below is the factual evidence retrieved from the simulation graph and real-time searches:
{observations}

[Initial Draft]
Here is an initial draft generated by the planning agent:
{draft}

[Writing Instructions]
1. Combine the ground truth facts and the initial draft into a highly professional, cohesive analysis section.
2. Focus on "what the future looks like" under the simulated scenario. Do not treat this as a study of current real-world state, but as prediction results.
3. You MUST extensively quote simulated Agents' original statements or key events from the observations in standalone blockquotes to back up the prediction (e.g. > "Statement"). Keep quotes word-for-word correct.
4. Keep the writing style analytical, objective, and expert.
5. Strictly do NOT use any headings (#, ##, ###, ####, etc.) inside the section. Instead, use **bold text**, paragraph breaks, and lists to organize your prose.
6. The entire section must be in English.
7. Only output the section body text itself. Do not add conversational intro/outro, and do not prefix with "Final Answer:" or any other wrapper.
8. [Epistemic and Analytical Rigor]
   - Preserve the rigorous analytical, probabilistic, and uncertainty-aware tone of the draft.
   - Maintain a clear distinction between verified facts, logical inferences, and speculative scenarios.
   - Retain explicit confidence calibration, alternative scenarios, falsifiability metrics, and risk analysis without smoothing them over into generic overconfident retail prose.
   - Anti-Anchoring Guardrail: Do NOT let a single highly salient retrieved event or agent statement dominate the final narrative unless corroborated by other high-credibility sources. Weight your arguments proportionally based on the source quality weights in the audit/digest.
   - Enforce quantitative grounding: Ensure central assertions cite revenue numbers, cost impacts, market sizing, or exact probability weights. If missing, explicitly mention that "the exact valuation impact remains unobserved in simulation".
9. [Temporal Accuracy]
   - Each observation block starts with a [Temporal Context] header. Use it to determine data freshness.
   - Append "as of [date]" to key statistics, sentiment figures, prices, and regulatory facts when a date is available.
   - If the draft contains conflicting claims from different dates, keep the most recent one and note the discrepancy inline.
   - If a source carries a staleness warning (⚠️) for a high-freshness topic, include a visible caveat in the prose, e.g., "Note: this data dates to [date]; current conditions may differ."
   - Do NOT smooth over temporal conflicts — contradictions are analytically significant and must be surfaced.
10. [Grounding — remove [SN] tags, preserve grounding intent]
    - The draft contains inline [S1], [S2], … citation tags. Strip all [SN] tags from the polished output.
    - Before stripping a tag, verify the claim it annotates is genuinely supported by the observations above. If a claim appears in the draft but has no corresponding evidence in the observations, either remove the claim or clearly label it as inference/speculation.
    - If the grounding note below warns of low coverage or unanchored sentences, actively fix them: either anchor each claim to an observation or reframe as inference.

[Analytical Audit & Critique Pass — CRITICAL REQUIREMENT]
Below is an automated audit pointing out exact analytical defects in the draft. You MUST correct each and every issue mentioned here:
{analytical_issues}

Critic & Revision Rules:
1. You MUST remove and rewrite any sentences containing vague phrases from the word list ("monitor closely", "potential opportunities", "careful management", "navigate uncertainty", "could pose risks", "may impact", "in the long run", "going forward"). Do NOT use these vague filler phrases verbatim unless they are part of a direct quote from a source.
2. Resolve any internal contradictions mentioned in the audit. Do not smooth them over; explain them or select the newer/better-grounded evidence.
3. Correct all numerical sanity anomalies (e.g., breakdown percentages that don't sum to ~100%, or order-of-magnitude mismatches vs. source observations).
4. Explicitly add uncertainty labels and confidence levels (e.g. "with high confidence", "moderately likely", "highly speculative") to speculative claims.
5. Back up any highly confident claims with source citation references before stripping the citation tags.
6. Preserve proper analytical hedging; do NOT convert a conditional/hedged statement into an overconfident claim.

[Confidence-Governed Synthesis — mandatory behavioral enforcement]
{confidence_governance}
- Match assertion strength, phrasing intensity, and recommendation aggressiveness to the section confidence ceiling above.
- Weight narrative emphasis by synthesis_weight allocation — not by retrieval salience or headline drama.
- Downrank tentative-only sources in conclusions; never promote social/retail commentary to strategic certainty.
"""

CONFIDENCE_REWRITE_USER_TEMPLATE = """\
You are performing a mandatory confidence-adjusted rewrite of one report section.

Section: {section_title}

[Simulation scenario]
{simulation_requirement}

[Ground-truth observations]
{observations}

[Current section — overconfident or mis-calibrated]
{current_content}

[Automated audit — every item MUST be fixed in the rewrite]
{analytical_issues}

[Confidence-governed synthesis rules]
{confidence_governance}

[Rewrite instructions — aggressive enforcement]
1. Rewrite the entire section to comply with the confidence ceiling and narrative weight allocation.
2. Replace definitive language backed by weak/low-weight sources with calibrated hedging and explicit uncertainty.
3. Demote episodic headlines (mergers, spikes, viral reactions) to context; elevate structural, high-weight drivers.
4. Add missing probability bands, sensitivity caveats, and quantified ranges where the audit flags gaps.
5. Resolve contradictions and numerical anomalies — do not smooth them away silently.
6. Preserve block quotes and simulation-grounded facts; strip all [SN] citation tags from output.
7. Do NOT use Markdown headings (# ## ###). Use **bold** for emphasis only.
8. Output ONLY the revised section body — no preamble, no "Final Answer:", no critique block.
"""

# ── ReACT loop message templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval result):

═══ Tool {tool_name} returned — [S{source_num}] ═══
{result}

═══════════════════════════════════════════════════════════════
Tools called {tool_calls_count}/{max_tool_calls} times (used: {used_tools_str}){unused_hint}
- When you cite a fact from this result in your Final Answer, tag it with [S{source_num}] directly after the claim.
- If you have enough information: output a `<self_critique>` block assessing assumptions, adversarial challenges, provenance of numbers, and confidence calibration, and then output your final content starting with "Final Answer:" (must quote the original text above, with [SN] citations)
- If you need more information: call another tool to continue retrieval
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Note] You have only called tools {tool_calls_count} time(s); at least {min_tool_calls} are required. "
    "Please call more tools to gather additional simulation data before outputting Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Only {tool_calls_count} tool call(s) so far; at least {min_tool_calls} are required. "
    "Please call a tool to retrieve simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}); no more tool calls allowed. "
    "Please perform an analytical critique in a `<self_critique>` block (covering adversarial challenge, provenance, epistemic leakage, and confidence calibration), and then immediately output section content starting with \"Final Answer:\" based on the information gathered."
)

REACT_UNUSED_TOOLS_HINT = "\nTip: You have not used: {unused_list}. Consider trying different tools for multi-angle information."

REACT_FORCE_FINAL_MSG = "Tool call limit reached. Please output a <self_critique> block (covering adversarial challenge, provenance, and calibration), and then output Final Answer: and generate the section content now."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient simulation-prediction assistant.

[Background]
Prediction conditions: {simulation_requirement}

[Generated analysis report]
{report_content}

[Rules]
1. Answer questions primarily based on the report content above
2. Answer directly; avoid lengthy reasoning
3. Only call tools when the report content is insufficient to answer
4. Keep answers concise, clear, and well-organized

[Available tools] (use only when needed, max 1–2 calls)
{tools_description}

[Tool call format]
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>

[Answer style]
- Be concise and direct
- Use > format to quote key content
- Lead with the conclusion, then explain the reasoning"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent main class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """Generates prediction reports using a ReACT loop over graph-backed tools."""
    
    MAX_TOOL_CALLS_PER_SECTION = 5
    MAX_REFLECTION_ROUNDS = 3
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None,
        composer_llm: Optional[LLMClient] = None
    ):
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        self.composer_llm = composer_llm or LLMClient.for_composer()
        self._web_search_corpus = None
        self._temporal_filter = TemporalRelevanceFilter()
        self._grounding_verifier = GroundingVerifier()
        self._credibility_scorer = SourceCredibilityScorer()
        self._contradiction_detector = ContradictionDetector()
        self._numerical_sanity_checker = NumericalSanityChecker()
        self._specificity_detector = SpecificityDetector()
        self._confidence_checker = ConfidenceCoverageChecker()
        self._evidence_evaluator = EvidenceEvaluator()
        self._skepticism_checker = SkepticismChecker()
        self._evidence_alignment_checker = EvidenceAlignmentChecker()
        self._anchoring_distribution_checker = AnchoringDistributionChecker()
        self._causal_completeness_checker = CausalCompletenessChecker()
        self._numerical_grounding_coverage_checker = NumericalGroundingCoverageChecker()
        self._thesis_balance_checker = ThesisBalanceChecker()
        self._epistemic_review_checker = EpistemicReviewChecker()

        self.tools = self._define_tools()
        self.report_logger: Optional[ReportLogger] = None
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(t('report.agentInitDone', graphId=graph_id, simulationId=simulation_id))
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to analyze in depth",
                    "report_context": "Current report section context (optional; helps generate more precise sub-questions)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement (e.g. 'Understand student opinions on the dormitory formaldehyde incident')",
                    "max_agents": "Maximum number of Agents to interview (optional, default 5, max 10)"
                }
            },
            "web_search": {
                "name": "web_search",
                "description": TOOL_DESC_WEB_SEARCH,
                "parameters": {
                    "query": "The web search query string (e.g. 'FDA formaldehyde standards in wood')"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        logger.info(t('report.executingTool', toolName=tool_name, params=parameters))
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Panorama search - get the full picture
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Quick search - fast retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            elif tool_name == "web_search":
                query = parameters.get("query", parameters.get("q", ""))
                if not query:
                    return "Error: web_search requires a 'query' parameter."
                
                from .web_search import search_queries_to_corpus, web_search_configured
                if not web_search_configured():
                    return "Error: Web search is not configured or enabled."
                
                try:
                    search_corpus, _ = search_queries_to_corpus(
                        [query],
                        simulation_requirement=self.simulation_requirement,
                        max_chars=8000
                    )
                    return search_corpus if search_corpus.strip() else "No relevant search results found."
                except Exception as e:
                    logger.error(f"Web search tool failed: {e}")
                    return f"Web search failed: {str(e)}"
            
            # Legacy tool names — redirect to new tools
            
            elif tool_name == "search_graph":
                
                logger.info(t('report.redirectToQuickSearch'))
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                
                logger.info(t('report.redirectToInsightForge'))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Please use one of: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(t('report.toolExecFailed', toolName=tool_name, error=str(e)))
            return f"Tool execution failed: {str(e)}"
    
    
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents", "web_search"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response. Supports <tool_call> XML and bare JSON."""
        tool_calls = []

        # Format 1: XML-style (standard)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback — bare JSON without <tool_call> tags
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Response may contain reasoning text + bare JSON; try extracting the last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate that parsed JSON is a valid tool call."""
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize key names to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)

    def _build_evidence_guidance(
        self,
        evidence_scores: List[Any],
        section_title: str,
        *,
        include_mandate: bool = False,
    ) -> str:
        """Append ranked digest and pre-synthesis mandate to ReACT user messages."""
        parts: List[str] = []
        concentration_warning = self._evidence_evaluator.detect_concentration(evidence_scores)
        if concentration_warning:
            parts.append(concentration_warning)
        if len(evidence_scores) >= 2:
            parts.append(self._evidence_evaluator.build_epistemic_hierarchy(evidence_scores))
        if len(evidence_scores) >= 3:
            parts.append(self._evidence_evaluator.build_digest(evidence_scores))
        if include_mandate or len(evidence_scores) >= 2:
            mandate = self._evidence_evaluator.build_synthesis_mandate(
                evidence_scores, section_title=section_title
            )
            if mandate:
                parts.append(mandate)
        elif len(evidence_scores) == 1:
            profile = self._evidence_evaluator.compute_section_profile(evidence_scores)
            parts.append(self._evidence_evaluator.build_confidence_governed_block(profile))
        if not parts:
            return ""
        return "\n\n" + "\n\n".join(parts)

    def _build_confidence_governance_block(
        self,
        evidence_scores: Optional[Dict[int, Any]],
    ) -> str:
        if not evidence_scores:
            return (
                "(No evidence scores — default to hedged, simulation-grounded language; "
                "avoid definitive strategic claims.)"
            )
        scores = list(evidence_scores.values())
        if not scores:
            return "(No evidence scores — use exploratory tone.)"
        profile = self._evidence_evaluator.compute_section_profile(scores)
        parts = [
            self._evidence_evaluator.build_confidence_governed_block(profile),
            self._evidence_evaluator.build_narrative_weight_allocation(scores),
        ]
        return "\n".join(parts)

    def _compose_final_section(
        self,
        section: ReportSection,
        draft: str,
        observation_log: List[str],
        analytical_report: Optional[AnalyticalReport] = None,
        evidence_scores: Optional[Dict[int, Any]] = None,
    ) -> str:
        """Use the composer LLM to write the final polished markdown for the section."""
        if not Config.REPORT_USE_COMPOSER:
            return draft

        logger.info(f"[Composer] Composing final polished markdown for section: {section.title}")

        composer_system = (
            "You are a senior analyst composing one section of a prediction report.\n"
            "Write polished markdown only. Do not call tools, do not output JSON.\n"
            "Never use Markdown headings (#, ##, ###) in your response; use **bold text** for sub-sections instead.\n"
            "Confidence-governed synthesis is mandatory: match tone and recommendations to evidence weights and ceilings.\n"
            "Keep the section highly focused and concise (target 800-1200 words). Do not repeat observations endlessly.\n"
            f"{get_language_instruction()}"
        )

        if analytical_report:
            if isinstance(analytical_report, AnalyticalReport):
                analytical_issues = analytical_report.to_composer_block_with_evidence(
                    evidence_scores
                )
            else:
                analytical_issues = f"[Grounding note — action required]\n{analytical_report.summary}"
                if getattr(analytical_report, 'warning', None):
                    analytical_issues += f"\nWarning: {analytical_report.warning}"
                if getattr(analytical_report, 'high_risk_sentences', None):
                    analytical_issues += "\nUnanchored sentences:\n" + "\n".join(f"  - {s}" for s in analytical_report.high_risk_sentences)
        else:
            analytical_issues = "(No automated audit results available. Ensure general analytical rigor and grounding.)"

        confidence_governance = self._build_confidence_governance_block(evidence_scores)

        # Truncate observation log to prevent exceeding context window of local models
        joined_observations = "\n\n---\n\n".join(observation_log) if observation_log else "(No tool observations recorded for this section)"
        # Roughly 16k chars ~ 4k tokens, leaves plenty of room for 8k context models
        if len(joined_observations) > 16000:
            joined_observations = joined_observations[:16000] + "\n\n... [Observations truncated due to length] ..."

        composer_user = COMPOSER_USER_TEMPLATE.format(
            section_title=section.title,
            simulation_requirement=self.simulation_requirement,
            observations=joined_observations,
            draft=draft,
            analytical_issues=analytical_issues,
            confidence_governance=confidence_governance,
        )

        try:
            final_answer = self.composer_llm.chat(
                messages=[
                    {"role": "system", "content": composer_system},
                    {"role": "user", "content": composer_user},
                ],
                temperature=0.4,
                max_tokens=Config.LLM_CHAT_MAX_TOKENS,
            )
            if final_answer and final_answer.strip():
                cleaned = final_answer.strip()
                cleaned = re.sub(r'^#+\s+.*?\n', '', cleaned)
                return cleaned.strip()
            return draft
        except Exception as e:
            logger.error(f"[Composer] Hand-off failed, falling back to original draft: {e}")
            return draft

    def _confidence_adjusted_rewrite(
        self,
        section: ReportSection,
        content: str,
        analytical_report: AnalyticalReport,
        observation_log: List[str],
        evidence_scores: Optional[Dict[int, Any]] = None,
    ) -> str:
        """Second-pass rewrite when audit flags overconfidence or epistemic collapse."""
        if not Config.REPORT_USE_COMPOSER:
            return content

        logger.info(
            f"[ConfidenceRewrite] Section '{section.title}' severity={analytical_report.severity_score()}"
        )

        rewrite_system = (
            "You are a senior analyst enforcing confidence-governed analytical reasoning.\n"
            "Rewrite to fix every audit issue; downgrade overconfident prose aggressively.\n"
            "Never use Markdown headings. Output section body only.\n"
            "Keep the output highly focused and concise (target 800-1200 words).\n"
            f"{get_language_instruction()}"
        )

        confidence_governance = self._build_confidence_governance_block(evidence_scores)
        analytical_issues = analytical_report.to_composer_block_with_evidence(evidence_scores)

        # Truncate observation log to prevent exceeding context window
        joined_observations = "\n\n---\n\n".join(observation_log) if observation_log else "(No observations)"
        if len(joined_observations) > 16000:
            joined_observations = joined_observations[:16000] + "\n\n... [Observations truncated due to length] ..."

        rewrite_user = CONFIDENCE_REWRITE_USER_TEMPLATE.format(
            section_title=section.title,
            simulation_requirement=self.simulation_requirement,
            observations=joined_observations,
            current_content=content,
            analytical_issues=analytical_issues,
            confidence_governance=confidence_governance,
        )

        try:
            revised = self.composer_llm.chat(
                messages=[
                    {"role": "system", "content": rewrite_system},
                    {"role": "user", "content": rewrite_user},
                ],
                temperature=0.25,
                max_tokens=Config.LLM_CHAT_MAX_TOKENS,
            )
            if revised and revised.strip():
                cleaned = re.sub(r'^#+\s+.*?\n', '', revised.strip())
                return cleaned.strip()
        except Exception as e:
            logger.error(f"[ConfidenceRewrite] Failed for {section.title}: {e}")
        return content

    def _finalize_section_with_governance(
        self,
        section: ReportSection,
        draft: str,
        observation_log: List[str],
        audit_report: AnalyticalReport,
        tool_calls_count: int,
        sources_metadata: List[Tuple[int, str, str]],
        evidence_scores: Optional[Dict[int, Any]],
        section_index: int,
    ) -> str:
        """Composer polish, optional confidence rewrite, and audit metrics."""
        final_answer = self._compose_final_section(
            section,
            draft,
            observation_log,
            audit_report,
            evidence_scores=evidence_scores,
        )

        post_audit = self._run_analytical_audit(
            final_answer,
            tool_calls_count,
            sources_metadata,
            section.title,
            evidence_scores=evidence_scores,
        )
        combined_severity = max(audit_report.severity_score(), post_audit.severity_score())
        if combined_severity >= 3 or post_audit.requires_confidence_rewrite():
            merged = AnalyticalReport(
                grounding=post_audit.grounding,
                credibility=post_audit.credibility,
                contradictions=post_audit.contradictions,
                numerical_sanity=post_audit.numerical_sanity,
                specificity=post_audit.specificity,
                confidence=post_audit.confidence,
                evidence_alignment=post_audit.evidence_alignment,
                anchoring=post_audit.anchoring,
                causal_completeness=post_audit.causal_completeness,
                numerical_grounding_coverage=post_audit.numerical_grounding_coverage,
                thesis_balance=post_audit.thesis_balance,
                skepticism=post_audit.skepticism,
                epistemic_review=post_audit.epistemic_review,
            )
            if merged.requires_confidence_rewrite(threshold=2):
                final_answer = self._confidence_adjusted_rewrite(
                    section,
                    final_answer,
                    merged,
                    observation_log,
                    evidence_scores,
                )

        self._log_analytical_audit_metrics(
            section.title,
            section_index,
            audit_report,
            final_answer,
            tool_calls_count,
            sources_metadata,
        )
        return final_answer

    def _select_analytical_module(self, section_title: str) -> Any:
        text = (self.simulation_requirement + " " + section_title).lower()
        
        from .analytical_modules import BaseAnalyticalModule
        from .analytical_modules.markets import MarketsAnalyticalModule
        from .analytical_modules.policy import PolicyAnalyticalModule
        from .analytical_modules.organizational_risk import OrganizationalRiskAnalyticalModule
        
        if any(kw in text for kw in ["market", "price", "stock", "bond", "crypto", "currency", "financial", "earnings", "investment", "portfolio", "trading"]):
            return MarketsAnalyticalModule()
            
        if any(kw in text for kw in ["policy", "regulation", "law", "bill", "act", "compliance", "regulatory", "sanction", "statute", "enforcement", "government"]):
            return PolicyAnalyticalModule()
            
        if any(kw in text for kw in ["risk", "operational", "vulnerability", "failure", "continuity", "supply chain", "incident", "disaster", "crisis"]):
            return OrganizationalRiskAnalyticalModule()
            
        return BaseAnalyticalModule()

    def _run_analytical_audit(
        self,
        final_answer: str,
        tool_calls_count: int,
        sources_metadata: List[Tuple[int, str, str]],
        section_title: str = "",
        evidence_scores: Optional[Dict[int, Any]] = None,
    ) -> AnalyticalReport:
        module = self._select_analytical_module(section_title)
        grounding = self._grounding_verifier.check(final_answer, tool_calls_count)
        credibility = self._credibility_scorer.check(final_answer, sources_metadata)
        contradictions = self._contradiction_detector.check(final_answer)
        numerical_sanity = self._numerical_sanity_checker.check(final_answer, sources_metadata, module)
        specificity = self._specificity_detector.check(final_answer, module)
        confidence = self._confidence_checker.check(final_answer)

        # Reconstruct evidence_scores on the fly if not provided
        if evidence_scores is None:
            evidence_scores = {}
            for num, tool_name, result in sources_metadata:
                fresh_score = 0.7
                try:
                    fc = self._temporal_filter.classify_topic(f"{self.simulation_requirement} {section_title}")
                    dates = self._temporal_filter.extract_dates(result)
                    fresh_score, _ = self._temporal_filter.score_freshness(dates, fc)
                except Exception:
                    pass
                score = self._evidence_evaluator.score_evidence(
                    source_num=num,
                    raw_result=result,
                    tool_name=tool_name,
                    section_title=section_title,
                    simulation_requirement=self.simulation_requirement,
                    freshness_score=fresh_score
                )
                evidence_scores[num] = score

        evidence_alignment = self._evidence_alignment_checker.check(
            final_answer, sources_metadata, evidence_scores
        )
        anchoring = self._anchoring_distribution_checker.check(
            final_answer, evidence_scores
        )
        causal_completeness = self._causal_completeness_checker.check(
            final_answer, self.simulation_requirement, section_title
        )
        numerical_grounding_coverage = self._numerical_grounding_coverage_checker.check(
            final_answer, self.simulation_requirement, section_title
        )

        skepticism = self._skepticism_checker.check(final_answer, evidence_scores)
        thesis_balance = self._thesis_balance_checker.check(
            final_answer, self.simulation_requirement, section_title
        )
        epistemic_review = self._epistemic_review_checker.check(final_answer, evidence_scores)

        return AnalyticalReport(
            grounding=grounding,
            credibility=credibility,
            contradictions=contradictions,
            numerical_sanity=numerical_sanity,
            specificity=specificity,
            confidence=confidence,
            evidence_alignment=evidence_alignment,
            anchoring=anchoring,
            causal_completeness=causal_completeness,
            numerical_grounding_coverage=numerical_grounding_coverage,
            thesis_balance=thesis_balance,
            skepticism=skepticism,
            epistemic_review=epistemic_review,
        )

    def _log_analytical_audit_metrics(
        self,
        section_title: str,
        section_index: int,
        initial_report: AnalyticalReport,
        final_answer: str,
        tool_calls_count: int,
        sources_metadata: List[Tuple[int, str, str]]
    ):
        if not self.report_logger:
            return

        final_audit = self._run_analytical_audit(final_answer, tool_calls_count, sources_metadata, section_title)
        
        initial_vague = len(initial_report.specificity.vague_sentences)
        final_vague = len(final_audit.specificity.vague_sentences)
        
        initial_contra = len(initial_report.contradictions.contradictions)
        final_contra = len(final_audit.contradictions.contradictions)
        
        initial_anom = len(initial_report.numerical_sanity.anomalies)
        final_anom = len(final_audit.numerical_sanity.anomalies)
        
        initial_uncalibrated = len(initial_report.confidence.uncalibrated_speculations)
        final_uncalibrated = len(final_audit.confidence.uncalibrated_speculations)
        
        vague_phrases_removed = max(0, initial_vague - final_vague)
        contradictions_resolved = max(0, initial_contra - final_contra)
        confidence_labels_added = max(0, initial_uncalibrated - final_uncalibrated)
        
        self.report_logger.log(
            action="analytical_audit",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "vague_phrases_removed": vague_phrases_removed,
                "contradictions_flagged": initial_contra,
                "contradictions_resolved": contradictions_resolved,
                "numerical_anomalies": final_anom,
                "confidence_labels_added": confidence_labels_added,
                "message": f"Analytical audit completed: {vague_phrases_removed} vague phrases removed, {contradictions_resolved}/{initial_contra} contradictions resolved, {final_anom} numerical anomalies remaining, {confidence_labels_added} confidence labels added."
            }
        )

    def _run_cross_section_pass(self, report_id: str, outline: ReportOutline):
        """
        Runs an analytical sanity and contradiction check across all sections.
        If a cross-section contradiction or numerical anomaly is found, it uses the composer
        to revise only the offending sections/paragraphs.
        
        This uses a bounded revalidation loop (max 2 passes) to detect, revise, persist, and re-detect.
        """
        logger.info("[Cross-Section] Starting cross-section consistency checks...")
        
        MAX_PASSES = 2
        for current_pass in range(1, MAX_PASSES + 1):
            logger.info(f"[Cross-Section] Running Pass {current_pass}/{MAX_PASSES}...")
            
            section_map = {}
            original_contents = {}
            for i, section in enumerate(outline.sections):
                section_map[i] = section.content
                original_contents[i] = section.content
                
            combined_text = "\n\n".join(section_map.values())
            
            # Map normalized sentences back to section indices (1-based)
            def normalize_text(text: str) -> str:
                return re.sub(r'\s+', ' ', text).strip()
                
            sentence_to_section = {}
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                if not section.content:
                    continue
                # Split the section content using the exact same logic
                sec_clean = re.sub(r'<self_critique>.*?</self_critique>', '', section.content, flags=re.DOTALL)
                sec_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sec_clean) if len(s.strip()) > 15]
                for s in sec_sentences:
                    norm_s = normalize_text(s)
                    if norm_s not in sentence_to_section:
                        sentence_to_section[norm_s] = set()
                    sentence_to_section[norm_s].add(section_num)
                    
            def find_involved_sections(sent: str) -> List[int]:
                norm_sent = normalize_text(sent)
                sections_involved = set()
                # 1. Exact match from our dictionary
                if norm_sent in sentence_to_section:
                    sections_involved.update(sentence_to_section[norm_sent])
                # 2. Substring match fallback
                for idx, sec in enumerate(outline.sections):
                    if sec.content and norm_sent in normalize_text(sec.content):
                        sections_involved.add(idx + 1)
                return sorted(list(sections_involved))
                
            def find_sections_for_general_issue(msg: str) -> List[int]:
                matched_sections = []
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', msg)
                if numbers:
                    for idx, sec in enumerate(outline.sections):
                        if sec.content and any(num in sec.content for num in numbers):
                            matched_sections.append(idx + 1)
                if not matched_sections:
                    words = re.findall(r'\b[a-zA-Z]{4,}\b', msg.lower())
                    common = {"potential", "mismatch", "suspicious", "detected", "discussion", "analysis", "mentions", "lacks", "concrete", "figures"}
                    keywords = [w for w in words if w not in common]
                    for idx, sec in enumerate(outline.sections):
                        if sec.content:
                            sec_lower = sec.content.lower()
                            if any(kw in sec_lower for kw in keywords):
                                matched_sections.append(idx + 1)
                return sorted(list(set(matched_sections)))

            # Run detection
            cross_contradictions = self._contradiction_detector.check(combined_text)
            
            # Module selection for numerical sanity check
            module = self._select_analytical_module("")
            cross_numerical = self._numerical_sanity_checker.check(combined_text, [], module)
            
            # Build structured issues list
            structured_issues = []
            
            # Add contradictions from detailed_contradictions
            if hasattr(cross_contradictions, "detailed_contradictions") and cross_contradictions.detailed_contradictions:
                for c in cross_contradictions.detailed_contradictions:
                    sent_1 = c["sent_1"]
                    sent_2 = c["sent_2"]
                    sections_1 = find_involved_sections(sent_1)
                    sections_2 = find_involved_sections(sent_2)
                    all_involved = sorted(list(set(sections_1 + sections_2)))
                    # Fallback to general issue matching if find_involved_sections is empty
                    if not all_involved:
                        all_involved = find_sections_for_general_issue(c["reason"])
                    structured_issues.append({
                        "type": "contradiction",
                        "message": c["reason"],
                        "affected_sections": all_involved,
                        "evidence_snippets": [sent_1, sent_2]
                    })
            else:
                # Fallback to raw contradictions parsing
                for ent, s1, s2, reason in cross_contradictions.contradictions:
                    all_involved = find_sections_for_general_issue(reason)
                    structured_issues.append({
                        "type": "contradiction",
                        "message": reason,
                        "affected_sections": all_involved,
                        "evidence_snippets": []
                    })
                    
            # Add numerical anomalies from detailed_anomalies
            if hasattr(cross_numerical, "detailed_anomalies") and cross_numerical.detailed_anomalies:
                for anom in cross_numerical.detailed_anomalies:
                    sent = anom.get("sentence", "")
                    all_involved = []
                    if sent:
                        all_involved = find_involved_sections(sent)
                    if not all_involved:
                        all_involved = find_sections_for_general_issue(anom["message"])
                    structured_issues.append({
                        "type": "numerical_anomaly",
                        "message": anom["message"],
                        "affected_sections": all_involved,
                        "evidence_snippets": [sent] if sent else []
                    })
            else:
                # Fallback to raw anomalies list
                for anom in cross_numerical.anomalies:
                    all_involved = find_sections_for_general_issue(anom)
                    structured_issues.append({
                        "type": "numerical_anomaly",
                        "message": anom,
                        "affected_sections": all_involved,
                        "evidence_snippets": []
                    })
                    
            if not structured_issues:
                logger.info(f"[Cross-Section] Pass {current_pass}: No cross-section contradictions or numerical anomalies detected.")
                break
                
            logger.warning(f"[Cross-Section] Pass {current_pass}: Found {len(structured_issues)} cross-section consistency issues.")
            for issue in structured_issues:
                logger.warning(f"  - [{issue['type']}] (Sections: {issue['affected_sections']}): {issue['message']}")
                
            # Perform targeted section revisions
            sections_revised_this_pass = set()
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                involved_issues = [issue for issue in structured_issues if section_num in issue["affected_sections"]]
                
                if involved_issues:
                    logger.info(f"[Cross-Section] Revising section {section_num:02d} '{section.title}' to resolve cross-section issues.")
                    sections_revised_this_pass.add(section_num)
                    
                    composer_system = (
                        "You are a senior analyst performing a final consistency review across a multi-section prediction report.\n"
                        "Your task is to revise and polish the section to resolve consistency errors with other sections.\n"
                        "Write polished markdown only. Do not use Markdown headings (#, ##, ###) in your response; use **bold text** instead.\n"
                        "Keep the section highly focused and concise (target 800-1200 words). Do not repeat observations endlessly.\n"
                        f"{get_language_instruction()}"
                    )
                    
                    # Group issues block with the conflicting snippets
                    issues_block_parts = []
                    for idx, issue in enumerate(involved_issues):
                        issues_block_parts.append(f"Issue {idx + 1}: {issue['message']}")
                        if issue["evidence_snippets"]:
                            issues_block_parts.append("Conflicting / relevant evidence snippets in the report:")
                            for snip in issue["evidence_snippets"]:
                                # Tag snippet with sections it belongs to
                                snip_secs = find_involved_sections(snip)
                                snip_secs_str = ", ".join(f"Section {s}" for s in snip_secs)
                                issues_block_parts.append(f"  - [{snip_secs_str}]: \"{snip}\"")
                        issues_block_parts.append("")
                        
                    issues_block = "\n".join(issues_block_parts)
                    
                    composer_user = (
                        f"You are performing a final cross-section review of the section: {section.title}\n\n"
                        f"[Prediction Requirement / Scenario]\n{self.simulation_requirement}\n\n"
                        f"[Current Section Content]\n{section.content}\n\n"
                        f"[Cross-Section Issues to Resolve]\n"
                        f"Our automated validator detected the following contradictions/anomalies involving this section:\n\n"
                        f"{issues_block}\n"
                        f"[Instructions]\n"
                        f"1. Revise the content to completely resolve the contradictions or numerical mismatches with other sections.\n"
                        f"2. Ensure you keep the tone analytical, expert, and grounded. Do not introduce speculative claims unless calibrated.\n"
                        f"3. Do NOT change parts of the text that are not involved in these issues.\n"
                        f"4. Output ONLY the polished, revised section content. Do not include intro, outro, conversational fillers, or headings."
                    )
                    
                    try:
                        revised_content = self.composer_llm.chat(
                            messages=[
                                {"role": "system", "content": composer_system},
                                {"role": "user", "content": composer_user},
                            ],
                            temperature=0.3,
                            max_tokens=Config.LLM_CHAT_MAX_TOKENS,
                        )
                        if revised_content and revised_content.strip():
                            cleaned = revised_content.strip()
                            cleaned = re.sub(r'^#+\s+.*?\n', '', cleaned)
                            section.content = cleaned.strip()
                            logger.info(f"[Cross-Section] Successfully resolved cross-section consistency issues for '{section.title}'")
                    except Exception as e:
                        logger.error(f"[Cross-Section] Revision of '{section.title}' failed: {e}")

            # Persist modified sections back to disk
            modified_count = 0
            modified_indices = []
            for i, section in enumerate(outline.sections):
                if section.content != original_contents[i]:
                    section_num = i + 1
                    try:
                        ReportManager.save_section(report_id, section_num, section)
                        modified_count += 1
                        modified_indices.append(section_num)
                        logger.info(f"[Cross-Section] Saved revised section {section_num:02d} ({section.title}) to disk.")
                    except Exception as e:
                        logger.error(f"[Cross-Section] Failed to save revised section {section_num:02d} ({section.title}) to disk: {e}")

            if self.report_logger:
                self.report_logger.log(
                    action="cross_section_revision",
                    stage="generating",
                    details={
                        "message": f"Pass {current_pass} complete. Revised and saved {modified_count} sections: {modified_indices}.",
                        "modified_count": modified_count,
                        "modified_indices": modified_indices,
                        "pass": current_pass
                    }
                )
                
            # If no sections were modified in this pass, the loop has converged and we can stop.
            if modified_count == 0:
                logger.info(f"[Cross-Section] Loop converged after pass {current_pass} (no sections modified).")
                break
        else:
            # We finished all passes and some issues might still remain. Log a warning.
            logger.warning(f"[Cross-Section] Reached maximum passes ({MAX_PASSES}) without complete consistency resolution.")
            if self.report_logger:
                self.report_logger.log(
                    action="cross_section_warning",
                    stage="generating",
                    details={
                        "message": f"Warning: Bounded revalidation loop finished after {MAX_PASSES} passes with unresolved issues remaining.",
                        "max_passes": MAX_PASSES
                    }
                )
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        logger.info(t('report.startPlanningOutline'))
        
        if progress_callback:
            progress_callback("planning", 0, t('progress.analyzingRequirements'))
        
        
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, t('progress.generatingOutline'))
        
        system_prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{get_language_instruction()}"
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )
        if self._web_search_corpus:
            user_prompt += f"\n\n[Recent external facts (auto-fetched, dated)]\n{self._web_search_corpus}"

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, t('progress.parsingOutline'))
            
            
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, t('progress.outlinePlanComplete'))
            
            logger.info(t('report.outlinePlanDone', count=len(sections)))
            return outline
            
        except Exception as e:
            logger.error(t('report.outlinePlanFailed', error=str(e)))
            return ReportOutline(
                title="Future Prediction Report",
                summary="Future trends and risk analysis based on simulation predictions",
                sections=[
                    ReportSection(title="Prediction Scenarios and Key Findings"),
                    ReportSection(title="Population Behavior Prediction Analysis"),
                    ReportSection(title="Trend Outlook and Risk Indicators")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """Generate a single section using a ReACT loop."""
        logger.info(t('report.reactGenerateSection', title=section.title))
        
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )
        if self._web_search_corpus:
            user_prompt += f"\n\n[Recent external facts (auto-fetched, dated)]\n{self._web_search_corpus}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 12  # Maximum number of successful iterations (tool calls + final answer + retries)
        min_tool_calls = 3  # Minimum number of tool calls
        conflict_retries = 0  # Continuous conflict retries when tool calls and Final Answer appear together
        used_tools = set()  # Record names of used tools
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents", "web_search"}
        observation_log = []
        sources_metadata: List[Tuple[int, str, str]] = []
        evidence_scores = []
        evidence_scores_dict = {}

        # Report context, used for sub-question generation in InsightForge
        report_context = f"Section Title: {section.title}\nSimulation Requirement: {self.simulation_requirement}"
        
        iteration = 0
        while iteration < max_iterations:
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    t('progress.deepSearchAndWrite', current=tool_calls_count, max=self.MAX_TOOL_CALLS_PER_SECTION)
                )
            
            # Call LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=Config.LLM_CHAT_MAX_TOKENS
            )

            # Check if LLM response is None (API exception or empty content)
            if response is None:
                logger.warning(t('report.sectionIterNone', title=section.title, iteration=iteration + 1))
                # If there are iterations left, append message and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({"role": "user", "content": "Please continue generating content."})
                    iteration += 1
                    continue
                # Last iteration also returned None, break loop and enter forced completion
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once, reuse result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict resolution: LLM outputted both tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t('report.sectionConflict', title=section.title, iteration=iteration+1, conflictCount=conflict_retries)
                )

                if conflict_retries <= 2:
                    # First two times: discard this response and ask LLM to reply again
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format Error] You included both a tool call and a Final Answer in a single response, which is not allowed.\n"
                            "Each response must only do one of the following:\n"
                            "- Call a tool (output a <tool_call> block, do not write a Final Answer)\n"
                            "- Output final content (start with 'Final Answer:', do not include a <tool_call>)\n"
                            "Please reply again and perform only one of these actions."
                        ),
                    })
                    # Do not increment iteration for format conflicts
                    continue
                else:
                    # Third time: downgrade, truncate to the first tool call, and force execution
                    logger.warning(
                        t('report.sectionConflictDowngrade', title=section.title, conflictCount=conflict_retries)
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Record LLM response log
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM outputted Final Answer ──
            if has_final_answer:
                # Insufficient number of tool calls, reject and ask to continue calling tools
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f" (These tools have not been used yet, recommending using them: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    iteration += 1
                    continue

                # Normal completion
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(t('report.sectionGenDone', title=section.title, count=tool_calls_count))

                audit_report = self._run_analytical_audit(
                    final_answer,
                    tool_calls_count,
                    sources_metadata,
                    section.title,
                    evidence_scores=evidence_scores_dict
                )
                logger.info(f"[Grounding] {section.title}: {audit_report.grounding.summary}")
                if audit_report.grounding.warning:
                    logger.warning(f"[Grounding] {audit_report.grounding.warning}")
                if audit_report.contradictions.warning:
                    logger.warning(f"[Contradictions] {audit_report.contradictions.warning}")
                if audit_report.numerical_sanity.warning:
                    logger.warning(f"[Numerical Sanity] {audit_report.numerical_sanity.warning}")
                if audit_report.specificity.warning:
                    logger.warning(f"[Specificity] {audit_report.specificity.warning}")
                if audit_report.confidence.warning:
                    logger.warning(f"[Confidence] {audit_report.confidence.warning}")

                final_answer = self._finalize_section_with_governance(
                    section,
                    final_answer,
                    observation_log,
                    audit_report,
                    tool_calls_count,
                    sources_metadata,
                    evidence_scores_dict,
                    section_index,
                )

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM tries to call a tool ──
            if has_tool_calls:
                # Tool limit reached -> notify explicitly, ask for Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    limit_msg = REACT_TOOL_LIMIT_MSG.format(
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                    )
                    limit_msg += self._build_evidence_guidance(
                        evidence_scores,
                        section.title,
                        include_mandate=True,
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": limit_msg})
                    iteration += 1
                    continue

                # Execute only the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(t('report.multiToolOnlyFirst', total=len(tool_calls), toolName=call['name']))

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                # Apply temporal relevance filter before feeding the result to the LLM.
                # This prepends a Temporal Context header with date extraction, freshness
                # classification, staleness score, and synthesis rules.
                annotation = self._temporal_filter.build_annotation(
                    raw_result=result,
                    simulation_requirement=self.simulation_requirement,
                    section_title=section.title,
                    tool_name=call["name"],
                )
                if annotation.staleness_warning:
                    logger.info(
                        f"[TemporalFilter] {call['name']}: {annotation.staleness_warning} (score={annotation.freshness_score:.1f}, dates={annotation.dates_found[:3]})"
                    )
                annotated_result = annotation.header  # includes raw_result appended

                # Score evidence and prepend evidence card
                score = self._evidence_evaluator.score_evidence(
                    source_num=tool_calls_count + 1,
                    raw_result=result,
                    tool_name=call["name"],
                    section_title=section.title,
                    simulation_requirement=self.simulation_requirement,
                    freshness_score=annotation.freshness_score
                )
                evidence_scores.append(score)
                evidence_scores_dict[tool_calls_count + 1] = score
                annotated_result = self._evidence_evaluator.build_evidence_card(score, annotated_result)

                observation_log.append(
                    f"Tool: {call['name']}\nParameters: {json.dumps(call.get('parameters', {}), ensure_ascii=False)}\nResult: {result}"
                )
                sources_metadata.append((tool_calls_count + 1, call["name"], result))

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=annotated_result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build unused tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                unused_hint += self._build_evidence_guidance(
                    evidence_scores,
                    section.title,
                    include_mandate=False,
                )

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=annotated_result,
                        source_num=tool_calls_count,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                iteration += 1
                continue

            # ── Case 3: Neither tool call nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Insufficient tool calls, recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = f" (These tools have not been used yet, recommending using them: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                iteration += 1
                continue

            # Tool calls are sufficient, LLM outputted content but without "Final Answer:" prefix
            # Directly use this content as the final answer, do not idle run
            logger.info(t('report.sectionNoPrefix', title=section.title, count=tool_calls_count))
            final_answer = response.strip()

            audit_report = self._run_analytical_audit(
                final_answer,
                tool_calls_count,
                sources_metadata,
                section.title,
                evidence_scores=evidence_scores_dict
            )
            logger.info(f"[Grounding] {section.title}: {audit_report.grounding.summary}")
            if audit_report.grounding.warning:
                logger.warning(f"[Grounding] {audit_report.grounding.warning}")
            if audit_report.contradictions.warning:
                logger.warning(f"[Contradictions] {audit_report.contradictions.warning}")
            if audit_report.numerical_sanity.warning:
                logger.warning(f"[Numerical Sanity] {audit_report.numerical_sanity.warning}")
            if audit_report.specificity.warning:
                logger.warning(f"[Specificity] {audit_report.specificity.warning}")
            if audit_report.confidence.warning:
                logger.warning(f"[Confidence] {audit_report.confidence.warning}")

            final_answer = self._finalize_section_with_governance(
                section,
                final_answer,
                observation_log,
                audit_report,
                tool_calls_count,
                sources_metadata,
                evidence_scores_dict,
                section_index,
            )

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Maximum iterations reached, force content generation
        logger.warning(t('report.sectionMaxIter', title=section.title))
        force_msg = REACT_FORCE_FINAL_MSG + self._build_evidence_guidance(
            evidence_scores,
            section.title,
            include_mandate=True,
        )
        messages.append({"role": "user", "content": force_msg})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=Config.LLM_CHAT_MAX_TOKENS
        )

        # Check if LLM response is None during forced completion
        if response is None:
            logger.error(t('report.sectionForceFailed', title=section.title))
            final_answer = t('report.sectionGenFailedContent')
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        audit_report = self._run_analytical_audit(
            final_answer,
            tool_calls_count,
            sources_metadata,
            section.title,
            evidence_scores=evidence_scores_dict
        )
        logger.info(f"[Grounding] {section.title}: {audit_report.grounding.summary}")
        if audit_report.grounding.warning:
            logger.warning(f"[Grounding] {audit_report.grounding.warning}")
        if audit_report.contradictions.warning:
            logger.warning(f"[Contradictions] {audit_report.contradictions.warning}")
        if audit_report.numerical_sanity.warning:
            logger.warning(f"[Numerical Sanity] {audit_report.numerical_sanity.warning}")
        if audit_report.specificity.warning:
            logger.warning(f"[Specificity] {audit_report.specificity.warning}")
        if audit_report.confidence.warning:
            logger.warning(f"[Confidence] {audit_report.confidence.warning}")

        final_answer = self._finalize_section_with_governance(
            section,
            final_answer,
            observation_log,
            audit_report,
            tool_calls_count,
            sources_metadata,
            evidence_scores_dict,
            section_index,
        )

        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )

        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate complete report (real-time output by section).
        
        Each section is saved to the folder immediately upon completion, 
        no need to wait for the entire report.
        File structure:
        reports/{report_id}/
            meta.json       - Report metadata
            outline.json    - Report outline
            progress.json   - Generation progress
            section_01.md   - Section 1
            section_02.md   - Section 2
            ...
            full_report.md  - Complete report
        
        Args:
            progress_callback: Progress callback function (stage, progress, message)
            report_id: Report ID (optional, auto-generated if not provided)
            
        Returns:
            Report: Complete report
        """
        import uuid
        
        # If no report_id is provided, auto-generate one
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # List of completed section titles (used for progress tracking)
        completed_section_titles = []
        
        try:
            # Initialization: Create report folder and save initial state
            ReportManager._ensure_report_folder(report_id)
            
            # Initialize logger (structured log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize console logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, t('progress.initReport'),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # One-shot Keiro pre-fetch step before planning
            from .web_search import web_search_configured, search_queries_to_corpus
            if Config.REPORT_PREFETCH_WEB_SEARCH and web_search_configured():
                logger.info("Starting report-level web search pre-fetch...")
                if progress_callback:
                    progress_callback("pending", 0, "Pre-fetching web search corpus...")
                
                try:
                    from .research_query_generator import ResearchQueryGenerator
                    rqg = ResearchQueryGenerator()
                    queries = rqg.generate_queries(self.simulation_requirement)
                    if queries:
                        queries = queries[:Config.REPORT_PREFETCH_MAX_QUERIES]
                        logger.info(f"Generated {len(queries)} pre-fetch queries: {queries}")
                        
                        search_corpus, _ = search_queries_to_corpus(
                            queries,
                            simulation_requirement=self.simulation_requirement,
                            max_chars=12000
                        )
                        if search_corpus and search_corpus.strip():
                            self._web_search_corpus = search_corpus
                            logger.info(f"Pre-fetch completed successfully! Corpus size: {len(search_corpus)} chars.")
                            self.report_logger.log("prefetch_complete", "pending", {
                                "message": f"Pre-fetched web search corpus of {len(search_corpus)} characters.",
                                "queries": queries
                            })
                except Exception as ex:
                    logger.warning(f"Report-level pre-fetch failed, proceeding without it: {ex}")
            
            # Check if there is an existing outline to resume from
            existing_outline = ReportManager.get_outline(report_id)
            sections_on_disk = ReportManager.get_generated_sections(report_id)
            completed_contents_by_num = {s["section_index"]: s["content"] for s in sections_on_disk}
            
            if existing_outline:
                logger.info(f"Resuming report generation for report_id={report_id} with existing outline.")
                outline = existing_outline
                report.outline = outline
                
                # Fill completed section contents
                for i, section in enumerate(outline.sections):
                    section_num = i + 1
                    if section_num in completed_contents_by_num:
                        section.content = completed_contents_by_num[section_num]
                        if section.title not in completed_section_titles:
                            completed_section_titles.append(section.title)
                
                # Record resume status log
                self.report_logger.log("resume_report", "generating", {
                    "message": f"Resumed report generation. Loaded {len(completed_contents_by_num)} completed sections.",
                    "report_id": report_id,
                    "completed_sections": completed_section_titles
                })
                
                ReportManager.update_progress(
                    report_id, "generating", 20, f"Resuming report generation. Loaded {len(completed_contents_by_num)} completed sections.",
                    completed_sections=completed_section_titles
                )
                ReportManager.save_report(report)
            else:
                # Phase 1: Plan outline
                report.status = ReportStatus.PLANNING
                ReportManager.update_progress(
                    report_id, "planning", 5, t('progress.startPlanningOutline'),
                    completed_sections=[]
                )
                
                # Record planning start log
                self.report_logger.log_planning_start()
                
                if progress_callback:
                    progress_callback("planning", 0, t('progress.startPlanningOutline'))
                
                outline = self.plan_outline(
                    progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
                )
                report.outline = outline
                
                # Record planning completed log
                self.report_logger.log_planning_complete(outline.to_dict())
                
                # Save outline to file
                ReportManager.save_outline(report_id, outline)
                ReportManager.update_progress(
                    report_id, "planning", 15, t('progress.outlineDone', count=len(outline.sections)),
                    completed_sections=[]
                )
                ReportManager.save_report(report)
                logger.info(t('report.outlineSavedToFile', reportId=report_id))
            
            # Phase 2: Generate section by section (saved by section)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Save content to be used as context
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # If resuming and the section is already on disk, skip generation and use as context for subsequent sections
                if existing_outline and section_num in completed_contents_by_num:
                    logger.info(f"Section {section_num:02d} ({section.title}) is already completed. Skipping generation.")
                    section_content = completed_contents_by_num[section_num]
                    section.content = section_content
                    generated_sections.append(f"## {section.title}\n\n{section_content}")
                    continue
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    t('progress.generatingSection', title=section.title, current=section_num, total=total_sections),
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        t('progress.generatingSection', title=section.title, current=section_num, total=total_sections)
                    )
                
                # Generate main section content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Record section completion log
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(t('report.sectionSaved', reportId=report_id, sectionNum=f"{section_num:02d}"))
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    t('progress.sectionDone', title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3: Assemble complete report
            if progress_callback:
                progress_callback("generating", 95, t('progress.assemblingReport'))
            
            ReportManager.update_progress(
                report_id, "generating", 95, t('progress.assemblingReport'),
                completed_sections=completed_section_titles
            )
            
            # Run cross-section pass before final assembly
            self._run_cross_section_pass(report_id, outline)
            
            # Assemble complete report using ReportManager
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calculate total elapsed time
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Record report completion log
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, t('progress.reportComplete'),
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, t('progress.reportComplete'))
            
            logger.info(t('report.reportGenDone', reportId=report_id))
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(t('report.reportGenFailed', error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Record error log
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Save failure status
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, t('progress.reportFailed', error=str(e)),
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignore error of saving failure
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with Report Agent.
        
        In the chat, the Agent can autonomously call retrieval tools to answer questions.
        
        Args:
            message: User message
            chat_history: Chat history
            
        Returns:
            {
                "response": "Agent reply",
                "tool_calls": [List of called tools],
                "sources": [Information sources]
            }
        """
        logger.info(t('report.agentChat', message=message[:50]))
        
        chat_history = chat_history or []
        
        # Get generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit report length to avoid overly long context
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(t('report.fetchReportFailed', error=e))
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report yet)",
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add historical chat
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT loop (Simplified version)
        tool_calls_made = []
        max_iterations = 2  # Reduce iteration rounds
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return response directly
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute tool calls (limit count)
            tool_results = []
            for call in tool_calls[:1]:  # Execute at most 1 tool call per round
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limit result length
                })
                tool_calls_made.append(call)
            
            # Add results to messages
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} Result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Maximum iterations reached, get final response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean response
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report Manager.
    
    Responsible for persistent storage and retrieval of reports.
    
    File structure (output by section):
    reports/
      {report_id}/
        meta.json          - Report metadata and status
        outline.json       - Report outline
        progress.json      - Generation progress
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Complete report
    """
    
    # Directory to store reports
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure report root directory exists."""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get report folder path."""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure report folder exists and return path."""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get the path of the report metadata file."""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get the path of the complete report Markdown file."""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get the path of the outline file."""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get the path of the progress file."""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get the path of the section Markdown file."""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get the path of the Agent log file."""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get the path of the console log file."""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get console log content.
        
        These are console output logs (INFO, WARNING, etc.) during report generation,
        different from the structured logs of agent_log.jsonl.
        
        Args:
            report_id: Report ID
            from_line: The line number to start reading from (used for incremental fetching, 0 means from the beginning)
            
        Returns:
            {
                "logs": [List of log lines],
                "total_lines": Total line count,
                "from_line": Starting line number,
                "has_more": Whether there are more logs
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep original log line, strip trailing newline
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Already read to the end
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Get the full console log (fetch all at once).
        
        Args:
            report_id: Report ID
            
        Returns:
            List of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent log content.
        
        Args:
            report_id: Report ID
            from_line: The line number to start reading from (used for incremental fetching, 0 means from the beginning)
            
        Returns:
            {
                "logs": [List of log entries],
                "total_lines": Total line count,
                "from_line": Starting line number,
                "has_more": Whether there are more logs
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that fail to parse
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Already read to the end
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete Agent log (used to fetch all at once).
        
        Args:
            report_id: Report ID
            
        Returns:
            List of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save report outline.
        
        Called immediately after the planning phase is completed.
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(t('report.outlineSaved', reportId=report_id))
    
    @classmethod
    def get_outline(cls, report_id: str) -> Optional[ReportOutline]:
        """Get the report outline."""
        path = cls._get_outline_path(report_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sections = []
            for s in data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            return ReportOutline(
                title=data.get('title', 'Simulation Analysis Report'),
                summary=data.get('summary', ''),
                sections=sections
            )
        except Exception as e:
            logger.warning(f"Failed to load outline for {report_id}: {e}")
            return None
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save a single section.

        Called immediately after each section is generated, achieving section-by-section output.

        Args:
            report_id: Report ID
            section_index: Section index (1-based)
            section: Section object

        Returns:
            Saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content - clean potential duplicate headings
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(t('report.sectionFileSaved', reportId=report_id, fileSuffix=file_suffix))
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean section content.
        
        1. Remove Markdown heading lines at the start that duplicate the section title.
        2. Convert all ### and lower level headings to bold text.
        
        Args:
            content: Original content
            section_title: Section title
            
        Returns:
            Cleaned content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if it is a Markdown heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Check if it is a heading that duplicates the section title (skip duplicates within the first 5 lines)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convert all heading levels (#, ##, ###, ####, etc.) to bold
                # Because the section heading is added by the system, there should not be any headings in the content
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add empty line
                continue
            
            # If the previous line was a skipped heading, and the current line is empty, skip it too
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remove leading empty lines
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remove leading separators
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Also remove empty lines after the separator
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Update report generation progress.
        
        The frontend can read progress.json to get real-time progress.
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation progress."""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the list of generated sections.
        
        Return information of all saved section files.
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse section index from filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble the complete report.
        
        Assemble the complete report from saved section files, and clean headings.
        """
        folder = cls._get_report_folder(report_id)
        
        # Build report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-processing: Clean heading issues in the entire report
        md_content = cls._post_process_report(md_content, outline)
        
        # Save complete report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(t('report.fullReportAssembled', reportId=report_id))
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process report content.
        
        1. Remove duplicate headings.
        2. Keep main report heading (#) and section headings (##), remove other heading levels (###, ####, etc.).
        3. Clean extra empty lines and separators.
        
        Args:
            content: Original report content
            outline: Report outline
            
        Returns:
            Processed content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collect all section titles from the outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if it is a heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check if it is a duplicate heading (identical content within 5 consecutive lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip duplicate heading and subsequent empty lines
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Heading level processing:
                # - # (level=1) Only keep main report title
                # - ## (level=2) Keep section titles
                # - ### and below (level>=3) Convert to bold text
                
                if level == 1:
                    if title == outline.title:
                        # Keep main report title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title incorrectly used #, fix to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other level 1 headings converted to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section titles
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Non-section level 2 headings converted to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### and below headings converted to bold text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skip separator immediately following heading
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Only keep one empty line after heading
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Clean consecutive multiple empty lines (keep at most 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and complete report."""
        cls._ensure_report_folder(report.report_id)
        
        # Save metadata JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save complete Markdown report
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(t('report.reportSaved', reportId=report.report_id))
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Get report."""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Backwards compatibility: Check files stored directly in the reports directory
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Rebuild Report object
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # If markdown_content is empty, try to read from full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get report by simulation ID."""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Backwards compatibility: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List reports."""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Backwards compatibility: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Sorted by creation time descending
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report (entire folder)."""
        import shutil
        
        # Clean up logging file handlers before deleting directory to avoid FileNotFoundError on write
        try:
            ReportConsoleLogger.cleanup_handlers(report_id)
        except Exception as e:
            logger.warning(f"Error cleaning up console log handlers for {report_id} during deletion: {e}")
            
        folder_path = cls._get_report_folder(report_id)
        
        # New format: delete the entire folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t('report.reportFolderDeleted', reportId=report_id))
            return True
        
        # Backwards compatibility: delete individual files
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
