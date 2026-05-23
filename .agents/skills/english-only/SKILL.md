---
name: english-only
description: Enforce English-only in touched files; translate Chinese and other non-English text to natural English. Use when the user wants an English-only codebase, a translation sweep, or to remove non-English from source, comments, logs, or UI.
---

# English-only

Enforce English as the sole language for all project material you create or edit in this session. Convert any non-English text you find to natural English.

## Scope

Apply to every file you touch:

- Source: comments, docstrings, log messages, debug/console output
- UI: inline strings in Vue templates; Python strings outside i18n
- Docs and Markdown in the repo
- LLM/system prompts and agent instructions
- Descriptive identifiers that are prose (not domain-specific terms)

## Convert to English

When you find non-English text (e.g. Chinese 简体/繁体, Japanese, Korean, or other languages):

1. Replace it with clear, natural English (UI: short; comments: concise).
2. Do not leave mixed-language files unless an exception applies.
3. For user-facing product strings, add or update keys in `locales/en.json` and wire through i18n—do not leave inline non-English in components.

## Do not change

- Non-English entries in `locales/*.json` other than `en.json` unless the user asked to translate locale files
- User data, cited quotations, legal or proper names
- Third-party or verbatim excerpts that must stay in another language

## Output

When done, briefly list files changed and what was translated. If nothing non-English was found in scope, say so.
