---
name: translate-chinese-post
description: "Translate a Chinese-language AI/ML blog post into English for the translations section of this site. Use when the user provides a Chinese post (URL, pasted text, or file) and wants to generate a translated markdown file for Scientific Spaces or Zhihu. Handles frontmatter, translator's notes, LaTeX math, citations, and licensing footers."
argument-hint: 'Paste Chinese post content or provide the source URL'
---

# Translate Chinese AI/ML Post to English

Translate a Chinese-language AI/ML blog post into a publish-ready English markdown file for the `src/translations/` section of this Eleventy site.

## When to Use

- User provides a Chinese blog post (pasted text, URL, or file) and asks for an English translation
- User says "translate this post", "add a new translation", or similar
- User wants to create a new Scientific Spaces or Zhihu translation page

## Procedure

### 1. Identify the Source

Determine the source platform and gather metadata:

- **Scientific Spaces (科学空间)** — posts from `kexue.fm` by Jianlin Su (苏剑林)
  - Output directory: `src/translations/scientific-spaces/`
  - Layout is auto-applied via `scientific-spaces.json` (do NOT add `layout:` to frontmatter)
- **Zhihu (知乎)** — posts from `zhihu.com` or `zhuanlan.zhihu.com`
  - Output directory: `src/translations/zhihu/`
  - Layout is auto-applied via `zhihu.json` (do NOT add `layout:` to frontmatter)

Collect:
- Original Chinese title
- Author name (Chinese + pinyin)
- Original publication date
- Original URL
- Whether the post uses math (LaTeX)

### 2. Generate the Filename

Create a kebab-case English slug from the translated title. Examples:
- `a-brief-history-of-linear-attention.md`
- `why-linear-attention-needs-short-conv.md`
- `moe-post-training-challenges-and-lessons.md`

### 3. Write the Frontmatter

Use the exact format from the [frontmatter reference](./references/frontmatter-format.md). Key rules:
- `title:` — English translation of the title, in double quotes
- `subtitle:` — always follows the pattern: `"Translated from [中文标题](original-url) by Author (中文名)"`
- `date:` — original publication date in ISO 8601 with timezone offset (e.g., `2025-06-20T00:00:00+08:00`)
- `blurb:` — a one-to-two sentence English summary of the post's content, in double quotes
- `tags:` — always include `"translation"` as the first tag, then topical tags in kebab-case
- `math: true` — include if the post contains any LaTeX/math

### 4. Write the Translator's Note

Immediately after the frontmatter, add a translator's note in italics. Follow the [translator's note format](./references/translators-note-format.md) exactly.

### 5. Translate the Body

Follow the [translation rules](./references/translation-rules.md) for the body content. Key principles:
- Preserve the author's first-person voice and tone
- Keep all LaTeX math expressions exactly as-is (only translate surrounding prose)
- Convert Chinese references to the original paper titles (use English paper titles where known)
- Use `<hr class="section-divider">` after the translator's note and before the citation footer
- Preserve markdown heading levels (`##`, `###`) from the original
- Keep image references and alt text, translating alt text to English
- Translate footnote content

### 6. Write the Citation Footer

End the file with the [citation and license footer](./references/citation-footer-format.md).

### 7. Validate

- Confirm the file is saved in the correct subdirectory
- Verify all LaTeX delimiters are intact (`\[...\]` for display, `\(...\)` for inline)
- Check that no `layout:` field appears in frontmatter (layouts are inherited from directory JSON files)
- Ensure the `"translation"` tag is present (this controls collection membership)
- Verify the translator's note, section dividers, and citation footer are all present
