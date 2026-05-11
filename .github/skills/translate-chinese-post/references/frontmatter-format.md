# Frontmatter Format

Every translation file uses this exact frontmatter structure. Do NOT include a `layout:` field — layouts are inherited from the directory-level JSON data files (`scientific-spaces.json` or `zhihu.json`).

## Scientific Spaces Template

```yaml
---
title: "English Title of the Post"
subtitle: "Translated from [中文标题](https://kexue.fm/archives/XXXXX) by Jianlin Su (苏剑林)"
date: YYYY-MM-DDT00:00:00+08:00
blurb: "One to two sentence summary of what the post covers."
tags: ["translation", "topic-1", "topic-2"]
math: true
---
```

## Zhihu Template

```yaml
---
title: "English Title of the Post"
subtitle: "Translated from [中文标题](https://zhuanlan.zhihu.com/p/XXXXX) by [Author Display Name](https://www.zhihu.com/people/author-slug)"
date: YYYY-MM-DDThh:mm:ss+08:00
blurb: "One to two sentence summary of what the post covers."
tags: ["translation", "topic-1", "topic-2"]
math: true
---
```

## Rules

- **title**: Translated English title in double quotes. Should read naturally in English — not a word-for-word transliteration.
- **subtitle**: Always starts with `"Translated from"` and links to both the original post and the author. For Scientific Spaces, the author is always `Jianlin Su (苏剑林)`. For Zhihu, link to the author's Zhihu profile.
- **date**: Use the original publication date in ISO 8601 format with China Standard Time offset (`+08:00`).
- **blurb**: A concise English summary (1–2 sentences) describing the post's key contribution or argument. Written in sentence case, ending with a period. This appears in the post listing.
- **tags**: Always include `"translation"` as the first tag. Additional tags should be kebab-case and describe the technical topics (e.g., `"linear-attention"`, `"mixture-of-experts"`, `"reinforcement-learning"`).
- **math**: Set to `true` if the post contains any LaTeX math. Omit entirely if no math is present.
