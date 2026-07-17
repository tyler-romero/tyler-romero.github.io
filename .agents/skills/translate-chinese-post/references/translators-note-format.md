# Translator's Note Format

The translator's note appears immediately after the frontmatter `---` and before the body content. It is always in italics and follows a fixed structure.

## Scientific Spaces

```markdown
*Translator's note (Opus 4.6): This is an English translation of [中文标题](https://kexue.fm/archives/XXXXX) by Jianlin Su (苏剑林), originally published on Month Day, Year on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*
```

## Zhihu

```markdown
*Translator's note (Opus 4.6): This is an English translation of [中文标题](https://zhuanlan.zhihu.com/p/XXXXX) by [Author Display Name](https://www.zhihu.com/people/author-slug), originally published on Month Day, Year on [Zhihu (知乎)](https://www.zhihu.com). The translation preserves the author's informal, first-person voice.*
```

## Rules

- The note is always wrapped in `*...*` (italics).
- The model identifier `(Opus 4.6)` may be updated to match the model used for translation.
- The original title is linked using the Chinese title text, not the English translation.
- The date is written in natural English format: `June 20, 2025`.
- For Zhihu posts with an informal tone, use "informal, first-person voice". For Scientific Spaces (which tends to be more academic), use just "first-person voice".
- A `<hr class="section-divider">` immediately follows the translator's note, separated by a blank line.

## Example

```markdown
*Translator's note (Opus 4.6): This is an English translation of [线性注意力简史：从模仿、创新到反哺](https://kexue.fm/archives/11033) by Jianlin Su (苏剑林), originally published on June 20, 2025 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">
```
