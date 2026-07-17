# Translation Rules

## Voice and Tone

- **Preserve the author's voice.** Translate in first person when the original is first person. Do not shift to third person or academic passive.
- **Match the register.** Scientific Spaces posts are thoughtful and measured — translate accordingly. Zhihu posts can be more casual and conversational — preserve that informality.
- **Don't add your own opinions or commentary.** The body is purely the author's words.

## LaTeX and Math

**This site renders math with KaTeX (not MathJax).** The original kexue.fm uses MathJax, which is more permissive. You MUST adapt the source LaTeX to be KaTeX-compatible:

- **Keep all LaTeX content as-is**, translating only surrounding prose — but apply the KaTeX adaptations below.
- **Display math** uses `\[...\]` (never `$$...$$`).
- **Inline math** uses `\(...\)` (never `$...$`).
- **Equation tags** like `\tag{1}` must be preserved with their original numbers.

### Required KaTeX adaptations (apply when translating from kexue.fm or similar MathJax sources)

1. **Never use `\begin{align}`.** It produces duplicate equation numbers in KaTeX (auto-numbering plus any explicit `\tag{}`). Always convert to `\begin{aligned}` inside `\[...\]`:
   ```
   \[
   \begin{aligned}
   & \text{line 1} \\[5pt]
   & \text{line 2}
   \end{aligned}
   \]
   ```
   Note: `aligned` does not support per-row `\tag{}`. If the source tags individual rows, split each tagged line into its own `\[ ... \tag{N} \]` block.

2. **Strip `\color{namedcolor}{...}` decorations.** KaTeX renders `\color{skyblue}` literally as colored text. Remove the wrapper, keeping the inner content. Example: `\color{skyblue}{\lfloor}x\color{skyblue}{\rfloor}_{\color{skyblue}{\Vert\cdot\Vert\leq\tau}}` becomes `\lfloor x \rfloor_{\Vert\cdot\Vert\leq\tau}`.

3. **Replace `\newcommand` definitions with explicit operators.** Inline `\newcommand` is unreliable in KaTeX. Substitute throughout:
   - `\newcommand{\argmin}{\mathop{\text{argmin}}}` + uses of `\argmin` → `\operatorname*{argmin}` (the `*` keeps subscripts below in display mode)
   - `\newcommand{\tr}{...}` + `\tr` → `\operatorname{tr}`
   - `\newcommand{\msign}{...}` + `\msign` → `\operatorname{msign}` (or `\text{msign}` to match repo convention)
   - `\newcommand{\mclip}{...}` + `\mclip` → `\operatorname{mclip}` (or `\text{mclip}`)
   - Remove all `\newcommand` declarations entirely after substitution.

4. **Equation references.** The source may use `\label{eq:foo}` and `\eqref{eq:foo}` (MathJax-specific). KaTeX doesn't support these. Convert to explicit `\tag{N}` on the equation and write `(N)` inline in prose where the reference occurs.

## Links and References

- **Paper titles**: When the original links to a paper, use the paper's official English title. If the Chinese text uses a Chinese translation of the paper title, replace it with the English original. Keep the URL pointing to the same target.
- **Blog cross-references**: When the original links to another post on the same blog (e.g., another kexue.fm post), keep the original URL. Translate the linked text to English.
- **Author names**: Provide both the English/pinyin name and the Chinese name in parentheses on first mention, e.g., `Jianlin Su (苏剑林)`. Subsequent mentions can use just the English name.
- **Chinese terms**: When a Chinese term is key to understanding (e.g., an idiom or culturally specific phrase), include it in parentheses after the English translation: `"feeding back" (反哺)`.

## Structure and Formatting

- **Headings**: Preserve the heading hierarchy from the original (`##`, `###`, etc.). Translate heading text to English.
- **Tables**: Preserve table structure. Translate header text and cell text, but keep LaTeX in cells unchanged.
- **Images**: Keep `![alt text](path)` format. Translate alt text to English. Keep the image path unchanged.
- **Footnotes**: Translate footnote content. Keep `[^label]` markers unchanged.
- **Bold and italic**: Preserve emphasis markers from the original.
- **Blockquotes**: Preserve blockquote formatting.
- **Lists**: Preserve list formatting (ordered/unordered).

## Section Dividers

- Use `<hr class="section-divider">` for major section breaks.
- One always appears after the translator's note.
- One always appears before the citation footer.
- Do not use `---` for horizontal rules (it conflicts with frontmatter syntax).

## Common Translation Patterns

| Chinese Pattern | English Pattern |
|---|---|
| 风水轮流转 | "what goes around comes around" |
| 剑走偏锋 | "unconventional approach" |
| 除旧迎新 | "out with the old, in with the new" |
| 反哺 | "feeding back" |

When encountering Chinese idioms or four-character phrases, find a natural English equivalent and include the original Chinese in parentheses.
