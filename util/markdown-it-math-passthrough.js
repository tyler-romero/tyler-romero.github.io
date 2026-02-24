/**
 * markdown-it plugin that protects KaTeX math delimiters from markdown
 * processing. Matches \(...\) for inline math and \[...\] for display math
 * (KaTeX auto-render defaults). Content passes through as raw text so
 * client-side KaTeX handles rendering.
 *
 * Must be registered BEFORE the escape rule to prevent markdown-it from
 * stripping the leading backslash.
 */
export default function mathPassthroughPlugin(md) {
  md.inline.ruler.before("escape", "math_inline", mathInlineRule);
}

/**
 * Inline rule — match \(...\) and \[...\].
 *
 * Fires before the escape rule so the backslash-paren / backslash-bracket
 * sequences are consumed intact. Emits content as html_inline, preventing
 * any further inline processing (emphasis, links, etc.) of math interiors.
 */
function mathInlineRule(state, silent) {
  const src = state.src;
  const pos = state.pos;
  const max = state.posMax;

  // Must start with backslash
  if (src.charCodeAt(pos) !== 0x5c /* \ */) return false;
  if (pos + 1 >= max) return false;

  const nextChar = src.charCodeAt(pos + 1);
  let open, close;

  if (nextChar === 0x28 /* ( */) {
    open = "\\(";
    close = "\\)";
  } else if (nextChar === 0x5b /* [ */) {
    open = "\\[";
    close = "\\]";
  } else {
    return false;
  }

  // Reject \\( and \\[ — double backslash is a LaTeX line break, not math
  if (pos > 0 && src.charCodeAt(pos - 1) === 0x5c) return false;

  const contentStart = pos + 2;
  if (contentStart >= max) return false;

  // Scan for closing delimiter
  let end = contentStart;
  while (end < max) {
    if (
      src.charCodeAt(end) === 0x5c /* \ */ &&
      end + 1 < max &&
      src.charAt(end + 1) === close.charAt(1)
    ) {
      // Make sure it's not \\) or \\] (escaped backslash before closing paren)
      if (end > 0 && src.charCodeAt(end - 1) === 0x5c) {
        end += 2;
        continue;
      }

      if (end === contentStart) {
        end += 2;
        continue; // empty math
      }

      if (silent) return true;

      const token = state.push("html_inline", "", 0);
      token.content = src.slice(pos, end + 2);
      state.pos = end + 2;
      return true;
    }
    end++;
  }

  return false;
}
