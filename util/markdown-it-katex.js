import katex from "katex";

/**
 * Render \(...\) and \[...\] delimiters during the Eleventy build.
 * Register before the escape rule so markdown-it preserves the backslashes.
 */
export default function katexPlugin(md) {
  md.inline.ruler.before("escape", "math_inline", mathInlineRule);
  md.renderer.rules.math_inline = renderMath;
}

function renderMath(tokens, index) {
  const token = tokens[index];

  try {
    return katex.renderToString(token.content, {
      displayMode: token.meta.displayMode,
      output: "htmlAndMathml",
      strict: (errorCode) =>
        errorCode === "newLineInDisplayMode" ? "ignore" : "warn",
      throwOnError: true,
    });
  } catch (error) {
    throw new Error(`Unable to render KaTeX expression: ${token.content}`, {
      cause: error,
    });
  }
}

function mathInlineRule(state, silent) {
  const src = state.src;
  const pos = state.pos;
  const max = state.posMax;

  if (src.charCodeAt(pos) !== 0x5c /* \ */) return false;
  if (pos + 1 >= max) return false;

  const nextChar = src.charCodeAt(pos + 1);
  let close;
  let displayMode;

  if (nextChar === 0x28 /* ( */) {
    close = "\\)";
    displayMode = false;
  } else if (nextChar === 0x5b /* [ */) {
    close = "\\]";
    displayMode = true;
  } else {
    return false;
  }

  // A doubled backslash is a LaTeX line break, not an opening delimiter.
  if (pos > 0 && src.charCodeAt(pos - 1) === 0x5c) return false;

  const contentStart = pos + 2;
  if (contentStart >= max) return false;

  let end = contentStart;
  while (end < max) {
    if (
      src.charCodeAt(end) === 0x5c /* \ */ &&
      end + 1 < max &&
      src.charAt(end + 1) === close.charAt(1)
    ) {
      // Ignore an escaped closing delimiter.
      if (end > 0 && src.charCodeAt(end - 1) === 0x5c) {
        end += 2;
        continue;
      }

      if (end === contentStart) {
        end += 2;
        continue; // empty math
      }

      if (silent) return true;

      const token = state.push("math_inline", "math", 0);
      token.content = src.slice(contentStart, end);
      token.meta = { displayMode };
      state.pos = end + 2;
      return true;
    }
    end++;
  }

  return false;
}
