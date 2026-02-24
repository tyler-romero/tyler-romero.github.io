/**
 * markdown-it plugin for Tufte-style sidenotes and margin notes.
 *
 * Syntax (standard Markdown footnotes):
 *   Inline reference: [^ref-id]
 *   Definition:       [^ref-id]: Sidenote content here
 *   Margin note:      [^ref-id]: {-} Margin note content here
 *
 * Sidenotes render as numbered notes in the margin (CSS counter via .sidenote-number).
 * Margin notes (prefixed with {-}) render unnumbered with a ⊕ toggle.
 */
export default function sidenotePlugin(md) {
  // Block rule: collect [^id]: definitions and suppress their output.
  md.block.ruler.before("reference", "sidenote_def", sidenoteDefRule, {
    alt: ["paragraph", "reference"],
  });

  // Inline rule: replace [^id] references with sidenote/marginnote HTML.
  md.inline.ruler.after("image", "sidenote_ref", sidenoteRefRule);
}

/**
 * Block rule — parse footnote definitions.
 *
 * Matches lines of the form:
 *   [^id]: content
 *       continuation line (4-space indent)
 *
 * Stores definitions in state.env.sidenotes and consumes the lines
 * without emitting any tokens.
 */
function sidenoteDefRule(state, startLine, endLine, silent) {
  const pos = state.bMarks[startLine] + state.tShift[startLine];
  const max = state.eMarks[startLine];

  // Quick checks: need at least [^x]:
  if (max - pos < 5) return false;
  if (state.src.charCodeAt(pos) !== 0x5b /* [ */) return false;
  if (state.src.charCodeAt(pos + 1) !== 0x5e /* ^ */) return false;

  const firstLine = state.src.slice(pos, max);

  // Must contain ]:
  const colonIdx = firstLine.indexOf("]:");
  if (colonIdx === -1) return false;

  const id = firstLine.slice(2, colonIdx);
  if (!id) return false;

  if (silent) return true;

  let content = firstLine.slice(colonIdx + 2).trim();

  // Collect continuation lines (indented by 4+ spaces).
  let nextLine = startLine + 1;
  while (nextLine < endLine) {
    const lineStart = state.bMarks[nextLine];
    const lineEnd = state.eMarks[nextLine];
    const rawLine = state.src.slice(lineStart, lineEnd);

    // Continuation: must start with 4+ spaces and have actual content.
    if (/^ {4}/.test(rawLine) && rawLine.trim()) {
      content += " " + rawLine.trim();
      nextLine++;
    } else {
      break;
    }
  }

  if (!state.env.sidenotes) {
    state.env.sidenotes = {};
  }
  state.env.sidenotes[id] = content;

  state.line = nextLine;
  return true;
}

/**
 * Inline rule — replace [^id] references with rendered HTML.
 *
 * Looks up the definition collected by the block rule and emits the
 * label + checkbox + span structure expected by Tufte CSS.
 */
function sidenoteRefRule(state, silent) {
  const src = state.src;
  const pos = state.pos;
  const max = state.posMax;

  if (pos + 2 >= max) return false;
  if (src.charCodeAt(pos) !== 0x5b /* [ */) return false;
  if (src.charCodeAt(pos + 1) !== 0x5e /* ^ */) return false;

  // Find closing ]
  let end = pos + 2;
  while (end < max && src.charCodeAt(end) !== 0x5d /* ] */) {
    end++;
  }
  if (end >= max) return false;

  const id = src.slice(pos + 2, end);
  if (!id) return false;

  const defs = state.env.sidenotes;
  if (!defs || !(id in defs)) return false;

  if (silent) return true;

  let content = defs[id];
  const isMarginNote = content.startsWith("{-}");
  if (isMarginNote) {
    content = content.slice(3).trim();
  }

  const rendered = state.md.renderInline(content);
  const prefix = isMarginNote ? "mn" : "sn";
  const noteId = `${prefix}-${id}`;

  const token = state.push("html_inline", "", 0);
  if (isMarginNote) {
    token.content =
      `<label for="${noteId}" class="margin-toggle">&#8853;</label>` +
      `<input type="checkbox" id="${noteId}" class="margin-toggle" />` +
      `<span class="marginnote">${rendered}</span>`;
  } else {
    token.content =
      `<label for="${noteId}" class="margin-toggle sidenote-number"></label>` +
      `<input type="checkbox" id="${noteId}" class="margin-toggle" />` +
      `<span class="sidenote">${rendered}</span>`;
  }

  state.pos = end + 1;
  return true;
}
