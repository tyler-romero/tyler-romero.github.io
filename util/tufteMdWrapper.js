import markdownIt from "markdown-it";
import anchor from "markdown-it-anchor";
import mathPassthroughPlugin from "./markdown-it-math-passthrough.js";
import sidenotePlugin from "./markdown-it-tufte-sidenotes.js";

const md = markdownIt({ html: true, typographer: false })
  .use(mathPassthroughPlugin)
  .use(sidenotePlugin)
  .use(anchor);

/**
 * Split rendered HTML at <h2> boundaries and wrap each segment in <section>.
 * Content before the first <h2> gets its own section.
 */
function wrapInSections(html) {
  const parts = html.split(/(?=<h2[\s>])/);
  if (parts.length <= 1) return html;
  return parts
    .filter((p) => p.trim())
    .map((p) => `<section>\n${p}</section>\n`)
    .join("");
}

export const tufteMdWrapper = {
  render: function (text, wrap = true) {
    let html = md.render(text);
    html = wrapInSections(html);
    if (wrap && !html.includes("<section>")) {
      html = `<section>\n${html}</section>\n`;
    }
    return html;
  },

  renderInline: function (text) {
    return md.renderInline(String(text));
  },
};
