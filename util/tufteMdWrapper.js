import markdownIt from "markdown-it";
import anchor from "markdown-it-anchor";
import katexPlugin from "./markdown-it-katex.js";
import sidenotePlugin from "./markdown-it-tufte-sidenotes.js";

const md = markdownIt({ html: true, typographer: false })
  .use(katexPlugin)
  .use(sidenotePlugin)
  .use(anchor);

md.renderer.rules.table_open = () =>
  '<div class="table-wrapper" role="region" aria-label="Scrollable table" tabindex="0">\n<table>\n';
md.renderer.rules.table_close = () => "</table>\n</div>\n";

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
