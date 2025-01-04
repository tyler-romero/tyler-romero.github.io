import configureParser from "@tufte-markdown/parser";

const parse = configureParser({ react: false });

export const tufteMdWrapper = {
  render: function (text, wrap = true) {
    let tmp = parse(text);
    if (wrap && tmp.indexOf("<section>") == -1) {
      return "<section>" + tmp + "</section>";
    } else {
      return tmp;
    }
  },

  renderInline: function (text) {
    return parse(String(text)).replace("<p>", "").replace("</p>", "");
  },
};
