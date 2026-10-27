import {
  EleventyHtmlBasePlugin,
  InputPathToUrlTransformPlugin,
} from "@11ty/eleventy";
import { eleventyImageTransformPlugin } from "@11ty/eleventy-img";
import { feedPlugin } from "@11ty/eleventy-plugin-rss";
import Cite from "citation-js";
import CleanCSS from "clean-css";
import { DateTime } from "luxon";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { tufteMdWrapper } from "./util/tufteMdWrapper.js";

const lastModifiedCache = new Map();

function getLastModifiedDate(filepath) {
  if (lastModifiedCache.has(filepath)) {
    return lastModifiedCache.get(filepath);
  }

  let modified;
  try {
    modified = execFileSync(
      "git",
      ["log", "-1", "--format=%cI", "--", filepath],
      { encoding: "utf8" },
    ).trim();
  } catch {
    // Fall back to the filesystem for untracked files or non-Git builds.
  }

  if (!modified) {
    modified = fs.statSync(filepath).mtime.toISOString();
  }

  lastModifiedCache.set(filepath, modified);
  return modified;
}

const namedEntities = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: "\u00a0",
};

function decodeHtmlEntities(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, entity) => {
    if (entity[0] === "#") {
      const codePoint =
        entity[1].toLowerCase() === "x"
          ? parseInt(entity.slice(2), 16)
          : parseInt(entity.slice(1), 10);
      return Number.isNaN(codePoint) ? match : String.fromCodePoint(codePoint);
    }
    return namedEntities[entity.toLowerCase()] ?? match;
  });
}

// Render inline Markdown to unescaped plain text so templates escape it exactly once.
function markdownToPlainText(text) {
  const html = tufteMdWrapper.renderInline(text ?? "");
  return decodeHtmlEntities(html.replace(/<[^>]*>/g, ""))
    .replace(/\s+/g, " ")
    .trim();
}

function addStyledListRoles(content) {
  return content.replace(
    /(<ul class="(?:item-list|post-list)" role="list">)([\s\S]*?)(<\/ul>)/g,
    (_match, openingTag, listItems, closingTag) =>
      openingTag +
      listItems.replace(/<li(?![^>]*\brole=)/g, '<li role="listitem"') +
      closingTag,
  );
}

export default function (eleventyConfig) {
  // Copy assets that are not handled by the image transform plugin.
  eleventyConfig.addPassthroughCopy({
    "src/assets/fonts": "assets/fonts",
    // Self-contained interactive figures embedded via iframe.
    "src/assets/widgets": "assets/widgets",
    "node_modules/katex/dist/katex.min.css": "assets/katex/katex.min.css",
    "node_modules/katex/dist/fonts": "assets/katex/fonts",
    "node_modules/rough-notation/lib/rough-notation.iife.js":
      "assets/js/rough-notation.iife.js",
    "src/assets/img/favicon.ico": "assets/img/favicon.ico",
    "src/assets/img/badge_selection_order.webm":
      "assets/img/badge_selection_order.webm",
    // Social preview images need stable, unhashed URLs.
    "src/assets/img/headshot_2.jpg": "assets/img/headshot_2.jpg",
    "src/assets/img/golden-gardens-social.jpg":
      "assets/img/golden-gardens-social.jpg",
  });

  // Copy some more files to `_site`
  eleventyConfig.ignores.add("src/assets/widgets/**");
  eleventyConfig.addPassthroughCopy("src/CNAME");
  eleventyConfig.addPassthroughCopy("src/robots.txt");
  eleventyConfig.addPassthroughCopy("src/manifest.json");

  // Asset Watch Targets
  eleventyConfig.addWatchTarget("./src/assets");

  /* Markdown Configuration */
  eleventyConfig.setLibrary("md", tufteMdWrapper);
  eleventyConfig.addFilter("markdown", tufteMdWrapper.render);
  eleventyConfig.addFilter("markdownInline", tufteMdWrapper.renderInline);
  eleventyConfig.addFilter("plainText", markdownToPlainText);

  // Date stuff
  eleventyConfig.addShortcode("year", () => `${new Date().getFullYear()}`); // useful for copyright

  // Static tweet card for margin notes. Inline-only markup on one line, since margin notes are spans inside
  // paragraphs; markdown and KaTeX characters are escaped so the tweet text renders verbatim.
  eleventyConfig.addShortcode("tweet", function (id) {
    const tweet = (this.ctx?.tweets ?? this.ctx?.environments?.tweets ?? {})[id];
    if (!tweet) throw new Error(`tweet shortcode: no data for ${id} in src/_data/tweets.json`);
    const esc = (s) =>
      String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/[\\*_`[\]$|~^]/g, (c) => `&#${c.charCodeAt(0)};`);
    const media = (tweet.media ?? [])
      .map((src) => `<img class="tweet-media" src="${src}" alt="Image attached to the post">`)
      .join("");
    // Math that the post showed as an image is re-typeset with KaTeX; left unescaped on purpose.
    // "@@MATH0@@" in the text places math[0] inline where the image appeared; unplaced math goes at the end.
    const mathSpans = (tweet.math ?? []).map((m) => `<span class="tweet-math">\\(\\displaystyle ${m}\\)</span>`);
    const placed = new Set();
    const text = esc(tweet.text)
      .replace(/\n/g, "<br>")
      .replace(/@@MATH(\d+)@@/g, (_, i) => (placed.add(Number(i)), mathSpans[Number(i)] ?? ""));
    const math = mathSpans.filter((_, i) => !placed.has(i)).join("");
    return (
      `<a class="tweet-card" href="${tweet.url}">` +
      `<span class="tweet-author"><span class="tweet-name">${esc(tweet.name)}</span> ` +
      `<span class="tweet-handle">@${esc(tweet.handle)}</span></span>` +
      `<span class="tweet-text">${text}</span>${math}${media}` +
      `<span class="tweet-date">${esc(tweet.date)} ↗</span></a>`
    );
  });
  eleventyConfig.addFilter("postDate", (dateObj) => {
    if (typeof dateObj === "string") {
      dateObj = DateTime.fromISO(dateObj);
    } else {
      dateObj = DateTime.fromJSDate(dateObj);
    }
    return dateObj.toFormat("MMM d, yyyy");
  });
  eleventyConfig.addFilter("lastModifiedDate", function (filepath) {
    return getLastModifiedDate(filepath);
  });

  eleventyConfig.addFilter("jsonLd", function (value) {
    return JSON.stringify(value);
  });

  // Add wordCount filter
  eleventyConfig.addFilter("wordCount", function (content) {
    if (typeof content !== "string") {
      return 0;
    }
    const words = content.split(/\s+/);
    return words.length;
  });

  // CSS Minification
  eleventyConfig.addFilter("cssmin", function (code) {
    return new CleanCSS({}).minify(code).styles;
  });

  // RSS Feed
  eleventyConfig.addPlugin(feedPlugin, {
    type: "atom",
    outputPath: "/feed.xml",
    collection: {
      name: "post", // Only posts, not recipes
      limit: 10, // 0 for no limit
    },
    metadata: {
      language: "en",
      title: "Tyler's Technical Blog",
      subtitle:
        "Technical writing on language models, training systems, and machine learning research.",
      base: "https://tylerromero.com/",
      author: {
        name: "Tyler Romero",
        email: "tyler.alexander.romero@gmail.com",
      },
    },
  });

  // Image Shortcode And Optimizations
  // When src/assets/img/<name>-mobile.<ext> exists alongside src/assets/img/<name>.<ext>,
  // emit <source media="(max-width: 760px)"> entries inside the same <picture>.
  eleventyConfig.htmlTransformer.addPosthtmlPlugin(
    "html",
    () => (tree) => {
      function injectMobileImgs(node) {
        const children = Array.isArray(node) ? node : node?.content;
        if (!Array.isArray(children)) return;
        for (let i = 0; i < children.length; i++) {
          const child = children[i];
          if (child && typeof child === "object") {
            const src = child.attrs?.src;
            const match =
              child.tag === "img" &&
              !child.attrs?.["data-mobile-art"] &&
              src?.match(/^\/assets\/img\/([^/]+?)(\.[a-z0-9]+)$/i);
            if (match && !match[1].endsWith("-mobile")) {
              const mobileRel = `/assets/img/${match[1]}-mobile${match[2]}`;
              if (fs.existsSync(`src${mobileRel}`)) {
                children.splice(i, 0, {
                  tag: "img",
                  attrs: {
                    src: mobileRel,
                    alt: "",
                    sizes: "150vw",
                    "data-mobile-art": "true",
                  },
                });
                i++;
              }
            }
            injectMobileImgs(child);
          }
        }
      }
      injectMobileImgs(tree);
      return tree;
    },
    { priority: 0 },
  );

  eleventyConfig.addPlugin(eleventyImageTransformPlugin, {
    extensions: "html",
    formats: ["webp", "auto"], // "auto" means use the original format
    sharpOptions: {
      animated: true, // Enable animated GIF and WebP support
    },
    widths: [300, 450, 600, 750, 900, 1200, "auto"],
    defaultAttributes: {
      loading: "lazy",
      decoding: "async",
      sizes: "(max-width: 760px) 150vw, (max-width: 900px) 100vw, 900px",
      class: "responsive-image",
    },
    urlPath: "/assets/img/",
    outputDir: "_site/assets/img/",
  });

  eleventyConfig.htmlTransformer.addPosthtmlPlugin(
    "html",
    () => (tree) => {
      function mergeMobilePictures(node) {
        const children = Array.isArray(node) ? node : node?.content;
        if (!Array.isArray(children)) return;
        for (let i = 0; i < children.length; i++) {
          const child = children[i];
          if (child && typeof child === "object") {
            const next = children[i + 1];
            const isMobilePic =
              child.tag === "picture" &&
              Array.isArray(child.content) &&
              child.content.some(
                (c) => c?.tag === "img" && c.attrs?.["data-mobile-art"] === "true",
              );
            if (isMobilePic && next?.tag === "picture" && Array.isArray(next.content)) {
              const mobileImg = child.content.find((c) => c?.tag === "img");
              const mobileSources = [];
              for (const c of child.content) {
                if (c?.tag === "source") {
                  mobileSources.push({
                    tag: "source",
                    attrs: {
                      media: "(max-width: 760px)",
                      ...c.attrs,
                      width: mobileImg?.attrs?.width,
                      height: mobileImg?.attrs?.height,
                    },
                  });
                }
              }
              if (mobileImg?.attrs) {
                mobileSources.push({
                  tag: "source",
                  attrs: {
                    media: "(max-width: 760px)",
                    srcset: mobileImg.attrs.srcset || mobileImg.attrs.src,
                    sizes: mobileImg.attrs.sizes || "150vw",
                    width: mobileImg.attrs.width,
                    height: mobileImg.attrs.height,
                  },
                });
              }
              next.content.unshift(...mobileSources);
              children.splice(i, 1);
              i--;
              continue;
            }
            mergeMobilePictures(child);
          }
        }
      }
      mergeMobilePictures(tree);
      return tree;
    },
    { priority: -2 },
  );

  // Plugins
  eleventyConfig.addPlugin(EleventyHtmlBasePlugin);
  eleventyConfig.addPlugin(InputPathToUrlTransformPlugin);

  // BibTeX → static bibliography (replaces client-side bibtex-js)
  eleventyConfig.addTransform("bibtex", function (content) {
    if (!this.page.outputPath?.endsWith(".html")) return content;

    const regex =
      /<textarea id="bibtex_input"[^>]*>([\s\S]*?)<\/textarea>\s*(?:<div id="bibtex_display"><\/div>)?/;
    const match = content.match(regex);
    if (!match) return content;

    const bibtex = match[1];
    const cite = new Cite(bibtex);
    let html = cite.format("bibliography", {
      format: "html",
      template: "apa",
      lang: "en-US",
    });

    // Linkify bare URLs that citation-js leaves as plain text
    html = html.replace(
      /(?<!href=["'])(?<!<a[^>]*>)(https?:\/\/[^\s<]+)/g,
      '<a href="$1">$1</a>',
    );

    return content.replace(regex, `<div class="bibliography">${html}</div>`);
  });

  eleventyConfig.addTransform("styledListRoles", function (content) {
    if (!this.page.outputPath?.endsWith(".html")) return content;
    return addStyledListRoles(content);
  });

  return {
    // When a passthrough file is modified, rebuild the pages:
    passthroughFileCopy: true,
    dir: {
      input: "src",
      layouts: "_layouts",
    },
  };
}
