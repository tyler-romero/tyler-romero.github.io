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

function renderTableOfContents(content) {
  const placeholder = /<nav class="toc" aria-label="Table of contents"><\/nav>/;
  if (!placeholder.test(content)) return content;

  const headings = Array.from(
    content.matchAll(/<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/gi),
    ([, level, id, label]) => ({ level: Number(level), id, label }),
  );

  if (headings.length < 3) return content;

  const items = [];
  let currentH2;
  for (const heading of headings) {
    if (heading.level === 2) {
      currentH2 = { ...heading, children: [] };
      items.push(currentH2);
    } else if (currentH2) {
      currentH2.children.push(heading);
    } else {
      items.push({ ...heading, children: [] });
    }
  }

  const toc = `<nav class="toc" aria-label="Table of contents"><ol>${items
    .map((item) => {
      const children = item.children.length
        ? `<ol>${item.children
            .map(
              (child) => `<li><a href="#${child.id}">${child.label}</a></li>`,
            )
            .join("")}</ol>`
        : "";
      return `<li><a href="#${item.id}">${item.label}</a>${children}</li>`;
    })
    .join("")}</ol></nav>`;

  return content.replace(placeholder, toc);
}

export default function (eleventyConfig) {
  // Copy `src/assets` to `_site/assets`
  eleventyConfig.addPassthroughCopy("src/assets");

  // Copy some more files to `_site`
  eleventyConfig.addPassthroughCopy("src/CNAME");
  eleventyConfig.addPassthroughCopy("src/robots.txt");
  eleventyConfig.addPassthroughCopy("src/manifest.json");

  // Asset Watch Targets
  eleventyConfig.addWatchTarget("./src/assets");

  /* Markdown Configuration */
  eleventyConfig.setLibrary("md", tufteMdWrapper);
  eleventyConfig.addFilter("markdown", tufteMdWrapper.render);
  eleventyConfig.addFilter("markdownInline", tufteMdWrapper.renderInline);

  // Date stuff
  eleventyConfig.addShortcode("year", () => `${new Date().getFullYear()}`); // useful for copyright
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
      subtitle: "Notes on Machine Learning and related topics.",
      base: "https://tylerromero.com/",
      author: {
        name: "Tyler Romero",
        email: "tyler.alexander.romero@gmail.com",
      },
    },
  });

  // Image Shortcode And Optimizations
  eleventyConfig.addPlugin(eleventyImageTransformPlugin, {
    extensions: "html",
    formats: ["webp", "auto"], // "auto" means use the original format
    sharpOptions: {
      animated: true, // Enable animated GIF and WebP support
    },
    widths: [300, 600, 900, "auto"], // mobile, tablet, desktop viewport widths, and original size
    defaultAttributes: {
      loading: "lazy",
      decoding: "async",
      sizes: "(max-width: 900px) 100vw, 900px",
      class: "responsive-image",
    },
    urlPath: "/assets/img/",
    outputDir: "_site/assets/img/",
  });

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

  eleventyConfig.addTransform("tableOfContents", function (content) {
    if (!this.page.outputPath?.endsWith(".html")) return content;
    return renderTableOfContents(content);
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
