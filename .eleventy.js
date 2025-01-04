import {
  EleventyHtmlBasePlugin,
  InputPathToUrlTransformPlugin,
} from "@11ty/eleventy";
import { eleventyImageTransformPlugin } from "@11ty/eleventy-img";
import { feedPlugin } from "@11ty/eleventy-plugin-rss";
import CleanCSS from "clean-css";
import { DateTime } from "luxon";
import fs from "node:fs";
import { tufteMdWrapper } from "./util/tufteMdWrapper.js";

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
    return DateTime.fromJSDate(dateObj).toFormat("MMMM yyyy");
  });
  eleventyConfig.addFilter("lastModifiedDate", function (filepath) {
    const stat = fs.statSync(filepath);
    return stat.mtime.toISOString();
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
      name: "post",
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

  return {
    // When a passthrough file is modified, rebuild the pages:
    passthroughFileCopy: true,
    dir: {
      input: "src",
      layouts: "_layouts",
    },
  };
}
