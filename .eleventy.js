const CleanCSS = require("clean-css");
const { DateTime } = require("luxon");
const tufteMdWrapper = require("./util/tufteMdWrapper");
const fs = require("fs");

module.exports = function (eleventyConfig) {
  // Copy `src/assets` to `_site/assets`
  eleventyConfig.addPassthroughCopy("src/assets");

  // Copy some more files to `_site`
  eleventyConfig.addPassthroughCopy("src/CNAME");
  eleventyConfig.addPassthroughCopy("src/robots.txt");

  // Asset Watch Targets
  eleventyConfig.addWatchTarget("./src/assets");

  /* Markdown Configuration */
  let options = {
    react: false,
  };

  // Markdown
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
  eleventyConfig.addFilter("wordCount", function(content) {
    if (typeof content !== 'string') {
      return 0;
    }
    const words = content.split(/\s+/);
    return words.length;
  });

  // CSS Minification
  eleventyConfig.addFilter("cssmin", function (code) {
    return new CleanCSS({}).minify(code).styles;
  });

  // Set custom directories for input, output, includes, and data
  return {
    // When a passthrough file is modified, rebuild the pages:
    passthroughFileCopy: true,
    dir: {
      input: "src",
      includes: "_includes",
      layouts: "_layouts",
      data: "_data",
      output: "_site",
    },
  };
};
