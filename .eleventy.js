const tufteMdWrapper = require('./util/tufteMdWrapper')

module.exports = function(eleventyConfig) {
    // Copy `src/assets` to `_site/assets`
    eleventyConfig.addPassthroughCopy("src/assets");

    // Copy some more files to `_site`
    eleventyConfig.addPassthroughCopy("src/CNAME");
    eleventyConfig.addPassthroughCopy("src/robots.txt");


    // Asset Watch Targets
	eleventyConfig.addWatchTarget('./src/assets')

    /* Markdown Configuration */
	let options = {
		react: false,
    };

	// Markdown
	eleventyConfig.setLibrary("md", tufteMdWrapper)
    eleventyConfig.addFilter("markdown", tufteMdWrapper.render)
    eleventyConfig.addFilter("markdownInline", tufteMdWrapper.renderInline)


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
        }
    };
};