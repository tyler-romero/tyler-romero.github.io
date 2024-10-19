.PHONY: help serve build format

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  serve       to run the site locally (with auto-reload)"
	@echo "  build       to build the site"

serve:
	@npx @11ty/eleventy --serve

build:
	@npx @11ty/eleventy

format:
	@npx prettier --write "src/**/*.{html,css,js}"
	@npx prettier --write "util/tufteMdWrapper.js"
	@npx prettier --write ".eleventy.js"
	@find src -name "*.njk" -print0 | xargs -0 djlint --reformat

clean:
	rm -rf _site