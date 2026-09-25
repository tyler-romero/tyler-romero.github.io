# tylerromero.com

Personal blog about AI/ML at [tylerromero.com](https://www.tylerromero.com/).

## Quick start

```
npm install
make serve    # local dev server
make build    # build to _site/
make format   # prettier + djlint
```

## Structure

- `src/posts/` — blog posts in Tufte Markdown
- `src/recipe-box/` — recipe collection
- `src/_layouts/` — Nunjucks page layouts
- `src/assets/` — CSS, fonts, images
  - Excalidraw diagrams keep their editable `.excalidraw` source next to the PNG in `src/assets/img/`. Run `make excalidraw` to re-render them as transparent PNGs at the `exportScale` stored in each scene (default 2). This needs a local Chrome, Edge, or Chromium; set `CHROME_PATH` to point at a specific one.
- `style_guide.md` — visual design reference
