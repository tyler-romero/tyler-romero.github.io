---
name: generate-site-images
description: "Generate, edit, curate, and publish traceable illustrations for this website with OpenAI image models or Microsoft MAI-Image-2.5 through Azure. Use for hero images, article illustrations, social cards, transparent-background assets, image-to-image edits, or any request to create site-consistent AI imagery. Applies the site's shared aesthetic prompt, records asset lineage, and guides selected outputs into src/assets/img/."
---

# Generate Site Images

Create illustrations with a consistent editorial, handmade visual language. Use the bundled script for every API call so prompts, dimensions, output handling, and provenance remain reproducible.

## Workflow

1. Read the target page and `style_guide.md`. Identify the subject, purpose, and intended placement. If no target page is provided, ask for the page or a concrete visual brief.
2. Choose a preset:
   - `hero`: wide composition with quiet space for an overlaid title.
   - `article`: explanatory editorial illustration with a clear focal point.
   - `social`: crop-safe preview image with a strong central silhouette.
   - `square`: square illustration for flexible reuse.
3. Write a concrete subject or edit prompt. Describe scene, objects, mood, lighting, composition, and what must remain unchanged; do not repeat the shared aesthetic language. Supply one or more `--input-image` values to edit instead of generating from scratch.
4. Run `--dry-run` and inspect the composed prompt and request settings.
5. Before a paid call, state the model, dimensions, quality, and candidate count. Confirm when the user has not already authorized that cost or when generating multiple candidates.
6. Generate exploratory batches into the ignored `imggen/pipeline_artifacts/` directory. Accumulate at least 20 variations before final curation unless the user explicitly waives the site's guideline; confirm the cost before additional batches.
7. Inspect candidates at full size. Reject malformed details, accidental text, generic AI gloss, or imagery that does not blend with the site background.
8. Generate or select a final image, then save it into `src/assets/img/` using a descriptive kebab-case filename. For artwork that should dissolve into the page rather than retain a rectangular canvas, read [references/transparency.md](./references/transparency.md) and publish a separate RGBA derivative.
9. Add descriptive alt text and wire the image into page frontmatter or Markdown. Use a separate compressed `featured_image` when the full hero source is unnecessarily large for social crawlers.
10. Run `npm run check`.

## Generator

Run from the repository root:

```bash
npm run image:generate -- \
  --preset hero \
  --name nanogpt-harbor \
  --prompt "A small sailboat crossing a quiet Pacific Northwest harbor at golden hour" \
  --quality low \
  --count 4 \
  --dry-run
```

Remove `--dry-run` after reviewing the composed prompt. Generate a final directly into the site assets when appropriate:

```bash
npm run image:generate -- \
  --preset hero \
  --name nanogpt-harbor \
  --prompt-file notes/nanogpt-image-brief.txt \
  --quality high \
  --output-dir src/assets/img
```

Edit an existing image while preserving its lineage:

```bash
npm run image:edit -- \
  --provider azure \
  --base-url https://example-resource.services.ai.azure.com \
  --input-image imggen/pipeline_artifacts/example/example.png \
  --preset hero \
  --name example-revision \
  --prompt "Move the focal subject farther right while preserving the medium and palette"
```

For transparent OpenAI output, use `--background transparent`. The script defaults to `gpt-image-1.5` because `gpt-image-2` does not support transparency.

The script writes PNG files plus a JSON manifest containing the complete request and cryptographic lineage metadata. Read [references/lineage.md](./references/lineage.md) when editing, copying selected assets, or auditing provenance. It never records credentials.

## API Setup

Read [references/providers.md](./references/providers.md) when selecting OpenAI or Azure, configuring credentials, overriding models or endpoints, or choosing provider-compatible sizes.

Fill in `OPENAI_API_KEY` or `AZURE_API_KEY` in the repository-root `.env` file. The generator loads it automatically without overriding a key already exported in the shell. Select the provider, model, and base URL at runtime; do not put non-secret endpoint configuration in `.env`. Never commit API keys, `.env` files, or raw responses containing sensitive metadata.

## Aesthetic Guardrails

The generator automatically appends [assets/site-aesthetic.txt](./assets/site-aesthetic.txt) and the selected preset's composition rules. Preserve these shared instructions unless the user explicitly requests a different art direction.

- Prefer calm editorial scenes over spectacle.
- Use tactile physical media, simplified shapes, and restrained detail.
- Keep typography out of generated images.
- Reserve low-detail space where site text will overlay a hero.
- Avoid glossy 3D rendering, neon futurism, photorealistic stock imagery, and default AI-art aesthetics.
- Treat generated output as a draft: crop, color-grade, or paint over artifacts when needed.

## Script Reference

```text
--prompt <text> | --prompt-file <path>
--input-image <path> (repeatable)
--mask <png-path>
--parent-manifest <path> (repeatable)
--provider openai|azure
--model <model-name>
--base-url <api-base-url>
--preset hero|article|social|square
--name <kebab-case-name>
--size <width>x<height>
--quality low|medium|high|auto
--background auto|opaque|transparent
--input-fidelity low|high
--count <1-10>
--output-dir <path>
--no-style
--force
--dry-run
```

Use `--no-style` only for debugging or when the user explicitly requests a different aesthetic.
