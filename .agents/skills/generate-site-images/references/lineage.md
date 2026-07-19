# Image Lineage

Every successful generation or edit writes a schema-versioned manifest beside its PNG outputs. Preserve this manifest with exploratory artifacts and use it to trace published images back to their source assets and API requests.

## Manifest structure

```json
{
  "schemaVersion": 2,
  "operation": "edit",
  "provider": "openai",
  "model": "gpt-image-1.5",
  "prompt": "...",
  "inputs": [
    {
      "role": "input",
      "path": "imggen/pipeline_artifacts/source/source.png",
      "mimeType": "image/png",
      "bytes": 123456,
      "sha256": "..."
    }
  ],
  "outputs": [
    {
      "path": "imggen/pipeline_artifacts/edit/edit.png",
      "mimeType": "image/png",
      "bytes": 123456,
      "sha256": "...",
      "width": 1536,
      "height": 1024,
      "hasAlpha": true
    }
  ],
  "lineage": {
    "schemaVersion": 1,
    "runId": "uuid",
    "operation": "edit",
    "parents": [],
    "parentManifests": []
  }
}
```

## Parent discovery

For every `--input-image` and `--mask`, the script records its path, byte length, MIME type, and SHA-256 digest. It scans the input directories for sibling `*.manifest.json` files whose outputs match the input path and records their run IDs automatically.

Use repeatable `--parent-manifest` arguments when an image was copied, renamed, or published outside its original artifact directory:

```bash
npm run image:edit -- \
  --input-image src/assets/img/published-hero.png \
  --parent-manifest imggen/pipeline_artifacts/original/original.manifest.json \
  --name revised-hero \
  --prompt "Preserve the subject while simplifying the background"
```

## Publishing

- Keep exploratory manifests under `imggen/pipeline_artifacts/`.
- Copy the selected PNG into `src/assets/img/`; do not overwrite its source artifact.
- Retain the source manifest path in commit context or create the next edit with `--parent-manifest`.
- Compare SHA-256 values when verifying that a published copy matches a selected artifact.
- Treat deterministic post-processing such as cropping or alpha-mask creation as a new lineage operation. Until those local transforms have their own CLI command, document them in the parent manifest or create a small companion manifest rather than silently replacing the generated asset.

Manifests intentionally record non-secret request configuration and never include API keys or authorization headers.
