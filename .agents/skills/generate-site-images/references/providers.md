# Image API Providers

Keep secrets in `.env` or the shell. Pass provider, model, and endpoint configuration as runtime arguments so manifests record the exact non-secret configuration used.

## OpenAI

Set the credential:

```bash
export OPENAI_API_KEY="..."
```

Generate with the defaults (`gpt-image-2` and `https://api.openai.com/v1`):

```bash
npm run image:generate -- \
  --provider openai \
  --preset hero \
  --name example-hero \
  --prompt "A quiet technical landscape"
```

The OpenAI request sends `quality` and accepts `low`, `medium`, `high`, or `auto`. Custom dimensions must use multiples of 16, stay between 655,360 and 8,294,400 total pixels, and remain within a 3:1 aspect ratio.

Edit one or more reference images by repeating `--input-image`. Add an optional PNG `--mask` for masked edits. Use `--input-fidelity high` with `gpt-image-1.5` or `gpt-image-1` when details such as faces or logos must be preserved.

Request native transparency with:

```bash
npm run image:edit -- \
  --provider openai \
  --background transparent \
  --input-image src/assets/img/source.png \
  --preset hero \
  --name transparent-revision \
  --prompt "Preserve the painted subject and remove the paper background"
```

When `--background transparent` is present and `--model` is omitted, the script selects `gpt-image-1.5` and emits PNG. It rejects `gpt-image-2` for transparency. Legacy OpenAI image models use `1024x1024`, `1536x1024`, or `1024x1536`; non-square site presets automatically use `1536x1024`.

## Azure MAI-Image-2.5

Set the credential:

```bash
export AZURE_API_KEY="..."
```

Pass the Azure resource URL at runtime. The script accepts either the resource root or the older `/openai/v1` form and normalizes it to the current MAI route:

```bash
npm run image:generate -- \
  --provider azure \
  --model MAI-Image-2.5 \
  --base-url https://tylerromero-proj-eastus-resource.services.ai.azure.com \
  --preset hero \
  --name example-mai-image \
  --prompt "A red fox in an autumn forest"
```

The Azure request calls `POST /mai/v1/images/generations`, authenticates with `AZURE_API_KEY` in the `api-key` header, sends dimensions as separate `width` and `height` fields, and requests one image per API call. The script performs repeated calls when `--count` is greater than one. The `--model` value is the Azure deployment name; `MAI-Image-2.5` is the default.

MAI dimensions must each be at least 768 pixels and their product must not exceed 1,048,576 pixels. The native `hero` size of `1344x768` satisfies those limits.

Supply one PNG or JPEG with `--input-image` to call `POST /mai/v1/images/edits`. MAI accepts one source image per edit, does not expose masks or native transparent-background output, and performs one edit per API call. Use the OpenAI transparent workflow or the deterministic Sharp process in [transparency.md](./transparency.md) when alpha is required.

## Shared behavior

Both providers save decoded PNG files, record provider/model/endpoint settings in the manifest, and retry HTTP 429 and 5xx responses with bounded exponential backoff.

Use `--dry-run` to inspect the complete composed request summary without requiring credentials or making a paid request.

References:

- `https://developers.openai.com/api/docs/guides/image-generation`
- `https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-mai`
