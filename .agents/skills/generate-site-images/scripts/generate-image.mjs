#!/usr/bin/env node

import fs from "node:fs/promises";
import { createHash, randomUUID } from "node:crypto";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(scriptPath);
const skillDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(skillDir, "../../..");
const presetsPath = path.join(skillDir, "assets", "presets.json");
const aestheticPath = path.join(skillDir, "assets", "site-aesthetic.txt");
const pngSignature = "89504e470d0a1a0a";

function usage() {
  return `Usage:
  node ${path.relative(repoRoot, scriptPath)} \\
    --prompt "Scene description" \\
    --preset hero|article|social|square \\
    --name output-name [options]

Options:
  --prompt-file <path>   Read the subject prompt from a UTF-8 file
  --input-image <path>   Edit an image; repeat for multiple OpenAI references
  --mask <path>          Optional OpenAI PNG edit mask
  --parent-manifest      Explicit parent manifest; repeat as needed
  --provider <name>      openai or azure (default: openai)
  --model <name>         Image model (provider default if omitted)
  --base-url <url>       API base URL (required for azure)
  --size <WxH>           Override the preset dimensions
  --quality <value>      OpenAI quality: low, medium, high, or auto
  --background <value>   OpenAI background: auto, opaque, or transparent
  --input-fidelity       OpenAI edit fidelity: low or high
  --count <1-10>         Number of candidates (default: 1)
  --output-dir <path>    Exact output directory
  --no-style             Do not append the shared aesthetic prompt
  --force                Overwrite existing output files
  --dry-run              Print the composed request without calling the API
  --help                  Show this help
`;
}

function parseArgs(argv) {
  const options = {};
  const booleans = new Set(["help", "dry-run", "no-style", "force"]);
  const repeatable = new Set(["input-image", "parent-manifest"]);
  const values = new Set([
    "prompt",
    "prompt-file",
    "input-image",
    "mask",
    "parent-manifest",
    "provider",
    "model",
    "base-url",
    "preset",
    "name",
    "size",
    "quality",
    "background",
    "input-fidelity",
    "count",
    "output-dir",
  ]);

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }

    const key = token.slice(2);
    if (booleans.has(key)) {
      options[key] = true;
      continue;
    }
    if (!values.has(key)) throw new Error(`Unknown option: --${key}`);

    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    if (repeatable.has(key)) {
      options[key] ||= [];
      options[key].push(value);
    } else {
      options[key] = value;
    }
    index += 1;
  }

  return options;
}

function parseSize(value) {
  if (value === "auto") {
    return { width: undefined, height: undefined, value };
  }
  const match = /^(\d+)x(\d+)$/i.exec(value);
  if (!match) throw new Error(`Invalid size '${value}'; expected WIDTHxHEIGHT`);
  return { width: Number(match[1]), height: Number(match[2]), value };
}

function validateSize(size, provider, model) {
  if (size.value === "auto") {
    if (provider === "azure") {
      throw new Error("Azure MAI requires explicit WIDTHxHEIGHT dimensions");
    }
    return;
  }
  if (size.width < 1 || size.height < 1) {
    throw new Error(`Size ${size.value} must use positive dimensions`);
  }
  if (provider === "azure") {
    if (
      size.width < 768 ||
      size.height < 768 ||
      size.width * size.height > 1_048_576
    ) {
      throw new Error(
        `Size ${size.value} is invalid for MAI-Image-2.5: width and height must each be at least 768 pixels and total pixels must not exceed 1,048,576`,
      );
    }
    return;
  }

  if (model !== "gpt-image-2") {
    const supported = new Set(["1024x1024", "1536x1024", "1024x1536"]);
    if (!supported.has(size.value)) {
      throw new Error(
        `Size ${size.value} is invalid for ${model}: use 1024x1024, 1536x1024, 1024x1536, or auto`,
      );
    }
    return;
  }

  const pixels = size.width * size.height;
  const ratio = Math.max(size.width / size.height, size.height / size.width);

  if (
    size.width % 16 !== 0 ||
    size.height % 16 !== 0 ||
    pixels < 655_360 ||
    pixels > 8_294_400 ||
    ratio > 3
  ) {
    throw new Error(
      `Size ${size.value} is invalid for gpt-image-2: use multiples of 16, 655,360-8,294,400 total pixels, and an aspect ratio no wider than 3:1`,
    );
  }
}

function slugify(value) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function displayPath(file) {
  const relative = path.relative(repoRoot, file);
  return relative.startsWith("..") ? file : relative;
}

function mimeTypeForFile(file) {
  const extension = path.extname(file).toLowerCase();
  const types = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
  };
  const mimeType = types[extension];
  if (!mimeType) {
    throw new Error(
      `Unsupported image type for ${file}: use PNG, JPEG, or WebP`,
    );
  }
  return mimeType;
}

async function loadImageAsset(value, role = "input") {
  const absolutePath = path.resolve(repoRoot, value);
  const data = await fs.readFile(absolutePath);
  return {
    absolutePath,
    data,
    role,
    path: displayPath(absolutePath),
    mimeType: mimeTypeForFile(absolutePath),
    bytes: data.byteLength,
    sha256: sha256(data),
  };
}

function pngMetadata(bytes) {
  if (bytes.subarray(0, 8).toString("hex") !== pngSignature) {
    return {};
  }
  const colorType = bytes[25];
  return {
    width: bytes.readUInt32BE(16),
    height: bytes.readUInt32BE(20),
    hasAlpha:
      colorType === 4 || colorType === 6 || bytes.includes(Buffer.from("tRNS")),
  };
}

function publicAsset(asset) {
  const { absolutePath, data, ...metadata } = asset;
  return metadata;
}

async function findParentManifests(inputAssets, explicitValues = []) {
  const explicitPaths = new Set(
    explicitValues.map((value) => path.resolve(repoRoot, value)),
  );
  const candidates = new Set(explicitPaths);

  for (const asset of inputAssets) {
    const entries = await fs.readdir(path.dirname(asset.absolutePath));
    for (const entry of entries) {
      if (entry.endsWith(".manifest.json")) {
        candidates.add(path.join(path.dirname(asset.absolutePath), entry));
      }
    }
  }

  const manifests = [];
  for (const manifestPath of candidates) {
    let manifest;
    try {
      manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
    } catch (error) {
      if (explicitPaths.has(manifestPath)) {
        throw new Error(
          `Could not read parent manifest ${displayPath(manifestPath)}: ${error.message}`,
        );
      }
      continue;
    }

    const outputs = manifest.outputs || manifest.files || [];
    const outputPaths = new Set(
      outputs
        .map((output) => (typeof output === "string" ? output : output.path))
        .filter(Boolean)
        .map((output) => path.resolve(repoRoot, output)),
    );
    const matchesInput = inputAssets.some((asset) =>
      outputPaths.has(asset.absolutePath),
    );
    const isExplicit = explicitPaths.has(manifestPath);
    if (!matchesInput && !isExplicit) continue;

    manifests.push({
      path: displayPath(manifestPath),
      runId: manifest.lineage?.runId,
      operation: manifest.lineage?.operation || manifest.operation,
    });
  }

  return manifests;
}

async function loadEnvFile() {
  const envFile = path.join(repoRoot, ".env");
  let contents;
  try {
    contents = await fs.readFile(envFile, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const separator = line.indexOf("=");
    if (separator < 1) continue;

    const key = line.slice(0, separator).trim();
    let value = line.slice(separator + 1).trim();
    if (!/^[A-Z_][A-Z0-9_]*$/.test(key)) continue;
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    if (process.env[key] === undefined) process.env[key] = value;
  }
}

async function requestJson(url, init, attempts = 3) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, init);
      const text = await response.text();
      const data = text ? JSON.parse(text) : {};

      if (response.ok) {
        return {
          data,
          requestId: response.headers.get("x-request-id"),
        };
      }

      const message = data?.error?.message || data?.message || text;
      const error = new Error(`HTTP ${response.status}: ${message}`);
      error.retryable = response.status === 429 || response.status >= 500;
      throw error;
    } catch (error) {
      lastError = error;
      const retryable = error.retryable ?? error instanceof TypeError;
      if (!retryable || attempt === attempts) throw error;
      await sleep(500 * 2 ** (attempt - 1));
    }
  }

  throw lastError;
}

function providerConfig(
  options,
  quality,
  background,
  operation,
  requireCredentials,
) {
  const provider = options.provider || "openai";
  if (!new Set(["openai", "azure"]).has(provider)) {
    throw new Error("--provider must be openai or azure");
  }

  if (provider === "azure" && !options["base-url"]) {
    throw new Error("--base-url is required for the azure provider");
  }

  const apiKey =
    provider === "azure"
      ? process.env.AZURE_API_KEY
      : process.env.OPENAI_API_KEY;
  const keyName = provider === "azure" ? "AZURE_API_KEY" : "OPENAI_API_KEY";
  if (requireCredentials && !apiKey) {
    throw new Error(`${keyName} is required for the ${provider} provider`);
  }

  let baseUrl =
    options["base-url"] ||
    (provider === "openai" ? "https://api.openai.com/v1" : undefined);
  baseUrl = baseUrl.replace(/\/$/, "");

  if (provider === "azure") {
    baseUrl = baseUrl.replace(/\/(?:openai|mai)\/v1$/i, "");
  }

  const model =
    options.model ||
    (provider === "azure"
      ? "MAI-Image-2.5"
      : background === "transparent"
        ? "gpt-image-1.5"
        : "gpt-image-2");
  if (background === "transparent" && model === "gpt-image-2") {
    throw new Error(
      "gpt-image-2 does not support transparent backgrounds; use gpt-image-1.5",
    );
  }
  if (provider === "azure" && background !== "auto") {
    throw new Error("Azure MAI does not expose a background option");
  }
  if (
    options["input-fidelity"] &&
    (provider !== "openai" || operation !== "edit" || model === "gpt-image-2")
  ) {
    throw new Error(
      "--input-fidelity is only supported for OpenAI edits with gpt-image-1.5 or gpt-image-1",
    );
  }

  return {
    provider,
    endpoint:
      provider === "azure"
        ? `${baseUrl}/mai/v1/images/${operation === "edit" ? "edits" : "generations"}`
        : `${baseUrl}/images/${operation === "edit" ? "edits" : "generations"}`,
    headers: {
      ...(apiKey
        ? provider === "azure"
          ? { "api-key": apiKey }
          : { Authorization: `Bearer ${apiKey}` }
        : {}),
    },
    model,
    requestOptions:
      provider === "azure"
        ? {}
        : {
            quality,
            background,
            ...(options["input-fidelity"]
              ? { input_fidelity: options["input-fidelity"] }
              : {}),
          },
  };
}

async function generateImages(config, prompt, size, count) {
  if (config.provider === "azure") {
    const images = [];
    const requestIds = [];

    for (let index = 0; index < count; index += 1) {
      const { data, requestId } = await requestJson(config.endpoint, {
        method: "POST",
        headers: { ...config.headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: config.model,
          prompt,
          width: size.width,
          height: size.height,
        }),
      });
      const image = data?.data?.[0]?.b64_json;
      if (!image) {
        throw new Error(
          `azure returned no image for candidate ${index + 1} of ${count}`,
        );
      }
      images.push(image);
      if (requestId) requestIds.push(requestId);
    }

    return { images, requestIds };
  }

  const { data, requestId } = await requestJson(config.endpoint, {
    method: "POST",
    headers: { ...config.headers, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: config.model,
      prompt,
      size: size.value,
      n: count,
      output_format: "png",
      ...config.requestOptions,
    }),
  });

  const images = data?.data?.map((item) => item.b64_json).filter(Boolean) || [];
  if (images.length !== count) {
    throw new Error(
      `${config.provider} returned ${images.length} images; expected ${count}`,
    );
  }

  return { images, requestIds: requestId ? [requestId] : [] };
}

function appendFormValue(form, key, value) {
  if (value !== undefined) form.append(key, String(value));
}

function appendImage(form, field, asset) {
  form.append(
    field,
    new Blob([asset.data], { type: asset.mimeType }),
    path.basename(asset.absolutePath),
  );
}

async function editImages(config, prompt, size, count, inputAssets, maskAsset) {
  if (config.provider === "azure") {
    if (inputAssets.length !== 1) {
      throw new Error("Azure MAI edits require exactly one --input-image");
    }
    if (!new Set(["image/png", "image/jpeg"]).has(inputAssets[0].mimeType)) {
      throw new Error("Azure MAI edits accept PNG or JPEG input images");
    }
    if (maskAsset) throw new Error("Azure MAI does not support --mask");

    const images = [];
    const requestIds = [];
    for (let index = 0; index < count; index += 1) {
      const form = new FormData();
      appendFormValue(form, "model", config.model);
      appendFormValue(form, "prompt", prompt);
      appendImage(form, "image", inputAssets[0]);
      const { data, requestId } = await requestJson(config.endpoint, {
        method: "POST",
        headers: config.headers,
        body: form,
      });
      const image = data?.data?.[0]?.b64_json;
      if (!image) {
        throw new Error(
          `azure returned no edited image for candidate ${index + 1} of ${count}`,
        );
      }
      images.push(image);
      if (requestId) requestIds.push(requestId);
    }
    return { images, requestIds };
  }

  if (inputAssets.length > 16) {
    throw new Error("OpenAI edits accept at most 16 --input-image values");
  }
  if (maskAsset && maskAsset.mimeType !== "image/png") {
    throw new Error("OpenAI edit masks must be PNG files");
  }

  const form = new FormData();
  appendFormValue(form, "model", config.model);
  appendFormValue(form, "prompt", prompt);
  appendFormValue(form, "size", size.value);
  appendFormValue(form, "n", count);
  appendFormValue(form, "output_format", "png");
  for (const [key, value] of Object.entries(config.requestOptions)) {
    appendFormValue(form, key, value);
  }
  for (const asset of inputAssets) appendImage(form, "image[]", asset);
  if (maskAsset) appendImage(form, "mask", maskAsset);

  const { data, requestId } = await requestJson(config.endpoint, {
    method: "POST",
    headers: config.headers,
    body: form,
  });
  const images = data?.data?.map((item) => item.b64_json).filter(Boolean) || [];
  if (images.length !== count) {
    throw new Error(
      `openai returned ${images.length} images; expected ${count}`,
    );
  }
  return { images, requestIds: requestId ? [requestId] : [] };
}

async function ensureWritable(file, force) {
  if (force) return;
  try {
    await fs.access(file);
    throw new Error(`Refusing to overwrite existing file: ${file}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

async function main() {
  await loadEnvFile();
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(usage());
    return;
  }

  const presets = JSON.parse(await fs.readFile(presetsPath, "utf8"));
  const provider = options.provider || "openai";
  if (!new Set(["openai", "azure"]).has(provider)) {
    throw new Error("--provider must be openai or azure");
  }
  const inputImageValues = options["input-image"] || [];
  const operation = inputImageValues.length ? "edit" : "generate";
  if (options.mask && operation !== "edit") {
    throw new Error("--mask requires at least one --input-image");
  }
  const presetName = options.preset || "article";
  const preset = presets[presetName];
  if (!preset) throw new Error(`Unknown preset: ${presetName}`);

  if (options.prompt && options["prompt-file"]) {
    throw new Error("Use either --prompt or --prompt-file, not both");
  }
  let subjectPrompt = options.prompt;
  if (options["prompt-file"]) {
    subjectPrompt = await fs.readFile(
      path.resolve(repoRoot, options["prompt-file"]),
      "utf8",
    );
  }
  if (!subjectPrompt?.trim()) {
    throw new Error("Provide --prompt or --prompt-file");
  }

  const name = slugify(
    options.name || subjectPrompt.split(/\s+/).slice(0, 8).join(" "),
  );
  if (!name) throw new Error("Could not derive a valid output name");

  const count = Number(options.count || 1);
  if (!Number.isInteger(count) || count < 1 || count > 10) {
    throw new Error("--count must be an integer from 1 to 10");
  }

  const quality = options.quality || "high";
  if (!["low", "medium", "high", "auto"].includes(quality)) {
    throw new Error("--quality must be low, medium, high, or auto");
  }
  const background = options.background || "auto";
  if (!new Set(["auto", "opaque", "transparent"]).has(background)) {
    throw new Error("--background must be auto, opaque, or transparent");
  }
  if (
    options["input-fidelity"] &&
    !new Set(["low", "high"]).has(options["input-fidelity"])
  ) {
    throw new Error("--input-fidelity must be low or high");
  }

  const config = providerConfig(
    options,
    quality,
    background,
    operation,
    !options["dry-run"],
  );
  let size;
  if (!(operation === "edit" && provider === "azure")) {
    const defaultSize =
      config.provider === "openai" && config.model !== "gpt-image-2"
        ? presetName === "square"
          ? "1024x1024"
          : "1536x1024"
        : preset.size;
    size = parseSize(options.size || defaultSize);
    validateSize(size, provider, config.model);
  }

  const inputAssets = await Promise.all(
    inputImageValues.map((value) => loadImageAsset(value)),
  );
  const maskAsset = options.mask
    ? await loadImageAsset(options.mask, "mask")
    : undefined;
  const parentManifests = await findParentManifests(
    [...inputAssets, ...(maskAsset ? [maskAsset] : [])],
    options["parent-manifest"] || [],
  );

  const aesthetic = options["no-style"]
    ? ""
    : (await fs.readFile(aestheticPath, "utf8")).trim();
  const prompt = [
    subjectPrompt.trim(),
    `Composition requirements: ${preset.composition}`,
    aesthetic,
  ]
    .filter(Boolean)
    .join("\n\n");

  const runDirectory = path.resolve(
    repoRoot,
    options["output-dir"] ||
      path.join("imggen", "pipeline_artifacts", `${name}-${timestamp()}`),
  );
  const requestSummary = {
    operation,
    provider: config.provider,
    model: config.model,
    endpoint: config.endpoint,
    preset: presetName,
    ...(size ? { size: size.value } : {}),
    ...(config.provider === "azure" && size
      ? { width: size.width, height: size.height }
      : {}),
    ...config.requestOptions,
    count,
    outputDirectory: displayPath(runDirectory),
    prompt,
    inputs: inputAssets.map(publicAsset),
    ...(maskAsset ? { mask: publicAsset(maskAsset) } : {}),
  };

  if (options["dry-run"]) {
    process.stdout.write(`${JSON.stringify(requestSummary, null, 2)}\n`);
    return;
  }

  await fs.mkdir(runDirectory, { recursive: true });
  const result =
    operation === "edit"
      ? await editImages(config, prompt, size, count, inputAssets, maskAsset)
      : await generateImages(config, prompt, size, count);
  const files = [];
  const outputs = [];

  for (let index = 0; index < result.images.length; index += 1) {
    const suffix =
      result.images.length === 1
        ? ""
        : `-${String(index + 1).padStart(2, "0")}`;
    const file = path.join(runDirectory, `${name}${suffix}.png`);
    await ensureWritable(file, options.force);

    const bytes = Buffer.from(result.images[index], "base64");
    if (bytes.subarray(0, 8).toString("hex") !== pngSignature) {
      throw new Error(`${config.provider} returned non-PNG data for ${file}`);
    }

    await fs.writeFile(file, bytes);
    const outputPath = displayPath(file);
    files.push(outputPath);
    outputs.push({
      path: outputPath,
      mimeType: "image/png",
      bytes: bytes.byteLength,
      sha256: sha256(bytes),
      ...pngMetadata(bytes),
    });
  }

  const manifestFile = path.join(runDirectory, `${name}.manifest.json`);
  await ensureWritable(manifestFile, options.force);
  await fs.writeFile(
    manifestFile,
    `${JSON.stringify(
      {
        schemaVersion: 2,
        ...requestSummary,
        createdAt: new Date().toISOString(),
        files,
        outputs,
        requestIds: result.requestIds,
        lineage: {
          schemaVersion: 1,
          runId: randomUUID(),
          operation,
          parents: [
            ...inputAssets.map(publicAsset),
            ...(maskAsset ? [publicAsset(maskAsset)] : []),
          ],
          parentManifests,
        },
      },
      null,
      2,
    )}\n`,
  );

  process.stdout.write(
    `${JSON.stringify({ files, manifest: displayPath(manifestFile) }, null, 2)}\n`,
  );
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
