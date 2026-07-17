import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const outputDir = path.resolve("_site");
const siteOrigin = "https://www.tylerromero.com";
const failures = new Set();
const checkedPassthroughUrls = new Set();

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(entryPath) : entryPath;
  });
}

function fail(file, message) {
  failures.add(`${path.relative(process.cwd(), file)}: ${message}`);
}

function attribute(tag, name) {
  const match = tag.match(new RegExp(`\\s${name}=(['"])(.*?)\\1`, "i"));
  return match?.[2];
}

function outputPathForUrl(url, sourceFile) {
  const sourceUrl = `https://local.test/${path
    .relative(outputDir, sourceFile)
    .split(path.sep)
    .join("/")}`;
  const pathname = decodeURIComponent(new URL(url, sourceUrl).pathname);
  const candidate = path.join(outputDir, pathname.replace(/^\//, ""));

  if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
    return candidate;
  }
  if (
    fs.existsSync(candidate) &&
    fs.statSync(candidate).isDirectory() &&
    fs.existsSync(path.join(candidate, "index.html"))
  ) {
    return path.join(candidate, "index.html");
  }
  if (
    pathname.endsWith("/") &&
    fs.existsSync(path.join(candidate, "index.html"))
  ) {
    return path.join(candidate, "index.html");
  }

  return candidate;
}

function isPassthroughFile(url) {
  const pathname = new URL(url, siteOrigin).pathname.toLowerCase();
  return pathname.endsWith(".ico") || pathname.endsWith(".webm");
}

function validatePassthroughUrl(file, url, description) {
  if (!url) {
    fail(file, `${description} URL is missing`);
    return;
  }

  const resolved = new URL(url, siteOrigin);
  if (resolved.origin !== siteOrigin) return;
  if (checkedPassthroughUrls.has(resolved.href)) return;
  checkedPassthroughUrls.add(resolved.href);

  const target = outputPathForUrl(resolved.pathname, file);
  if (!fs.existsSync(target)) {
    fail(file, `${description} is not passthrough copied: ${url}`);
  }
}

function validateInternalLinks(file, html) {
  for (const match of html.matchAll(
    /<(?:a|link)\b[^>]*\bhref=(['"])(.*?)\1/gi,
  )) {
    const href = match[2];
    if (
      !href ||
      href.startsWith("#") ||
      href.startsWith("//") ||
      /^(?:https?:|mailto:|tel:|javascript:|data:)/i.test(href)
    ) {
      continue;
    }
    if (isPassthroughFile(href)) continue;

    const target = outputPathForUrl(href, file);
    if (!fs.existsSync(target)) {
      fail(file, `broken internal link: ${href}`);
    }
  }
}

function validateInternalAssets(file, html) {
  const urls = [];
  for (const match of html.matchAll(
    /<(?:img|script|source|video)\b[^>]*\bsrc=(['"])(.*?)\1/gi,
  )) {
    urls.push(match[2]);
  }
  for (const match of html.matchAll(/\bsrcset=(['"])(.*?)\1/gi)) {
    for (const candidate of match[2].split(",")) {
      urls.push(candidate.trim().split(/\s+/)[0]);
    }
  }

  for (const url of urls) {
    if (!url || url.startsWith("//") || /^(?:https?:|data:)/i.test(url)) {
      continue;
    }
    if (isPassthroughFile(url)) continue;

    const target = outputPathForUrl(url, file);
    if (!fs.existsSync(target)) {
      fail(file, `missing local asset: ${url}`);
    }
  }
}

function validatePassthroughReferences(file, html) {
  for (const match of html.matchAll(/\b(?:href|src)=(['"])(.*?)\1/gi)) {
    const url = match[2];
    if (isPassthroughFile(url)) {
      validatePassthroughUrl(file, url, "ICO/WebM asset");
    }
  }

  for (const match of html.matchAll(/<meta\b[^>]*>/gi)) {
    const key = attribute(match[0], "property") || attribute(match[0], "name");
    if (key === "og:image" || key === "twitter:image") {
      validatePassthroughUrl(
        file,
        attribute(match[0], "content"),
        `${key} social image`,
      );
    }
  }
}

function validateJsonLd(file, html) {
  for (const match of html.matchAll(
    /<script type="application\/ld\+json">([\s\S]*?)<\/script>/gi,
  )) {
    try {
      const data = JSON.parse(match[1]);
      const serialized = JSON.stringify(data);
      if (serialized.includes('"undefined"')) {
        fail(file, "JSON-LD contains an undefined value");
      }
      if ("headline" in data && !data.headline) {
        fail(file, "JSON-LD headline is empty");
      }
      if (data.image === "https://www.tylerromero.com") {
        fail(file, "JSON-LD image points to the site root");
      }
      if (typeof data.image === "string") {
        validatePassthroughUrl(file, data.image, "JSON-LD social image");
      }
    } catch (error) {
      fail(file, `invalid JSON-LD: ${error.message}`);
    }
  }
}

function validateAccessibilityBasics(file, html) {
  if (!/<html\b[^>]*\blang=(['"])[^'"]+\1/i.test(html)) {
    fail(file, "missing document language");
  }

  const h1Count = (html.match(/<h1\b/gi) || []).length;
  if (h1Count !== 1) {
    fail(file, `expected exactly one h1, found ${h1Count}`);
  }

  for (const match of html.matchAll(/<img\b[^>]*>/gi)) {
    if (attribute(match[0], "alt") === undefined) {
      fail(file, "image is missing alt text");
    }
  }

  for (const match of html.matchAll(/<input\b[^>]*>/gi)) {
    const id = attribute(match[0], "id");
    if (
      id &&
      !new RegExp(`<label\\b[^>]*\\bfor=(['"])${id}\\1`, "i").test(html)
    ) {
      fail(file, `input #${id} has no associated label`);
    }
    if (
      match[0].includes('class="margin-toggle"') &&
      attribute(match[0], "aria-label") === undefined
    ) {
      fail(file, `margin toggle #${id} has no accessible name`);
    }
  }

  for (const match of html.matchAll(
    /(<ul\b[^>]*class=(['"])(?:post-list|item-list)\2[^>]*>)([\s\S]*?)<\/ul>/gi,
  )) {
    if (attribute(match[1], "role") !== "list") {
      fail(file, "styled list is missing role=list");
    }
    for (const item of match[3].matchAll(/<li\b[^>]*>/gi)) {
      if (attribute(item[0], "role") !== "listitem") {
        fail(file, "styled list entry is missing role=listitem");
      }
    }
  }
}

function validateConditionalAssets(file, html) {
  const hasAnnotations = html.includes("RoughNotation.annotate");
  const hasRoughNotation = html.includes(
    "rough-notation@0.5.1/lib/rough-notation.iife.js",
  );
  if (hasAnnotations !== hasRoughNotation) {
    fail(file, "Rough Notation script does not match page behavior");
  }
}

if (!fs.existsSync(outputDir)) {
  console.error("_site does not exist. Run npm run build first.");
  process.exit(1);
}

const htmlFiles = walk(outputDir).filter((file) => file.endsWith(".html"));
for (const file of htmlFiles) {
  const html = fs.readFileSync(file, "utf8");
  const title = html.match(/<title>([\s\S]*?)<\/title>/i)?.[1].trim();
  const description = html.match(
    /<meta\s+name="description"\s+content="([^"]*)"/i,
  )?.[1];

  if (!title || title === "undefined") fail(file, "missing page title");
  if (!description || description === "undefined") {
    fail(file, "missing meta description");
  }
  if (/content="undefined"/i.test(html)) {
    fail(file, "metadata contains undefined");
  }

  validateInternalLinks(file, html);
  validateInternalAssets(file, html);
  validatePassthroughReferences(file, html);
  validateJsonLd(file, html);
  validateAccessibilityBasics(file, html);
  validateConditionalAssets(file, html);
}

const sitemapPath = path.join(outputDir, "sitemap.xml");
const sitemap = fs.readFileSync(sitemapPath, "utf8");
if (sitemap.includes("/404.html")) fail(sitemapPath, "includes the 404 page");
if (sitemap.includes("/feed.xml")) fail(sitemapPath, "includes the Atom feed");

const manifestPath = path.join(outputDir, "manifest.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
for (const icon of manifest.icons || []) {
  validatePassthroughUrl(manifestPath, icon.src, "Manifest icon");
}

if (failures.size > 0) {
  console.error(`Site validation failed with ${failures.size} issue(s):`);
  for (const failure of [...failures].sort()) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Validated ${htmlFiles.length} HTML pages with no issues.`);
