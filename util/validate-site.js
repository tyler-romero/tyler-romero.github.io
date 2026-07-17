import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const outputDir = path.resolve("_site");
const failures = new Set();

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

    const target = outputPathForUrl(href, file);
    if (!fs.existsSync(target)) {
      fail(file, `broken internal link: ${href}`);
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
  }
}

function validateToc(file, html) {
  const toc = html.match(
    /<nav class="toc" aria-label="Table of contents">([\s\S]*?)<\/nav>/i,
  );
  if (!toc) return;

  const headingCount = (html.match(/<h[23]\b/gi) || []).length;
  if (headingCount >= 3 && !/<a\s+href="#.+?">/i.test(toc[1])) {
    fail(file, `TOC is empty despite ${headingCount} section headings`);
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
  validateJsonLd(file, html);
  validateAccessibilityBasics(file, html);
  validateToc(file, html);
}

const sitemapPath = path.join(outputDir, "sitemap.xml");
const sitemap = fs.readFileSync(sitemapPath, "utf8");
if (sitemap.includes("/404.html")) fail(sitemapPath, "includes the 404 page");
if (sitemap.includes("/feed.xml")) fail(sitemapPath, "includes the Atom feed");

if (failures.size > 0) {
  console.error(`Site validation failed with ${failures.size} issue(s):`);
  for (const failure of [...failures].sort()) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Validated ${htmlFiles.length} HTML pages with no issues.`);
