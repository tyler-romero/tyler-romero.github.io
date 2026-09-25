// Render Excalidraw scenes to transparent PNGs using Excalidraw's own exporter in a headless browser.
//
// Usage:
//   node util/render-excalidraw.mjs [file.excalidraw ...]
//
// With no arguments, renders every src/assets/img/*.excalidraw to a PNG with the same basename.
// Output scale comes from the scene's appState.exportScale (default 2).
// Set CHROME_PATH to use a specific Chromium-based browser.
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const EXCALIDRAW_URL =
  "https://esm.sh/@excalidraw/excalidraw@0.18.1?deps=react@19.1.0,react-dom@19.1.0";
const DEFAULT_SCALE = 2;
const PADDING = 24;
const IMG_DIR = "src/assets/img";

const BROWSER_CANDIDATES = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];

function findBrowser() {
  const found = BROWSER_CANDIDATES.find((p) => p && fs.existsSync(p));
  if (!found) {
    throw new Error("No Chromium-based browser found. Set CHROME_PATH to one.");
  }
  return found;
}

const inputs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : fs
      .readdirSync(IMG_DIR)
      .filter((f) => f.endsWith(".excalidraw"))
      .map((f) => path.join(IMG_DIR, f));

const browser = await puppeteer.launch({ executablePath: findBrowser(), headless: true });
try {
  const page = await browser.newPage();
  await page.setContent("<!doctype html><html><body></body></html>");

  for (const input of inputs) {
    const scene = JSON.parse(fs.readFileSync(input, "utf8"));
    const dataUrl = await page.evaluate(
      async (url, scene, scale, padding) => {
        const { exportToBlob } = await import(url);
        const blob = await exportToBlob({
          elements: scene.elements,
          appState: { ...scene.appState, exportBackground: false },
          files: scene.files ?? {},
          exportPadding: padding,
          mimeType: "image/png",
          getDimensions: (w, h) => ({ width: w * scale, height: h * scale, scale }),
        });
        return await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        });
      },
      EXCALIDRAW_URL,
      scene,
      scene.appState?.exportScale ?? DEFAULT_SCALE,
      PADDING,
    );

    const output = input.replace(/\.excalidraw$/, ".png");
    fs.writeFileSync(output, Buffer.from(dataUrl.split(",")[1], "base64"));
    console.log(`${input} -> ${output}`);
  }
} finally {
  await browser.close();
}
