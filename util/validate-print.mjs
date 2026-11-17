// Build first, then run: node util/validate-print.mjs [directory-for-pdfs]
// Set CHROME_PATH to use a specific Chromium-based browser.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const root = path.resolve("_site");
const output = process.argv[2] && path.resolve(process.argv[2]);
const executablePath = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
].find((candidate) => candidate && fs.existsSync(candidate));
assert(executablePath, "Set CHROME_PATH to a local Chromium-based browser.");
assert(
  fs.existsSync(path.join(root, "index.html")),
  "Run npm run build first.",
);
if (output) fs.mkdirSync(output, { recursive: true });

const routes = [
  "/posts/2026-10-off-policy-rl/",
  "/posts/2025-02-selective-log-softmax/",
  "/posts/2025-01-badge-extension/",
  "/translations/scientific-spaces/transformer-upgrade-2-rotary-position-embedding/",
];
const mimeTypes = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".woff2": "font/woff2",
};

const browser = await puppeteer.launch({ executablePath, headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  const overflowFailures = [];
  let mediaAssets;
  function articleContent() {
    const copy = document.querySelector("main > article").cloneNode(true);
    // Existing scroll/resize effects update these independently of printing.
    copy
      .querySelectorAll("nav, svg.rough-annotation")
      .forEach((element) => element.remove());
    return copy.outerHTML;
  }
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.hostname !== "print.test") {
      if (
        url.hostname.includes("goatcounter") ||
        url.hostname === "gc.zgo.at"
      ) {
        return request.abort();
      }
      return request.continue();
    }
    const file = path.resolve(
      root,
      "." + decodeURIComponent(url.pathname),
      url.pathname.endsWith("/") ? "index.html" : "",
    );
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) {
      errors.push("Missing local asset: " + url.pathname);
      return request.respond({ status: 404, body: "Not found" });
    }
    return request.respond({
      contentType: mimeTypes[path.extname(file)] || "application/octet-stream",
      body: fs.readFileSync(file),
    });
  });

  for (const route of routes) {
    await page.emulateMediaType("screen");
    await page.setViewport({ width: 1440, height: 1000 });
    await page.goto("http://print.test" + route, { waitUntil: "networkidle0" });
    await page.evaluate(() => document.fonts.ready);
    const original = await page.evaluate(articleContent);
    if (route === routes[0]) {
      mediaAssets = await page.$$eval("main > article img", (images) => ({
        animation: images
          .find((img) => /\.gif(?:$|\?)/i.test(img.src))
          .getAttribute("src"),
        poster: images
          .find((img) => !/\.gif(?:$|\?)/i.test(img.src))
          .getAttribute("src"),
      }));
    }
    if (output && route.includes("selective-log-softmax")) {
      await page.screenshot({ path: path.join(output, "screen-desktop.png") });
      await page.setViewport({ width: 390, height: 844 });
      await page.screenshot({ path: path.join(output, "screen-mobile.png") });
    }

    // Exercise the actual shortcut, including images below the initial viewport.
    await page.evaluate(() => {
      window.originalPrint = window.print;
      window.print = () => {
        window.shortcutReady = Array.from(
          document.querySelectorAll(".print-article img"),
        ).every((image) => image.complete && image.naturalWidth);
      };
    });
    for (const modifier of ["Control", "Meta"]) {
      await page.keyboard.down(modifier);
      await page.keyboard.press("p");
      await page.keyboard.up(modifier);
      await page.waitForFunction(() => window.shortcutReady === true);
      await page.evaluate(() => {
        window.shortcutReady = false;
        window.dispatchEvent(new Event("afterprint"));
      });
    }
    await page.evaluate(() => {
      window.print = window.originalPrint;
      delete window.originalPrint;
      delete window.shortcutReady;
    });

    // Use the browser's own print events now that the shortcut has loaded assets.
    await page.emulateMediaType(null);
    const nativePdf = await page.pdf({ format: "A4", printBackground: false });
    assert(
      nativePdf.byteLength > 10000,
      "The native print lifecycle should produce a document.",
    );
    assert.equal(
      await page.$(".print-article"),
      null,
      "Native afterprint must clean up.",
    );
    if (output)
      fs.writeFileSync(
        path.join(
          output,
          route.split("/").filter(Boolean).at(-1) + "-native.pdf",
        ),
        nativePdf,
      );

    for (const [format, paperWidth] of [
      ["A4", 210],
      ["Letter", 215.9],
    ]) {
      await page.setViewport({
        width: Math.floor(((paperWidth - 36) * 96) / 25.4),
        height: 1000,
      });
      await page.emulateMediaType("print");
      await page.evaluate(() => {
        window.dispatchEvent(new Event("beforeprint"));
        window.dispatchEvent(new Event("beforeprint"));
      });
      await page.evaluate(async () => {
        await document.fonts.ready;
        await Promise.all(
          Array.from(document.querySelectorAll(".print-article img"), (image) =>
            image.decode(),
          ),
        );
        window.dispatchEvent(new Event("beforeprint"));
      });
      const state = await page.evaluate(() => {
        const article = document.querySelector(".print-article");
        const box = article.getBoundingClientRect();
        const overflows = Array.from(
          article.querySelectorAll(
            "p, pre > code, table, img, .katex-display .katex-html, .katex-display .tag",
          ),
        )
          .filter((element) => {
            const rect = element.getBoundingClientRect();
            return (
              rect.width &&
              (rect.left < box.left - 2 || rect.right > box.right + 2)
            );
          })
          .map((element) => element.outerHTML.slice(0, 180));
        const ids = Array.from(document.querySelectorAll("[id]"))
          .filter((element) => !element.closest("svg"))
          .map((element) => element.id);
        const brokenReferences = Array.from(
          article.querySelectorAll(".print-note-reference a"),
        ).filter((link) => !document.getElementById(link.hash.slice(1))).length;
        const noteMarkers = Array.from(
          article.querySelectorAll(".print-note-reference a"),
        );
        const misplacedNotes = noteMarkers.filter((marker) => {
          const item = document.getElementById(marker.hash.slice(1));
          const group = item?.closest(".print-notes");
          const section = marker.closest("section") || article;
          if (group?.parentElement !== section) return true;
          let block = marker;
          while (block.parentElement !== section) block = block.parentElement;
          for (
            let next = block.nextElementSibling;
            next;
            next = next.nextElementSibling
          ) {
            if (next === group) return false;
            if (next.matches("h1, h2, h3, h4, h5, h6, section")) return true;
          }
          return true;
        }).length;
        const misnumberedNotes = noteMarkers.filter((marker) => {
          const item = document.getElementById(marker.hash.slice(1));
          return item?.value !== Number(marker.textContent);
        }).length;
        const overlappingTags = Array.from(
          article.querySelectorAll(".katex-display .tag"),
        ).filter((tag) => {
          const formula = tag.parentElement.getBoundingClientRect();
          const number = tag.getBoundingClientRect();
          return number.left < formula.right && number.right > formula.left;
        }).length;
        return {
          copies: document.querySelectorAll(".print-article").length,
          originalHidden:
            getComputedStyle(document.querySelector("main > article"))
              .display === "none",
          metadata: article.querySelector(".print-metadata")?.textContent,
          notes: article.querySelectorAll(".print-notes > ol > li").length,
          noteMarkers: article.querySelectorAll(".print-note-reference").length,
          misplacedNotes,
          misnumberedNotes,
          generatedLinks: article.querySelectorAll(
            ".print-references, .print-link-reference",
          ).length,
          nestedParagraphs: article.querySelectorAll("p p").length,
          undersizedTechnicalText: Array.from(
            article.querySelectorAll("pre > code, th, td"),
          ).filter(
            (element) => parseFloat(getComputedStyle(element).fontSize) < 12,
          ).length,
          unkeptCodeIntroductions: Array.from(
            article.querySelectorAll("p:has(+ pre)"),
          ).filter(
            (paragraph) =>
              getComputedStyle(paragraph).breakAfter !== "avoid-page",
          ).length,
          remainingNotes: article.querySelectorAll(
            ".sidenote, .marginnote, .margin-toggle",
          ).length,
          duplicateIds: ids.length - new Set(ids).size,
          brokenReferences,
          overlappingTags,
          overflows,
        };
      });
      assert.equal(
        state.copies,
        1,
        "Repeated print events must not duplicate content.",
      );
      assert(state.originalHidden, "Only the print copy should render.");
      assert(
        state.metadata.includes("www.tylerromero.com"),
        "Print metadata is missing.",
      );
      assert.equal(
        state.notes,
        state.noteMarkers,
        "Every section note needs a reference.",
      );
      assert.equal(
        state.misplacedNotes,
        0,
        "Notes must stay before the next section or subsection.",
      );
      assert.equal(
        state.misnumberedNotes,
        0,
        "Note numbers must match markers.",
      );
      assert.equal(
        state.generatedLinks,
        0,
        "Do not add a Links appendix or numbered link markers.",
      );
      assert.equal(
        state.nestedParagraphs,
        0,
        "Generated media fallbacks must not nest paragraphs.",
      );
      assert.equal(
        state.undersizedTechnicalText,
        0,
        "Code listings and table cells must use at least 9pt type.",
      );
      assert.equal(
        state.unkeptCodeIntroductions,
        0,
        "Code lead-ins must stay with their listings.",
      );
      assert.equal(
        state.remainingNotes,
        0,
        "Margin content must be placed in the print flow.",
      );
      assert.equal(
        state.duplicateIds,
        0,
        "Print copies must use distinct IDs.",
      );
      assert.equal(
        state.brokenReferences,
        0,
        "Print references must target their entries.",
      );
      assert.equal(
        state.overlappingTags,
        0,
        "Equation numbers must not overlap formulas.",
      );
      if (state.overflows.length)
        overflowFailures.push({ route, format, elements: state.overflows });

      const filename =
        route.split("/").filter(Boolean).at(-1) + "-" + format + ".pdf";
      await page.pdf({
        format,
        preferCSSPageSize: true,
        printBackground: false,
        ...(output && { path: path.join(output, filename) }),
      });
      await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
      assert.equal(
        await page.$(".print-article"),
        null,
        "Closing print must remove the copy.",
      );
      await page.emulateMediaType("screen");
      const restored = await page.evaluate(articleContent);
      if (restored !== original) {
        let index = 0;
        while (restored[index] === original[index]) index++;
        errors.push({
          route,
          format,
          message: "Screen article changed",
          before: original.slice(index - 60, index + 120),
          after: restored.slice(index - 60, index + 120),
        });
      }
      console.log(
        `${route} ${format}: ${state.notes} notes; ${state.overflows.length} overflows; references and cleanup passed`,
      );
    }
  }

  await page.emulateMediaType("print");
  const media = await page.evaluate(async ({ animation, poster }) => {
    window.dispatchEvent(new Event("afterprint"));
    const source = document.querySelector("main > article");
    const alt = "A detailed description of the animation. ".repeat(12);
    source.innerHTML = `
      <p><picture><img src="${animation}" alt="${alt}"></picture><span data-caption>Authored caption.</span></p>
      <p><video title="Recorded example" poster="${poster}"></video></p>
      <p><iframe title="Interactive example" src="about:blank"></iframe></p>`;
    const original = source.innerHTML;
    window.dispatchEvent(new Event("beforeprint"));
    const article = document.querySelector(".print-article");
    await Promise.all(
      Array.from(article.querySelectorAll("img"), (img) => img.decode()),
    );
    const description = article.querySelector("picture").nextElementSibling;
    const state = {
      altPreserved: article.querySelector("picture img").alt === alt,
      caption: article.querySelector("[data-caption]").textContent,
      noticeLength: description.textContent.length,
      animationUrl: description.querySelector("a")?.href,
      nestedParagraphs: article.querySelectorAll("p p").length,
      notices: Array.from(
        article.querySelectorAll(".print-media-description"),
        (notice) => ({
          display: getComputedStyle(notice).display,
          fontSize: getComputedStyle(notice).fontSize,
        }),
      ),
    };
    window.dispatchEvent(new Event("afterprint"));
    return { ...state, sourcePreserved: source.innerHTML === original };
  }, mediaAssets);
  assert(media.altPreserved && media.sourcePreserved);
  assert.equal(media.caption, "Authored caption.");
  assert(media.noticeLength < 100, "Do not repeat long alt text as a caption.");
  assert.equal(
    media.animationUrl,
    new URL(mediaAssets.animation, "https://www.tylerromero.com").href,
  );
  assert.equal(media.nestedParagraphs, 0);
  assert.deepEqual(
    media.notices,
    Array(3).fill({ display: "block", fontSize: "12px" }),
  );
  console.log("Compact media notices preserve alt text and authored captions.");

  await page.evaluate(() => {
    document.querySelector("main > article").innerHTML = `
      <section id="first">
        <h2 id="encoded%2C-title">First section</h2>
        <p>First<span class="sidenote">First note.</span></p>
        <section id="nested">
          <h3 id="decoded,title">Nested section</h3>
          <p>Second<span class="sidenote">Second note.</span></p>
        </section>
        <p>Third<span class="sidenote">Third note.</span></p>
      </section>
      <section id="second">
        <h2>Second section</h2>
        <p>Fourth<span class="marginnote">Fourth note.</span></p>
        <h3 id="following-topic">Following topic without notes</h3>
        <p><a href="https://example.com/cited">Cited paper</a>
          <a href="/relative-citation">Relative citation</a>
          <a href="https://example.com/other">Other source</a>
          <a href="https://example.com/other">Same source again</a>
          <a href="#first">Return to first section</a>
          <a href="#encoded%2C-title">Encoded heading</a>
          <a href="#decoded%2Ctitle">Escaped heading</a></p>
      </section>
      <section id="without-notes">
        <h2>Code section without notes</h2>
        <p>In pseudocode:</p>
        <pre><code>print("Hello")</code></pre>
      </section>
      <section>
        <h2>References</h2>
        <div class="csl-bib-body">
          <p><a href="https://example.com/cited">https://example.com/cited</a></p>
          <p><a href="https://www.tylerromero.com/relative-citation">https://www.tylerromero.com/relative-citation</a></p>
        </div>
      </section>`;
  });
  await page.emulateMediaType("print");
  await page.evaluate(() => window.dispatchEvent(new Event("beforeprint")));
  const groupedNotes = await page.evaluate(() => {
    const article = document.querySelector(".print-article");
    const groups = Array.from(article.querySelectorAll(".print-notes"));
    return {
      groups: groups.map((group) => ({
        owner: group.parentElement.id,
        heading: group.firstElementChild.tagName,
        numbers: Array.from(
          group.querySelectorAll(":scope > ol > li"),
          (li) => li.value,
        ),
      })),
      emptyNotes: !!article.querySelector("#print-without-notes .print-notes"),
      notesBeforeSubheading: article
        .querySelector("#print-following-topic")
        .previousElementSibling.classList.contains("print-notes"),
      linkTargets: Array.from(article.querySelectorAll("p > a"), (a) =>
        a.getAttribute("href"),
      ),
      generatedLinks: article.querySelectorAll(
        ".print-references, .print-link-reference",
      ).length,
      appendedUrl: getComputedStyle(article.querySelector("p > a"), "::after")
        .content,
    };
  });
  assert.deepEqual(groupedNotes.groups, [
    { owner: "print-first", heading: "H3", numbers: [1] },
    { owner: "print-nested", heading: "H4", numbers: [2] },
    { owner: "print-first", heading: "H3", numbers: [3] },
    { owner: "print-second", heading: "H3", numbers: [4] },
  ]);
  assert(groupedNotes.notesBeforeSubheading);
  assert.equal(
    groupedNotes.emptyNotes,
    false,
    "Do not add empty Notes sections.",
  );
  assert.deepEqual(
    groupedNotes.linkTargets,
    [
      "https://example.com/cited",
      "https://www.tylerromero.com/relative-citation",
      "https://example.com/other",
      "https://example.com/other",
      "#print-first",
      "#print-encoded%2C-title",
      "#print-decoded,title",
      "https://example.com/cited",
      "https://www.tylerromero.com/relative-citation",
    ],
    "Keep hyperlinks and resolve relative and internal targets for PDFs.",
  );
  assert.equal(groupedNotes.generatedLinks, 0);
  assert(["none", "normal"].includes(groupedNotes.appendedUrl));
  if (output) {
    for (const format of ["A4", "Letter"]) {
      await page.evaluate(() => window.dispatchEvent(new Event("beforeprint")));
      await page.pdf({
        format,
        printBackground: false,
        path: path.join(output, `hyperlinks-${format}.pdf`),
      });
    }
  }
  await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
  console.log(
    "Section-local notes, preserved hyperlinks, and appendix removal passed.",
  );

  const image =
    `<img alt="Figure pagination fixture" style="width:80mm;height:60mm" src="data:image/svg+xml,` +
    encodeURIComponent(
      '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240"><rect width="320" height="240" fill="#ddd"/></svg>',
    ) +
    '">';
  for (const [name, intro, figure] of [
    [
      "figure",
      'FINTRO:<span class="sidenote">A note about the figure.</span>',
      `<figure>${image}<figcaption style="font-family:monospace">FCAPTION</figcaption></figure>`,
    ],
    [
      "picture",
      "PINTRO. The diagram below shows the result.",
      `<p style="font-family:monospace"><a href="https://example.com/figure"><picture>${image}</picture></a>PCAPTION</p>`,
    ],
    [
      "unrelated",
      "UINTRO. This paragraph discusses a different topic.",
      `<figure>${image}<figcaption style="font-family:monospace">UCAPTION</figcaption></figure>`,
    ],
    [
      "long",
      Array.from(
        { length: 60 },
        (_, i) => `LCONTEXT${String(i).padStart(3, "0")}`,
      )
        .concat("LINTRO:")
        .join("<br>"),
      `<figure>${image}<figcaption style="font-family:monospace">LCAPTION</figcaption></figure>`,
    ],
    [
      "margin",
      `MINTRO. A related illustration<span class="marginnote">${image}<span style="font-family:monospace">MCAPTION</span></span>.`,
      "",
    ],
  ]) {
    const keepLead = name !== "unrelated" && name !== "margin";
    const state = await page.evaluate(
      async ({ intro, figure, spacer }) => {
        document.querySelector("main > article").innerHTML =
          (spacer ? `<div style="height:${spacer}mm"></div>` : "") +
          `<p id="figure-intro" style="font-family:monospace">${intro}</p>` +
          figure;
        window.dispatchEvent(new Event("beforeprint"));
        const article = document.querySelector(".print-article");
        await Promise.all(
          Array.from(article.querySelectorAll("img"), (img) => img.decode()),
        );
        const lead = article.querySelector("#print-figure-intro");
        const imageBlock = article.querySelector(
          "figure, .print-margin-figure, p:has(picture, img)",
        );
        return {
          keepLead: lead.classList.contains("print-figure-lead"),
          leadBreakAfter: getComputedStyle(lead).breakAfter,
          imageBreakInside: getComputedStyle(imageBlock).breakInside,
        };
      },
      { intro, figure, spacer: name === "long" ? 0 : 205 },
    );
    assert.equal(state.keepLead, keepLead, name);
    assert.equal(state.leadBreakAfter, keepLead ? "avoid-page" : "auto", name);
    assert.equal(state.imageBreakInside, "avoid-page", name);
    if (output) {
      for (const format of ["A4", "Letter"]) {
        await page.evaluate(async () => {
          window.dispatchEvent(new Event("beforeprint"));
          await Promise.all(
            Array.from(document.querySelectorAll(".print-article img"), (img) =>
              img.decode(),
            ),
          );
        });
        await page.pdf({
          format,
          printBackground: false,
          path: path.join(output, `figure-lead-in-${name}-${format}.pdf`),
        });
      }
    }
    await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
  }
  console.log(
    "Figure introductions, unrelated paragraphs, and image captions passed.",
  );

  if (output) {
    for (const [name, prefix, count, spacer] of [
      ["short", "S", 12, 205],
      ["long", "L", 120, 0],
    ]) {
      await page.evaluate(
        ({ prefix, count, spacer }) => {
          const lines = Array.from(
            { length: count },
            (_, i) => prefix + String(i).padStart(3, "0"),
          );
          document.querySelector("main > article").innerHTML =
            (spacer ? `<div style="height:${spacer}mm"></div>` : "") +
            `<p style="font-family:monospace">${prefix}INTRO</p>` +
            `<pre><code>${lines.join("\n")}</code></pre>`;
          window.dispatchEvent(new Event("beforeprint"));
        },
        { prefix, count, spacer },
      );
      for (const format of ["A4", "Letter"]) {
        await page.pdf({
          format,
          printBackground: false,
          path: path.join(output, `code-lead-in-${name}-${format}.pdf`),
        });
      }
      await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
    }
  }
  assert.deepEqual(errors, [], "Browser errors occurred.");
  assert.deepEqual(
    overflowFailures,
    [],
    "Printed content extends beyond the page.",
  );
} finally {
  await browser.close();
}
