# Visual Style Guide: Approachable Intelligence

## Identity

**tylerromero.com** is a personal blog about AI/ML. The governing metaphor is **hand-drawn notes on a beautiful academic manuscript or journal** — the kind of paper that's been typeset with care, then annotated in the margins with a warm pen. The typography is serious and well-set. The annotations, dividers, and decorative elements are sketchy, human, and slightly imperfect.

Two layers work together:

1. **The manuscript**: Clean serif typography, generous whitespace, Tufte-style sidenotes and margin notes. This is the foundation — the part that says "the content here is rigorous and worth your time."
2. **The hand-drawn layer**: Wavy dividers, sketchy brackets on headings, a handwriting font (Virgil), organic SVG borders on code blocks, warm accent colors. This is what makes it feel personal — like someone actually lives in this document.

The visual language draws from editorial design and well-set books — never from sci-fi, cyberpunk, or tech-bro aesthetics. If it could appear in a thoughtful essay collection, it belongs here. If it looks like a startup landing page or a conference slide deck, it doesn't.

---

## How to Use This Guide

The descriptive sections explain how the site should feel. Tables, measurements, and checklists define the default implementation. Preserve the spirit of the guide when a new format does not fit an existing rule exactly, but do not use aesthetic intent to override readability, accessibility, or factual clarity.

- Use the existing design tokens and component patterns before introducing new ones.
- Treat `src/assets/tufte.css` as the implementation source of truth. When a deliberate design-system change alters a token or component rule, update this guide and the CSS together.
- Prefer the simplest presentation that serves the content. Decoration should clarify hierarchy or add human character, not fill empty space.
- Technical posts default to typography-only headers. Images require an editorial reason beyond making the page feel more complete.

---

## Color Philosophy

### Palette

The manuscript layer uses warm neutrals — cream paper, soft charcoal ink. The hand-drawn layer introduces color from nature: golden hour skies, temperate forests, wildflower meadows. Colors should feel like places you've been, not places that exist only on screens.

| Role                 | Tones                                       | Notes                                       |
| -------------------- | ------------------------------------------- | ------------------------------------------- |
| **Primary warmth**   | Terracotta, salmon, dusty rose, amber       | Used for focal points and emotional anchors |
| **Grounding greens** | Sage, olive, moss, muted emerald            | Conveys growth, calm, and the organic       |
| **Sky and water**    | Slate blue, periwinkle, soft teal, lavender | Creates depth, openness, and breathing room |
| **Neutrals**         | Warm cream, stone, soft charcoal            | Page backgrounds and typography             |

### Canonical Tokens

Use these colors for interface and editorial components. Illustration and photography may use a broader natural palette, but should harmonize with these anchors.

| Token           | Value     | Primary use                                   |
| --------------- | --------- | --------------------------------------------- |
| `--bg-color`    | `#FEF9ED` | Page background and paper-colored fades       |
| `--text-color`  | `#1A1A1A` | Body copy and primary headings                |
| `--link-color`  | `#568259` | Links and navigational actions                |
| `--brand-color` | `#6B7FA3` | Site identity and selected structural accents |
| `--warm-accent` | `#B07A65` | Dates, annotations, and small focal details   |
| `--warm-glow`   | `#C49A6C` | Dividers, borders, and atmospheric highlights |
| `--code-bg`     | `#F3EFE4` | Inline and block code backgrounds             |

### Rules

- Never use neon, electric blue, or saturated cyan. These read as "tech company default."
- Black is used sparingly and always slightly warm (e.g., `#1a1a1a` not `#000000`).
- Gradients are allowed but should feel atmospheric — like light through clouds, not like a UI button.
- Do not introduce a new interface color when an existing token can express the same hierarchy.
- Color must never be the only signal for state, meaning, or interactivity.
- When in doubt, reference the light at 6:45pm on a clear day in late September.

---

## Illustration Style

### General Direction

When a post calls for an illustration — usually a conceptual diagram or occasional ambient figure — it should be **painterly, stylized, and slightly abstracted**. Think gouache, soft pastel, or risograph. It should feel deliberately made rather than mechanically polished.

### Do

- Use flat-to-semi-flat depth with soft, implied shadows rather than hard lighting.
- Allow forms to dissolve at edges. Not everything needs to be sharply defined.
- Favor landscapes, architecture, gardens, and natural subjects.
- Use visible texture — paper grain, brush strokes, ink bleed, slight imperfections.
- Compose with generous negative space and clear focal hierarchy.

### Don't

- Don't attempt photorealistic rendering in illustrations.
- Don't use visual clichés for AI: no neural networks, no glowing brains, no circuit boards, no floating holographic interfaces.
- Don't use pure geometric abstraction. Every piece should contain at least one recognizable, grounded element (a tree, a building, water, sky).
- Don't publish telltale generated-image artifacts such as melted details, impossible geometry, or texture soup. Reject, redraw, or retouch the illustration instead.

### Typical Uses

- **Post header images**: Not the default. Use one only when an original photograph or deliberately made illustration contributes editorial meaning. A technical post should otherwise use a typography-only header.
- **Ambient figures**: Close crops of flowers, leaves, water, or sky, used sparingly to pace an unusually long piece rather than decorate routine sections.
- **Conceptual diagrams**: When a technical diagram is needed, lean into the hand-drawn layer — Virgil font for labels, sketchy borders, organic lines. The diagram should look like something scribbled in the margin of a paper, not exported from a drawing tool.

---

## Photography

Photos on the site are personal — a headshot, a snapshot from a project, a photo from the kitchen. They should feel real and unstaged.

- Natural light strongly preferred. If artificial, it should be warm and diffused.
- Depth of field should be moderate — the subject is clear, but the environment is present and readable.
- Color grade warm but not orange. Shadows should stay open and slightly cool for contrast.
- No seamless studio backdrops, no stock photography, no AI-generated faces.
- Photos are presented as clean rectangles with minimal border-radius (3px desktop, 2px mobile) — never circular or heavily rounded. No drop shadows. A photo should sit flat on the page like a print pasted into a manuscript.

---

## Typography

Typography spans both layers: the manuscript carries the main text; the hand-drawn layer adds annotations and human touches.

### Manuscript Layer

The manuscript layer is properly typeset — like a paper you'd want to read in print.

- **Display titles**: Large serif headlines that feel literary and human. Italic styling may distinguish a subtitle or special display treatment from the roman section headers below.
- **Section Headers**: Same serif family as display (Fraunces), roman weight 300. The contrast between compact display titles, italic subtitles, and roman section headers creates hierarchy without introducing another typeface.
- **Body**: Newsreader — a warm, readable serif with humanist warmth at comfortable reading sizes with tall line-height (1.6–1.7). Sidenotes and margin notes use Gentium Book Plus.
- **Pull Quotes**: Set in a serif italic, often oversized, used to surface human voices and values.

### Type Scale

Sizes are expressed in `rem` so the site can scale its root size across viewport widths. Preserve the hierarchy even when a component needs a local adjustment.

| Role                       | Typeface            | Size / line-height | Notes                                                |
| -------------------------- | ------------------- | ------------------ | ---------------------------------------------------- |
| Page title (`h1`)          | Fraunces 300        | `3rem / 1`         | Keep titles compact and avoid forced line breaks     |
| Section heading (`h2`)     | Fraunces 300        | `2.4rem / 1`       | Primary article divisions                            |
| Subheading (`h3`)          | Fraunces 300        | `1.9rem / 1`       | Use only when it improves scanability                |
| Subtitle                   | Fraunces 300 italic | `2rem / 1`         | One concise sentence or phrase                       |
| Body and lists             | Newsreader          | `1.4rem / 1.6`     | Default long-form reading text                       |
| Sidenotes and margin notes | Gentium Book Plus   | `1.1rem / 1.3`     | Never carry the primary argument only in the margin  |
| Figure captions            | Newsreader          | `1.1rem / 1.6`     | Describe the visual's subject or takeaway concisely  |
| Code blocks                | JetBrains Mono      | `0.9rem / 1.42`    | Allow horizontal scrolling rather than wrapping code |

**Rules**: No all-caps except for small UI labels (tags like "research," "announcements"). Monospaced fonts are reserved for code blocks only — never in editorial or decorative contexts. Generous whitespace between sections. Let every element breathe.

### Hand-Drawn Layer

The Virgil font (a handwriting face) is the typographic voice of the annotation layer. It should appear only in the "notes" layer, never in body text or headings.

- **Dates**: Set in Virgil with abbreviated months (e.g., "Jan 10, 2026").
- **Diagram labels**: Virgil for labels on conceptual diagrams — like something scribbled in the margin.
- **Decorative elements**: The margin-toggle icon on mobile is a Virgil asterisk (`*`) in superscript, rendered in the warm accent color.

---

## Layout Principles

### Layout Reference

| Property                     | Default                                       |
| ---------------------------- | --------------------------------------------- |
| Maximum page width           | `1400px`                                      |
| Desktop article column       | `55%` of the page body                        |
| Desktop margin-note region   | Approximately `40%` beside the article column |
| Full-width editorial content | No more than `90%` of the page body           |
| Primary mobile breakpoint    | `760px`                                       |
| Image corner radius          | `3px` desktop, `2px` mobile                   |

Preferred spacing steps are `4`, `8`, `16`, `24`, `40`, `48`, and `80px`. Use the smaller values within components and the larger values to separate ideas. Avoid one-off spacing values unless optical alignment clearly requires them.

### Pacing

The site should feel like **reading an annotated manuscript**, not scrolling through a feed. The Tufte CSS foundation — a 55% content column with margin notes — mirrors the layout of an academic paper with a wide margin for the reader's notes.

- The main column carries the argument. Sidenotes, margin notes, and figures populate the margins — just as a reader's annotations would.
- Hand-drawn wavy dividers separate major sections, like a pen stroke between chapters.
- Code blocks have organic, sketchy left borders — they're part of the manuscript, not pasted in from somewhere else.
- Blockquotes and epigraphs should feel like they're floating in air — generous vertical margin, warm left border.

### White Space

White space is a first-class design element, not leftover. Use approximately `80px` between major sections on wide screens and `48px` on small screens. A well-set paper has generous margins for a reason — the page should have the quiet confidence of a manuscript that trusts its typography and leaves room for the reader to think.

### Responsive Behavior

- At `760px` and below, the main reading column expands to the available width and margin content moves into the reading flow.
- Sidenotes become keyboard-accessible inline disclosures. The surrounding sentence must remain understandable while a note is collapsed.
- Figures expand toward the content width; captions move below the figure rather than remaining in the margin.
- Tables and code blocks may scroll horizontally inside their own containers. They must never cause the whole page to scroll sideways.
- Preserve image aspect ratios. Do not crop technical figures in ways that remove labels, legends, or data.
- Avoid hiding substantive content on mobile. Responsive changes may alter placement and interaction, not meaning.
- Check long titles, equations, URLs, tables, code, and sidenotes at narrow widths; these are the most common sources of overflow.

---

## Motion and Interaction

Animations belong to the hand-drawn layer. The manuscript itself is static; the annotations appear as if someone is marking up the page as you read.

- Rough Notation brackets and underlines animate in as the reader scrolls — like a pen drawing them in real time. These use the warm accent color (`--warm-accent`), not the link green.
- Use `200ms ease` for hover and state transitions. Hand-drawn entrance annotations may run up to `600ms`. Avoid spring, bounce, or elastic easing.
- Hover states are subtle (slight background shift, soft shadow deepening). No color shifts.
- The overall feeling should be **calm confidence**, never urgency.
- Motion is decorative. No information or required action may depend on an animation completing.

### Navigation

The header slides out of view as the reader scrolls down and slides back in when scrolling up — tracking 1:1 with scroll delta, not snapping. This balances editorial pacing (the nav doesn't permanently occupy the page) with usability (it's always one scroll-up away). At the top of the page, the header is always fully visible.

### Reduced Motion

Respect `prefers-reduced-motion: reduce`. Disable smooth scrolling and decorative drawing animations, and make state changes immediate. The reading order, navigation, and visibility of content must remain unchanged.

---

## Accessibility

Accessibility is part of the manuscript's quality, not a separate visual mode.

- Text and essential interface elements must meet WCAG AA contrast: `4.5:1` for normal text and `3:1` for large text and meaningful graphical controls.
- Links in body copy remain underlined. Do not rely on green alone to distinguish them from surrounding text.
- Every interactive element needs a visible `:focus-visible` treatment with at least a `2px` outline and clear separation from the element.
- Interactive targets should be at least `44 × 44px` on touch layouts, even when the visible mark is smaller.
- Write useful alt text for informative images. Use empty alt text for purely decorative images, and do not repeat a nearby caption verbatim.
- Use headings in a logical sequence. Do not select a heading level for its visual size.
- Tables need header cells and a clear reading order. Introduce complex tables in the surrounding prose.
- Captions, labels, and annotations must remain legible at 200% zoom.
- Never encode meaning through color, position, or motion alone.
- Print output should preserve the article, citations, figures, and sidenotes while omitting navigation and decorative motion.

---

## Voice

The writing voice is:

- **Confident but not grandiose** — "here's what I found" not "this changes everything"
- **Educational and precise** — clear explanations that respect the reader's intelligence
- **Quietly ambitious** — the work speaks for itself; the words stay grounded

The blog is earnest. The visuals exist to make that earnestness feel credible rather than disposable.

### Technical Posts

- Lead with the question, result, or reason the work matters. Do not make readers traverse project history before learning what was found.
- Keep the title, subtitle, and blurb distinct: the title names the subject, the subtitle frames the specific claim, and the blurb summarizes the value for listings and metadata.
- Use descriptive headings that expose the argument when scanned. Number sections only when sequence or later cross-reference makes numbering useful.
- Keep paragraphs focused on one idea. Prefer a short transition sentence over a dense paragraph that changes subjects halfway through.
- Use descriptive link text rather than “here” or a bare URL. Place citations close to the claim they support and prefer primary sources for technical assertions.
- Introduce figures and tables in the prose. Give each one a useful caption and explain the takeaway rather than asking the visual to speak for itself.
- Explain why a code sample exists, what assumptions it makes, and what the reader should notice. Avoid unexplained code dumps.
- State uncertainty, limitations, failed experiments, and material implementation details near the relevant conclusion.
- Use typographic punctuation consistently: curly apostrophes, en dashes for ranges, and em dashes for interruptions.

---

## Footer / Colophon

The footer is a typographic colophon, not a row of social icons. It contains:

1. **Text links** in Fraunces small-caps (GitHub / LinkedIn / Scholar / Twitter / Email / RSS), separated by `/` dividers.
2. **A typesetting line** that names the fonts used — set self-referentially (typeface names in Newsreader italic, "Annotations in Virgil" in Virgil with warm accent color).
3. **Copyright** in Newsreader at reduced opacity.

The colophon is separated from the page content by a hand-drawn wavy divider. No SVG icons, no icon fonts.

---

## Recipe Box

The recipe box section has a different voice from the technical blog. Where blog posts are educational and precise, recipes are **conversational and warm**, in the spirit of Bon Appetit. Think of a friend telling you about a dish they love.

### Writing Style

- **Conversational tone**: Write like you're telling someone about a recipe over dinner, not writing documentation. Short, direct sentences. Personal opinions welcome.
- **Headnotes over instructions**: The editorial value is in the headnotes (why you love this recipe, what makes it work, tips and substitutions). Many recipes simply link out to the original source.
- **Honest and specific**: "No frying required, but they still come out incredibly well" is better than "a delicious treat the whole family will love." Say what you actually think.
- **Proper formatting**: Use correct typographic conventions (en dashes for ranges, curly apostrophes, °F with the degree symbol). Capitalize proper nouns (Trader Joe's, English muffin). Spell out ingredients fully (extra-virgin olive oil, not EVOO).

### Structure

- Each recipe page has a short, punchy subtitle and blurb that reads as a sentence fragment starting lowercase (the list page capitalizes the first letter).
- Recipes that link to an external source should use a descriptive link like `[Full recipe on sitename.com](url)`, not a bare URL.
- Personal notes and tips go above the recipe link or ingredients. Keep them brief.
- Yield and time go in italics at the top: `*Serves 8 — 35 minutes total*`

---

## Pre-Publish Checklist

### Content

- [ ] The title, subtitle, and blurb each serve a distinct purpose.
- [ ] The introduction establishes the question or result early.
- [ ] Headings form a clear outline and use logical levels.
- [ ] Technical claims have nearby links or citations where appropriate.
- [ ] Figures, tables, equations, and code are introduced and explained.

### Visual System

- [ ] Existing colors, type styles, spacing steps, and component patterns were reused.
- [ ] The post uses a typography-only header unless a hero image has a clear editorial purpose.
- [ ] Decorative elements support hierarchy and do not compete with the text.
- [ ] New assets are necessary, appropriately licensed, compressed, and referenced by the page.

### Responsive and Accessible Behavior

- [ ] The page was checked at desktop and narrow mobile widths without horizontal page overflow.
- [ ] Sidenotes, tables, code blocks, equations, and long links remain usable on mobile.
- [ ] Informative images have useful alt text; decorative images use empty alt text.
- [ ] Links, controls, and sidenote toggles work with a keyboard and have visible focus states.
- [ ] Color contrast meets WCAG AA and meaning does not depend on color or motion alone.
- [ ] Reduced-motion and print presentations preserve all substantive content.

### Verification

- [ ] `npm run check` passes.
- [ ] The built page has no broken internal links or missing assets.
- [ ] The final page was read once as an article, not only inspected as a layout.

---

## What This Is Not

This is not a developer portfolio template or a tech company blog. There are no dark-mode dashboards, no terminal-green-on-black hero sections, no "built with" badge walls, no Medium-clone layouts. It's also not a sterile LaTeX PDF — the manuscript is warm, not clinical.

The aesthetic goal is closer to **Monocle** than **Hacker News**, closer to a marked-up galley proof than a documentation site. If the site could be mistaken for a SaaS landing page, something has gone wrong. If it reads like a carefully typeset paper that someone has annotated with a warm pen — underlining key phrases, bracketing section headings, sketching dividers between ideas — this guide is working.
