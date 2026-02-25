# Visual Style Guide: Approachable Intelligence

## Identity

**tylerromero.com** is a personal blog about AI/ML. The governing metaphor is **hand-drawn notes on a beautiful academic manuscript or journal** — the kind of paper that's been typeset with care, then annotated in the margins with a warm pen. The typography is serious and well-set. The annotations, dividers, and decorative elements are sketchy, human, and slightly imperfect.

Two layers work together:
1. **The manuscript**: Clean serif typography, generous whitespace, Tufte-style sidenotes and margin notes. This is the foundation — the part that says "the content here is rigorous and worth your time."
2. **The hand-drawn layer**: Wavy dividers, sketchy brackets on headings, a handwriting font (Virgil), organic SVG borders on code blocks, warm accent colors. This is what makes it feel personal — like someone actually lives in this document.

The visual language draws from editorial design and well-set books — never from sci-fi, cyberpunk, or tech-bro aesthetics. If it could appear in a thoughtful essay collection, it belongs here. If it looks like a startup landing page or a conference slide deck, it doesn't.

---

## Color Philosophy

### Palette

The manuscript layer uses warm neutrals — cream paper, soft charcoal ink. The hand-drawn layer introduces color from nature: golden hour skies, temperate forests, wildflower meadows. Colors should feel like places you've been, not places that exist only on screens.

| Role | Tones | Notes |
|------|-------|-------|
| **Primary warmth** | Terracotta, salmon, dusty rose, amber | Used for focal points and emotional anchors |
| **Grounding greens** | Sage, olive, moss, muted emerald | Conveys growth, calm, and the organic |
| **Sky and water** | Slate blue, periwinkle, soft teal, lavender | Creates depth, openness, and breathing room |
| **Neutrals** | Warm cream, stone, soft charcoal | Page backgrounds and typography |

### Rules

- Never use neon, electric blue, or saturated cyan. These read as "tech company default."
- Black is used sparingly and always slightly warm (e.g., `#1a1a1a` not `#000000`).
- Gradients are allowed but should feel atmospheric — like light through clouds, not like a UI button.
- When in doubt, reference the light at 6:45pm on a clear day in late September.

---

## Illustration Style

### General Direction

When a post calls for an illustration — a header image, a conceptual diagram, an ambient figure — it should be **painterly, stylized, and slightly abstracted**. Think gouache, soft pastel, or risograph. It should feel handmade even if it isn't.

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
- Don't make anything look "generated" — if an output has telltale AI artifacts (melted details, impossible geometry, texture soup), regenerate or paint over it.

### Typical Uses

- **Post header images**: A landscape or natural scene that sets a mood for the piece, not a literal depiction of the topic.
- **Ambient figures**: Close crops of flowers, leaves, water, or sky — used to break up long text sections.
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

- **Display / Hero**: Large, serif-italic headlines that feel literary and human. Italic style distinguishes display headings from the roman section headers below.
- **Section Headers**: Same serif family as display (Fraunces), roman weight 400. The italic/roman contrast between hero and section headers creates hierarchy without introducing a second typeface.
- **Body**: Newsreader — a warm, readable serif with humanist warmth at comfortable reading sizes with tall line-height (1.6–1.7). Sidenotes and margin notes use Gentium Book Plus.
- **Pull Quotes**: Set in a serif italic, often oversized, used to surface human voices and values.

**Rules**: No all-caps except for small UI labels (tags like "research," "announcements"). Monospaced fonts are reserved for code blocks only — never in editorial or decorative contexts. Generous whitespace between sections. Let every element breathe.

### Hand-Drawn Layer

The Virgil font (a handwriting face) is the typographic voice of the annotation layer. It should appear only in the "notes" layer, never in body text or headings.

- **Dates**: Set in Virgil with abbreviated months (e.g., "Jan 10, 2026").
- **Diagram labels**: Virgil for labels on conceptual diagrams — like something scribbled in the margin.
- **Decorative elements**: The margin-toggle icon on mobile is a Virgil asterisk (`*`) in superscript, rendered in the warm accent color.

---

## Layout Principles

### Pacing

The site should feel like **reading an annotated manuscript**, not scrolling through a feed. The Tufte CSS foundation — a 55% content column with margin notes — mirrors the layout of an academic paper with a wide margin for the reader's notes.

- The main column carries the argument. Sidenotes, margin notes, and figures populate the margins — just as a reader's annotations would.
- Hand-drawn wavy dividers separate major sections, like a pen stroke between chapters.
- Code blocks have organic, sketchy left borders — they're part of the manuscript, not pasted in from somewhere else.
- Blockquotes and epigraphs should feel like they're floating in air — generous vertical margin, warm left border.

### White Space

White space is a first-class design element, not leftover. Minimum 80px between major sections. A well-set paper has generous margins for a reason — the page should have the quiet confidence of a manuscript that trusts its typography and leaves room for the reader to think.

---

## Motion and Interaction

Animations belong to the hand-drawn layer. The manuscript itself is static; the annotations appear as if someone is marking up the page as you read.

- Rough Notation brackets and underlines animate in as the reader scrolls — like a pen drawing them in real time. These use the warm accent color (`--warm-accent`), not the link green.
- Transitions are slow and eased — nothing snappy or bouncy. Think: the pace of a deep breath.
- Hover states are subtle (slight background shift, soft shadow deepening). No color shifts.
- The overall feeling should be **calm confidence**, never urgency.

### Navigation

The header slides out of view as the reader scrolls down and slides back in when scrolling up — tracking 1:1 with scroll delta, not snapping. This balances editorial pacing (the nav doesn't permanently occupy the page) with usability (it's always one scroll-up away). At the top of the page, the header is always fully visible.

---

## AI Image Generation Guidelines

When using generative AI tools to produce illustrations for blog posts:

### Prompting Strategy

- Always specify a **medium** (gouache, oil pastel, risograph print, watercolor on rough paper) rather than leaving the style open.
- Include **environmental lighting** cues ("golden hour light," "overcast soft diffusion," "early morning blue shadows").
- Name **real-world reference points** for style: "in the style of mid-century travel posters," "like a Rifle Paper Co. print," "reminiscent of David Hockney's landscapes."
- Avoid prompting for "AI art," "digital art," "concept art," or "futuristic" — these trigger default AI aesthetics.

### Curation

- Generate at minimum 20 variations before selecting a candidate.
- Evaluate candidates at 200% zoom. If details fall apart, reject.
- The final image should pass the "could a person have painted this?" test when viewed at normal size.

### Post-Production

- Always color-grade outputs to match the site palette. Raw AI output rarely nails the warmth.
- Composite multiple generations if needed to get the right composition.
- Paint over any artifacts, especially in areas of fine detail (foliage, architecture, water reflections).
- Add subtle paper or canvas texture overlays to sell the "handmade" feel.
- Export at high resolution (minimum 2x display size) to survive responsive scaling.

---

## Voice

The writing voice is:

- **Confident but not grandiose** — "here's what I found" not "this changes everything"
- **Educational and precise** — clear explanations that respect the reader's intelligence
- **Quietly ambitious** — the work speaks for itself; the words stay grounded

The blog is earnest. The visuals exist to make that earnestness feel credible rather than disposable.

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

## What This Is Not

This is not a developer portfolio template or a tech company blog. There are no dark-mode dashboards, no terminal-green-on-black hero sections, no "built with" badge walls, no Medium-clone layouts. It's also not a sterile LaTeX PDF — the manuscript is warm, not clinical.

The aesthetic goal is closer to **Monocle** than **Hacker News**, closer to a marked-up galley proof than a documentation site. If the site could be mistaken for a SaaS landing page, something has gone wrong. If it reads like a carefully typeset paper that someone has annotated with a warm pen — underlining key phrases, bracketing section headings, sketching dividers between ideas — this guide is working.
