# Dissolving Artwork into the Site Background

Use real per-pixel alpha when an illustration should blend into the page like watercolor on unbounded paper. Do not rely only on CSS masks: they preserve the image's paper-colored rectangle and usually produce a visibly geometric vignette.

## Process

1. Start with the selected full-resolution RGB PNG.
2. Use Sharp to read its raw RGB pixels.
3. Estimate pigment strength from luminance:
   - Make pale paper pixels transparent.
   - Keep dark ink and saturated painted details opaque.
   - Retain intermediate watercolor tones with partial alpha.
4. Multiply pigment opacity by four feathered edge ramps so every canvas boundary reaches zero alpha.
5. Vary the ramp boundaries slightly with deterministic sine waves or seeded noise. This prevents a perfect rectangle or oval.
6. Join the generated mask as the image's alpha channel and save a new, non-destructive RGBA PNG.
7. Remove CSS masks and opaque gradient overlays from the rendered image. Let transparent pixels reveal `--bg-color` directly.
8. Verify the optimized PNG and WebP outputs preserve alpha.

The mask is conceptually:

```text
alpha = pigment × left × right × top × bottom
```

## Sharp Pattern

Adapt thresholds and fade positions to the image rather than treating these values as universal:

```js
import sharp from "sharp";

const input = "src/assets/img/source.png";
const output = "src/assets/img/source-transparent.png";
const { data, info } = await sharp(input)
  .removeAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });
const alpha = Buffer.alloc(info.width * info.height);

const smoothstep = (start, end, value) => {
  const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
  return t * t * (3 - 2 * t);
};
const pigmentOpacity = (luminance) =>
  Math.pow(1 - smoothstep(166, 236, luminance), 0.82);

for (let y = 0; y < info.height; y++) {
  const ny = y / (info.height - 1);
  for (let x = 0; x < info.width; x++) {
    const nx = x / (info.width - 1);
    const offset = (y * info.width + x) * info.channels;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    const wobbleX =
      0.018 * Math.sin(ny * 19.7) + 0.009 * Math.sin(ny * 43.1 + 1.4);
    const wobbleY =
      0.016 * Math.sin(nx * 17.3 + 0.8) + 0.008 * Math.sin(nx * 38.9);
    const left = smoothstep(0.035 + wobbleX, 0.29 + wobbleX, nx);
    const right = 1 - smoothstep(0.89 + wobbleX, 1, nx);
    const top = smoothstep(0.015 + wobbleY, 0.15 + wobbleY, ny);
    const bottom = 1 - smoothstep(0.72 + wobbleY, 0.995, ny);

    alpha[y * info.width + x] = Math.round(
      255 * pigmentOpacity(luminance) * left * right * top * bottom,
    );
  }
}

await sharp(data, { raw: info })
  .joinChannel(alpha, {
    raw: { width: info.width, height: info.height, channels: 1 },
  })
  .png({ compressionLevel: 9 })
  .toFile(output);
```

## Tuning

- Raise the pale threshold when too much paper remains visible.
- Lower the dark threshold when pale painted details disappear.
- Move the edge ramps inward when a hard canvas boundary remains.
- Keep the focal subject outside aggressive fade zones.
- Prefer a lopsided composition with transparent negative space near overlaid text.
- Preview by flattening the RGBA image onto `#FEF9ED`, the site's current background, rather than onto white or a checkerboard.

## Validation

Check all of the following before publishing:

- Sharp reports four channels and `hasAlpha: true`.
- All four corner alpha values equal zero.
- A representative focal-subject pixel remains near alpha 255.
- The Eleventy-generated PNG and WebP variants retain alpha.
- No CSS background, mask, or overlay reintroduces a rectangular layer.
- The page looks blended at desktop and mobile widths.

Keep the original generated image and publish the transparent derivative under a distinct filename so the mask can be tuned without another paid generation call.
