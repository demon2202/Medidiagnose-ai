/* Backend payloads occasionally contain emoji (e.g. in recommendation
   titles). The UI is strictly emoji-free, so display strings are cleaned. */

const EMOJI_RE = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}]/gu;
const INVISIBLE_RE = /[\uFE00-\uFE0F\u200D]/g;

export function stripEmoji(str) {
  if (typeof str !== 'string') return str;
  return str.replace(EMOJI_RE, '').replace(INVISIBLE_RE, '').replace(/\s{2,}/g, ' ').trim();
}

/* Deep-clean an API result object so every rendered string is emoji-free. */
export function cleanResult(value) {
  if (typeof value === 'string') return stripEmoji(value);
  if (Array.isArray(value)) return value.map(cleanResult);
  if (value && typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = cleanResult(v);
    return out;
  }
  return value;
}
