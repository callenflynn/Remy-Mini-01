//! Conservative, deterministic text cleaning for pretraining corpora.
//!
//! What it removes: web-extraction artifacts (HTML tags, script/style bodies, entities, page
//! chrome lines, URL remnants, `[edit]` markers), encoding damage (BOM, NUL and control
//! characters), degenerate repetition (the same line three times, one word repeated eight
//! times, punctuation runs) and excess whitespace.
//!
//! What it does not do: rewrite prose, invent text, drop informal language, or apply HTML rules
//! to something that is not HTML. Tag stripping only runs when the document actually looks like
//! markup, so Markdown, code and plain prose pass through with their structure intact.
//!
//! Cleaning is idempotent: `clean(clean(x)) == clean(x)` for every input, so re-running the
//! pipeline over its own output cannot keep peeling text away.

use std::collections::HashSet;

use crate::rng::hash_key;

/// Knobs for one cleaning pass. Defaults are the conservative choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// Documents shorter than this after cleaning are rejected.
    pub min_chars: usize,
    /// Strip bare URLs and Markdown/HTML link targets, keeping the link text.
    pub strip_urls: bool,
    /// Collapse repeated lines, words and punctuation runs.
    pub collapse_repeats: bool,
    /// Minimum share of alphabetic characters for a document to count as prose.
    pub min_alpha_ratio_pct: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            // Dictionary entries are the shortest legitimate documents in the corpus: WordNet
            // glosses run from ~45 characters up. 48 keeps 95% of them; anything higher starts
            // deleting real definitions. Table scraping is caught by the prose gate below, not
            // by length.
            min_chars: 48,
            strip_urls: true,
            collapse_repeats: true,
            min_alpha_ratio_pct: 40,
        }
    }
}

/// Why a document was dropped. Every rejection has an explicit reason, never a silent drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    Empty,
    TooShort { chars: usize, min_chars: usize },
    NotProse { alpha_ratio_pct: u32, min_pct: u32 },
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::Empty => write!(f, "empty after cleaning"),
            Rejection::TooShort { chars, min_chars } => {
                write!(f, "only {chars} chars after cleaning (min {min_chars})")
            }
            Rejection::NotProse {
                alpha_ratio_pct,
                min_pct,
            } => write!(f, "only {alpha_ratio_pct}% alphabetic (min {min_pct}%)"),
        }
    }
}

/// Per-document counters, all measured, so the report can say what changed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CleanStats {
    pub input_chars: usize,
    pub output_chars: usize,
    /// Characters that were inside HTML tags or entities before cleaning.
    pub markup_chars_removed: usize,
    pub entities_decoded: usize,
    pub urls_removed: usize,
    pub chrome_lines_dropped: usize,
    pub repeated_lines_collapsed: usize,
    pub repeated_words_collapsed: usize,
    pub char_runs_collapsed: usize,
    pub control_chars_removed: usize,
}

impl CleanStats {
    /// Accumulate another document's counters (used for per-source totals).
    pub fn add(&mut self, other: &CleanStats) {
        self.input_chars += other.input_chars;
        self.output_chars += other.output_chars;
        self.markup_chars_removed += other.markup_chars_removed;
        self.entities_decoded += other.entities_decoded;
        self.urls_removed += other.urls_removed;
        self.chrome_lines_dropped += other.chrome_lines_dropped;
        self.repeated_lines_collapsed += other.repeated_lines_collapsed;
        self.repeated_words_collapsed += other.repeated_words_collapsed;
        self.char_runs_collapsed += other.char_runs_collapsed;
        self.control_chars_removed += other.control_chars_removed;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cleaned {
    pub text: String,
    pub stats: CleanStats,
}

/// Lines that are page furniture rather than content. Matched only when the whole trimmed line
/// equals the entry (case-insensitively), so prose that happens to mention "Contents" survives.
const CHROME_LINES: [&str; 26] = [
    "jump to content",
    "jump to navigation",
    "main page",
    "contents",
    "current events",
    "random article",
    "about wikipedia",
    "contact us",
    "donate",
    "donations",
    "create account",
    "log in",
    "log out",
    "personal tools",
    "namespaces",
    "views",
    "more",
    "navigation",
    "contribute",
    "tools",
    "print/export",
    "languages",
    "edit links",
    "privacy policy",
    "disclaimers",
    "cookie statement",
];

/// Prefixes of boilerplate lines. `Retrieved from "..."` and MediaWiki's footer text repeat on
/// every page and teach the model nothing.
const CHROME_PREFIXES: [&str; 6] = [
    "retrieved from",
    "this page was last edited",
    "content is available under",
    "text is available under",
    "powered by mediawiki",
    "privacy policy",
];

/// Clean one document. Returns the cleaned text or the reason it was rejected.
pub fn clean(raw: &str, policy: &Policy) -> Result<Cleaned, Rejection> {
    let mut stats = CleanStats {
        input_chars: raw.chars().count(),
        ..Default::default()
    };

    let mut text = normalize_encoding(raw, &mut stats);
    if text.trim().is_empty() {
        return Err(Rejection::Empty);
    }

    // MediaWiki's rendered pages carry JSON payloads in attributes (`data-mw`, `data-parsoid`).
    // They are removed while still entity-escaped: after decoding they contain unbalanced
    // quotes and `<`, which makes any tag scan ambiguous and leaks fragments into the prose.
    text = strip_metadata_attributes(&text);
    // Pandoc renders MediaWiki transclusions as `{#mwAfc .wikitable style="..." mw="{\"parts\":...}"}`.
    // That block is template machinery; the prose around it stays.
    text = strip_attribute_blocks(&text);

    // Entities are decoded before tag stripping so that escaped markup (`&lt;div&gt;`) is dealt
    // with in the same pass instead of surviving as literal text, and so a second pass over the
    // output finds nothing left to decode.
    text = decode_entities_until_stable(&text, &mut stats.entities_decoded);

    // Markup rules only where there is markup. A Markdown file with a stray `<` is left alone.
    if looks_like_html(&text) {
        let before = text.chars().count();
        text = strip_scripts_and_comments(&text);
        text = strip_tags(&text);
        stats.markup_chars_removed += before.saturating_sub(text.chars().count());
    }

    // Wiki and Markdown link/template syntax survives tag stripping: the dumps are rendered
    // HTML converted to Markdown, so `[[target|text]]`, `{{cite ...}}` and `[](//url)` remain.
    text = strip_links_and_templates(&text);
    text = drop_edit_markers(&text);
    if policy.strip_urls {
        text = strip_urls(&text, &mut stats.urls_removed);
    }
    text = drop_chrome_lines(&text, &mut stats.chrome_lines_dropped);
    if policy.collapse_repeats {
        text = collapse_repeated_lines(&text, &mut stats.repeated_lines_collapsed);
        text = collapse_repeated_words(&text, &mut stats.repeated_words_collapsed);
        text = collapse_char_runs(&text, &mut stats.char_runs_collapsed);
    }
    text = normalize_whitespace(&text);

    stats.output_chars = text.chars().count();
    if text.is_empty() {
        return Err(Rejection::Empty);
    }
    if stats.output_chars < policy.min_chars {
        return Err(Rejection::TooShort {
            chars: stats.output_chars,
            min_chars: policy.min_chars,
        });
    }
    let alpha_ratio_pct = alpha_ratio_pct(&text);
    if alpha_ratio_pct < policy.min_alpha_ratio_pct {
        return Err(Rejection::NotProse {
            alpha_ratio_pct,
            min_pct: policy.min_alpha_ratio_pct,
        });
    }
    Ok(Cleaned { text, stats })
}

fn alpha_ratio_pct(text: &str) -> u32 {
    let mut total = 0usize;
    let mut alpha = 0usize;
    for c in text.chars() {
        total += 1;
        if c.is_alphabetic() {
            alpha += 1;
        }
    }
    if total == 0 {
        return 0;
    }
    ((alpha * 100) / total) as u32
}

/// Strip BOM, normalize line endings, and drop NUL/control characters that carry no meaning.
fn normalize_encoding(raw: &str, stats: &mut CleanStats) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            '\u{feff}' => stats.control_chars_removed += 1,
            '\r' => {
                // Normalized away entirely; `\r\n` becomes `\n` on the next character.
                stats.control_chars_removed += 1;
            }
            '\n' | '\t' => out.push(c),
            c if c.is_control() => stats.control_chars_removed += 1,
            c => out.push(c),
        }
    }
    out
}

/// HTML if tags account for a meaningful share of the document. Counting characters rather than
/// tags matters: one `<div class="...">` wrapper is a single tag but thousands of characters.
fn looks_like_html(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let mut tag_bytes = 0usize;
    let mut tags = 0usize;
    for (i, _) in text.match_indices('<') {
        if let Some(end) = find_tag_end(&text[i..]) {
            tag_bytes += end + 1;
            tags += 1;
        }
    }
    let pct = tag_bytes * 100 / text.len();
    pct >= 5 || (tags >= 20 && pct >= 2)
}

/// Index of the `>` that closes the tag starting at `text[0]`, preferring one outside quotes.
/// If the quotes never balance (decoded JSON with escaped quotes does this), fall back to the
/// first `>` seen: a slightly early cut is far better than keeping the whole document raw.
fn find_tag_end(text: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    let mut inside_quote_end: Option<usize> = None;
    for (i, c) in text.char_indices().skip(1) {
        if c == '>' {
            if quote.is_none() {
                return Some(i);
            }
            inside_quote_end.get_or_insert(i);
        }
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                } else if c == '<' {
                    // A second `<` outside quotes: this was never a tag.
                    return None;
                }
            }
        }
    }
    inside_quote_end
}

/// Remove MediaWiki metadata attributes, value and all, before any entity decoding.
fn strip_metadata_attributes(text: &str) -> String {
    const ATTRS: [&str; 6] = [
        "data-mw",
        "data-parsoid",
        "data-mw-section-id",
        "typeof",
        "about",
        "resource",
    ];
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    'outer: while !rest.is_empty() {
        for attr in ATTRS {
            if rest.starts_with(attr) {
                let after = &rest[attr.len()..];
                let trimmed = after.trim_start();
                if let Some(value) = trimmed.strip_prefix('=') {
                    let value = value.trim_start();
                    let quote = value.chars().next().filter(|c| *c == '"' || *c == '\'');
                    let skip = match quote {
                        Some(q) => value[1..].find(q).map(|p| p + 2),
                        None => value.find([' ', '>']),
                    };
                    if let Some(skip) = skip {
                        // Drop the attribute name plus its value, keep surrounding whitespace.
                        rest = &value[skip..];
                        continue 'outer;
                    }
                }
            }
        }
        let c = rest.chars().next().unwrap_or(' ');
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}
/// Drop pandoc attribute blocks: `{#id .class key="value"}`. They carry template payloads and
/// table styling, never prose. Only `{#` and `{.` open a block, and an unbalanced block is left
/// untouched, so ordinary braces (LaTeX, JSON, code) survive.
fn strip_attribute_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find('{') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        let opens_block = matches!(tail[1..].chars().next(), Some('#') | Some('.'));
        match opens_block.then(|| matching_brace(tail)).flatten() {
            Some(end) => rest = &tail[end + 1..],
            None => {
                out.push('{');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Byte offset of the `}` closing the `{` at `text[0]`, honouring nesting.
fn matching_brace(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in text.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

/// Remove `<script>...</script>` and `<style>...</style>` bodies and `<!-- -->` comments.
fn strip_scripts_and_comments(text: &str) -> String {
    // `to_ascii_lowercase` preserves byte offsets, so one lowercase shadow can be searched while
    // slices are taken from the original.
    let lower = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < text.len() {
        let next = ["<script", "<style", "<!--"]
            .iter()
            .filter_map(|m| lower[i..].find(m).map(|p| (i + p, *m)))
            .min();
        let Some((start, marker)) = next else {
            out.push_str(&text[i..]);
            return out;
        };
        out.push_str(&text[i..start]);
        let close = if marker == "<!--" {
            lower[start..].find("-->").map(|p| p + 3)
        } else {
            let closing = format!("</{}>", &marker[1..]);
            lower[start..].find(&closing).map(|p| p + closing.len())
        };
        match close {
            Some(end) => i = start + end,
            // An unterminated block would otherwise swallow all following prose, so keep the
            // opener as text and let tag stripping deal with it.
            None => {
                out.push_str(&text[start..start + marker.len()]);
                i = start + marker.len();
            }
        }
    }
    out
}

/// Replace block-level tags with a newline (so paragraphs stay separated) and inline tags with
/// nothing, then remove any remaining tag-shaped text.
fn strip_tags(text: &str) -> String {
    const BLOCK: [&str; 18] = [
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "tr",
        "td",
        "th",
        "table",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "hr",
    ];
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('<') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let Some(end) = find_tag_end(tail) else {
            // Not a tag (a math comparison, or a `<` inside an attribute value). Keep the
            // character and carry on: bailing here would leave the rest of the document raw.
            out.push('<');
            rest = &tail[1..];
            continue;
        };
        let inner = &tail[1..end];
        if !tag_shaped(inner) {
            out.push_str(&tail[..=end]);
            rest = &tail[end + 1..];
            continue;
        }
        let name: String = inner
            .trim_start_matches('/')
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if BLOCK.contains(&name.as_str()) {
            out.push('\n');
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

fn tag_shaped(inner: &str) -> bool {
    if inner.is_empty() {
        return false;
    }
    let first = inner.chars().next().unwrap_or('x');
    if !(first.is_ascii_alphabetic() || first == '/' || first == '!') {
        return false;
    }
    // `a < b` and `x < 3` must stay text: a real tag names an element or closes one.
    let name_end = inner
        .trim_start_matches('/')
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(inner.len());
    name_end > 0 || inner.starts_with("!--")
}

/// Decode until no further entity is found, so double-escaped text (`&amp;amp;`) is fully
/// decoded in one pass and cleaning is idempotent.
fn decode_entities_until_stable(text: &str, decoded: &mut usize) -> String {
    let mut current = text.to_string();
    for _ in 0..3 {
        let before = *decoded;
        current = decode_entities(&current, decoded);
        if *decoded == before {
            break;
        }
    }
    current
}

/// Decode the HTML entities that actually appear in web text. Unknown entities are left alone.
fn decode_entities(text: &str, decoded: &mut usize) -> String {
    const NAMED: [(&str, char); 20] = [
        ("&amp;", '&'),
        ("&lt;", '<'),
        ("&gt;", '>'),
        ("&quot;", '"'),
        ("&apos;", '\''),
        ("&nbsp;", ' '),
        ("&ndash;", '–'),
        ("&mdash;", '—'),
        ("&hellip;", '…'),
        ("&lsquo;", '‘'),
        ("&rsquo;", '’'),
        ("&ldquo;", '“'),
        ("&rdquo;", '”'),
        ("&copy;", '©'),
        ("&reg;", '®'),
        ("&deg;", '°'),
        ("&times;", '×'),
        ("&middot;", '·'),
        ("&bull;", '•'),
        ("&sect;", '§'),
    ];
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let end = tail[1..].find(';').map(|i| i + 1).filter(|i| *i <= 12);
        let Some(end) = end else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let entity = &tail[..=end];
        let body = &entity[1..entity.len() - 1];
        let replacement = if let Some(hex) = body.strip_prefix("#x").or(body.strip_prefix("#X")) {
            u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
        } else if let Some(dec) = body.strip_prefix('#') {
            dec.parse::<u32>().ok().and_then(char::from_u32)
        } else {
            NAMED
                .iter()
                .find(|(name, _)| *name == entity)
                .map(|(_, c)| *c)
        };
        match replacement {
            Some(c) => {
                out.push(c);
                *decoded += 1;
            }
            None => out.push_str(entity),
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Strip wiki and Markdown link/template syntax while keeping the text a reader would see:
///
/// - `[[Target|shown text]]` → `shown text`, `[[Target]]` → `Target`
/// - `{{cite web|...}}` and other template calls are dropped: they are citation machinery and
///   navigation boxes, not prose
/// - `[text](url "title")` → `text`, `![alt](img)` → dropped, `[](url)` → dropped
/// - `[12]` numeric reference markers are dropped
fn strip_links_and_templates(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    // Only `[` and `{` can start anything here; `!` is handled by looking back at `out`.
    while let Some(pos) = rest.find(['[', '{']) {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        // `{` is only syntax as a complete `{{template}}`. Everything else -- LaTeX
        // (`\frac{{a}}{b}`), JSON, code -- is copied through byte for byte, because inventing
        // a `[` here is exactly the corruption this cleaner exists to remove.
        if tail.starts_with('{') {
            if let Some(body) = tail.strip_prefix("{{")
                && let Some(end) = body.find("}}")
            {
                rest = &body[end + 2..];
            } else {
                out.push('{');
                rest = &tail[1..];
            }
            continue;
        }
        // `[[Target|shown text]]`; an unterminated `[[` is text, not a link.
        if let Some(body) = tail.strip_prefix("[[") {
            match body.find("]]") {
                Some(end) => {
                    let inner = &body[..end];
                    // The display text is what follows the last pipe.
                    let shown = inner.rsplit('|').next().unwrap_or(inner);
                    out.push_str(shown.trim());
                    rest = &body[end + 2..];
                }
                None => {
                    out.push_str("[[");
                    rest = body;
                }
            }
            continue;
        }
        {
            let after_bracket = &tail[1..];
            match after_bracket.find(']') {
                Some(close) => {
                    let label = &after_bracket[..close];
                    let after = &after_bracket[close + 1..];
                    if let Some(paren_body) = after.strip_prefix('(')
                        && let Some(paren) = paren_body.find(')')
                    {
                        let is_image = out.ends_with('!') || label.trim().is_empty();
                        if is_image {
                            if out.ends_with('!') {
                                out.pop();
                            }
                        } else {
                            out.push_str(label.trim());
                        }
                        rest = &paren_body[paren + 1..];
                    } else if !label.is_empty()
                        && label.chars().all(|c| c.is_ascii_digit() || c == ' ')
                    {
                        // `[12]` reference marker.
                        rest = after;
                    } else {
                        out.push('[');
                        rest = after_bracket;
                    }
                }
                None => {
                    out.push('[');
                    rest = after_bracket;
                }
            }
        }
    }
    out.push_str(rest);
    out
}

/// `[edit]` / `[ edit ]` section links are MediaWiki chrome, not content.
fn drop_edit_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('[') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let Some(end) = tail.find(']').filter(|e| *e <= 10) else {
            out.push('[');
            rest = &tail[1..];
            continue;
        };
        let inner = tail[1..end].trim().to_ascii_lowercase();
        if inner == "edit" || inner == "edit source" {
            rest = &tail[end + 1..];
        } else {
            out.push_str(&tail[..=end]);
            rest = &tail[end + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// Remove URL remnants while keeping whatever they were labelling: `[text](url)` keeps `text`,
/// `see https://x` keeps `see`. `<a href="...">` was already reduced to its text by tag
/// stripping.
fn strip_urls(text: &str, removed: &mut usize) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Deliberately no bare `//`: after tag stripping a scheme-relative URL has lost its
        // owner, while `//` also starts comments and paths in code and prose.
        let url_at = ["http://", "https://"]
            .iter()
            .find(|p| text[i..].starts_with(**p));
        match url_at {
            Some(_) => {
                let mut j = i;
                while j < bytes.len() && !bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                // Also consume a trailing `)` `]` only when the URL itself balanced them.
                *removed += 1;
                i = j;
                // Collapse the space a removed trailing URL leaves behind.
                if out.ends_with(' ') && text[i..].starts_with(' ') {
                    i += 1;
                }
            }
            None => {
                let c = text[i..].chars().next().unwrap_or(' ');
                out.push(c);
                i += c.len_utf8();
            }
        }
    }
    out
}

/// Drop whole lines that are page furniture. Short exact matches plus known footer prefixes.
fn drop_chrome_lines(text: &str, dropped: &mut usize) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split('\n') {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        let is_chrome = !trimmed.is_empty()
            && (CHROME_LINES.contains(&lower.as_str())
                || CHROME_PREFIXES.iter().any(|p| lower.starts_with(p)));
        if is_chrome {
            *dropped += 1;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Collapse runs of three or more identical non-empty lines to one. Two identical lines in a
/// row are left alone: quoted text and dialogue legitimately repeat a single line.
fn collapse_repeated_lines(text: &str, collapsed: &mut usize) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    let mut i = 0usize;
    while i < lines.len() {
        let mut j = i + 1;
        while j < lines.len() && lines[j] == lines[i] {
            j += 1;
        }
        let run = j - i;
        if run >= 3 && !lines[i].trim().is_empty() {
            out.push(lines[i]);
            *collapsed += run - 1;
        } else {
            out.extend_from_slice(&lines[i..j]);
        }
        i = j;
    }
    out.join("\n")
}

/// Collapse a word repeated eight or more times in a row to two occurrences. Three or four is
/// rhetoric ("very very good"); eight in a row is a generation or scraping artifact.
fn collapse_repeated_words(text: &str, collapsed: &mut usize) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split('\n') {
        let words: Vec<&str> = line.split(' ').collect();
        let mut i = 0usize;
        while i < words.len() {
            let mut j = i + 1;
            while j < words.len() && words[j].eq_ignore_ascii_case(words[i]) {
                j += 1;
            }
            let run = j - i;
            if run >= 8 && !words[i].is_empty() {
                out.push_str(words[i]);
                out.push(' ');
                out.push_str(words[i]);
                out.push(' ');
                *collapsed += run - 2;
            } else {
                for w in &words[i..j] {
                    out.push_str(w);
                    out.push(' ');
                }
            }
            i = j;
        }
        if out.ends_with(' ') && !line.ends_with(' ') {
            out.pop();
        }
        out.push('\n');
    }
    out.trim_end_matches('\n').to_string()
}

/// Collapse runs of eight or more identical punctuation characters (table borders, separators).
/// Keeps the first three so intentional `---` rules still read as a rule.
fn collapse_char_runs(text: &str, collapsed: &mut usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let mut j = i + 1;
        while j < chars.len() && chars[j] == c {
            j += 1;
        }
        let run = j - i;
        if run >= 8 && !c.is_alphanumeric() && c != '\n' && !c.is_whitespace() {
            for _ in 0..3 {
                out.push(c);
            }
            *collapsed += run - 3;
        } else {
            for _ in 0..run {
                out.push(c);
            }
        }
        i = j;
    }
    out
}

/// Collapse runs of spaces/tabs, trim line ends, and leave at most one blank line between
/// paragraphs.
fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.split('\n') {
        let collapected = collapse_spaces(line);
        let trimmed = collapected.trim();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(trimmed);
        out.push('\n');
    }
    out.trim().to_string()
}

fn collapse_spaces(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut last_space = false;
    for c in line.chars() {
        if c == ' ' || c == '\t' {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
        } else {
            out.push(c);
            last_space = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy::default()
    }

    fn cleaned(raw: &str) -> String {
        clean(raw, &policy()).expect("should be accepted").text
    }

    const PROSE: &str = "The arctic tern migrates between hemispheres each year, following the \
        summer sun across open water and coastal staging grounds where it feeds on small fish \
        and invertebrates. Researchers tracking tagged birds recorded round trips of more than \
        seventy thousand kilometres, which is among the longest migrations known for any animal.";

    #[test]
    fn strips_html_tags_and_keeps_prose() {
        let html = format!(
            "<div class=\"mw-parser-output\"><p>{PROSE}</p><span data-mw=\"{{x}}\">tail</span></div>"
        );
        let out = cleaned(&html);
        assert!(out.contains("arctic tern"), "{out}");
        assert!(!out.contains('<'), "{out}");
        assert!(!out.contains("mw-parser-output"), "{out}");
        assert!(out.contains("tail"), "{out}");
    }

    #[test]
    fn decodes_entities_including_numeric_and_double_escaped() {
        let html = format!("<p>&quot;{PROSE}&quot; 5 &amp;amp; 7 &lt; 9 &#233; &#x41;</p>");
        let out = cleaned(&html);
        assert!(out.contains("\"The arctic tern"), "{out}");
        assert!(out.contains("5 & 7 < 9"), "{out}");
        assert!(out.contains('é') && out.contains('A'), "{out}");
        assert!(!out.contains("&quot;") && !out.contains("&amp;"), "{out}");
    }

    #[test]
    fn drops_script_and_style_bodies() {
        let html = format!(
            "<style>.mw {{ color: red }}</style><script>var x = \"secret\";</script><p>{PROSE}</p>"
        );
        let out = cleaned(&html);
        assert!(out.contains("arctic tern"), "{out}");
        assert!(!out.contains("secret") && !out.contains("color"), "{out}");
    }

    #[test]
    fn keeps_markdown_structure_untouched() {
        let md = format!(
            "# Migration\n\n## Routes\n\n- Arctic tern\n- Bar-tailed godwit\n\n> {PROSE}\n\nUse `--verbose` for details."
        );
        let out = cleaned(&md);
        assert!(out.contains("# Migration"));
        assert!(out.contains("- Bar-tailed godwit"));
        assert!(out.contains("> The arctic tern"));
        assert!(out.contains("`--verbose`"));
    }

    #[test]
    fn strips_urls_but_keeps_the_text_they_labelled() {
        let text = format!(
            "See [the tracking study](https://example.org/tracking) or https://example.org/raw for details. {PROSE}"
        );
        let out = cleaned(&text);
        assert!(!out.contains("http"), "{out}");
        assert!(out.contains("the tracking study"), "{out}");
        assert!(out.contains("for details"), "{out}");
    }

    #[test]
    fn quoted_attribute_values_do_not_leak_into_prose() {
        // The real dumps carry decoded `data-mw` JSON in attributes, quotes and all.
        let html = format!(
            "<div data-mw=\"{{{{\"parts\":[{{{{\"template\":{{{{\"target\":{{{{\"wt\":\"cite\"}}}}}}}}}}]}}}}\" class=\"mw\"><p>{PROSE}</p></div>"
        );
        let out = cleaned(&html);
        assert!(out.contains("arctic tern"), "{out}");
        assert!(!out.contains("parts"), "{out}");
        assert!(!out.contains("class=") && !out.contains("mw\">"), "{out}");
    }

    #[test]
    fn converts_wiki_links_and_drops_template_calls() {
        let raw = format!(
            "[[Algebraic operation|operations]] of [[addition]] ({{{{tmath|1= + }}}}) are studied.[12] {PROSE}"
        );
        let out = cleaned(&raw);
        assert!(out.contains("operations of addition"), "{out}");
        assert!(!out.contains("[[") && !out.contains("{{"), "{out}");
        assert!(!out.contains("[12]"), "{out}");
    }

    #[test]
    fn keeps_link_text_and_drops_images_and_empty_links() {
        let raw = format!(
            "See [the ring of integers](//en.wikipedia.org/wiki/Ring_of_integers \"Ring\"). ![](image.png) [](//en.wikipedia.org/wiki/X) {PROSE}"
        );
        let out = cleaned(&raw);
        assert!(out.contains("See the ring of integers."), "{out}");
        assert!(!out.contains("image.png") && !out.contains("]["), "{out}");
    }

    #[test]
    fn latex_braces_and_math_are_never_rewritten() {
        // LibreTexts content is LaTeX-heavy: `\frac{1}{2}` must not become `\frac[1}[2}`.
        let text = format!(
            "Scale by \\(\\frac{{1}}{{2}}\\) and \\(\\PageIndex{{3}}\\). {{x}} is a set. {PROSE}"
        );
        let out = cleaned(&text);
        assert!(out.contains("\\frac{1}{2}"), "{out}");
        assert!(out.contains("\\PageIndex{3}"), "{out}");
        assert!(!out.contains("frac[1}"), "{out}");
    }

    #[test]
    fn an_unclosed_template_brace_is_copied_through() {
        // `\frac{{(x+2)}^2(x-2)}{2...}` from LibreTexts: `{{` but no `}}`. Falling through to the
        // link branch here used to emit a `[` and silently rewrite real maths.
        let text =
            format!("Given \\(f(x)=\\dfrac{{{{(x+2)}}^2(x-2)}}{{2{{(x-1)}}^2(x-3)}}\\). {PROSE}");
        let out = cleaned(&text);
        assert!(
            out.contains("\\dfrac{{(x+2)}^2(x-2)}{2{(x-1)}^2(x-3)}"),
            "{out}"
        );
        assert!(!out.contains("\\dfrac["), "{out}");
    }

    #[test]
    fn pandoc_attribute_blocks_are_dropped_but_prose_stays() {
        let raw = format!(
            "Row one {{{{#mwAfc .wikitable style=\"text-align: center\" mw=\"{{\\\"parts\\\":[{{\\\"template\\\":{{\\\"wt\\\":\\\"Worldwide Internet users\\\"}}}}]}}\"}}}} {PROSE}"
        );
        let out = cleaned(&raw);
        assert!(out.contains("Row one"), "{out}");
        assert!(out.contains("arctic tern"), "{out}");
        assert!(
            !out.contains("mwAfc") && !out.contains("wikitable"),
            "{out}"
        );
        assert!(!out.contains("parts"), "{out}");
        // An unbalanced `{` is prose or a typo: keep it rather than eat the rest of the page.
        let broken = format!("a stray {{ brace and {PROSE}");
        assert!(cleaned(&broken).contains("stray { brace"));
    }

    #[test]
    fn drops_mediawiki_chrome_lines() {
        let page = format!(
            "Jump to content\nMain page\nContents\n\n{PROSE}\n\nRetrieved from \"https://en.wikipedia.org/x\"\n\n[ edit ]\n\nThis page was last edited on 12 May 2024."
        );
        let out = cleaned(&page);
        assert!(out.contains("arctic tern"), "{out}");
        for chrome in [
            "Jump to content",
            "Retrieved from",
            "last edited",
            "[ edit ]",
        ] {
            assert!(!out.contains(chrome), "{chrome} survived: {out}");
        }
    }

    #[test]
    fn collapses_repeated_lines_words_and_punctuation() {
        let text = format!(
            "{PROSE}\n{PROSE}\n{PROSE}\n\nprivate private private private private private private private private here\n\n==============================\nend"
        );
        let out = cleaned(&text);
        assert_eq!(out.matches("arctic tern").count(), 1, "{out}");
        assert!(!out.contains("private private private private"), "{out}");
        assert!(!out.contains("=============================="), "{out}");
        assert!(out.ends_with("end"), "{out}");
    }

    #[test]
    fn removes_bom_controls_and_normalizes_line_endings() {
        let raw = format!(
            "{bom}first\r\nsecond\r\n{nul}third{bell} {PROSE}",
            bom = '\u{feff}',
            nul = '\u{0}',
            bell = '\u{7}'
        );
        let out = cleaned(&raw);
        assert!(out.starts_with("first\nsecond\nthird"), "{out}");
        assert!(!out.contains('\u{feff}') && !out.contains('\u{0}'));
    }

    #[test]
    fn plain_speech_text_survives_byte_for_byte() {
        let speech = "Donald Trump (00:00):\n\nThank you.\n\n(00:14)\nWell, a very, very special hello to Syracuse. We love Syracuse. \"Always loved it,\" he said, at 25 days out.";
        let keep_short = Policy {
            min_chars: 20,
            ..Policy::default()
        };
        let out = clean(speech, &keep_short).expect("speech is prose").text;
        assert_eq!(out, speech);
    }

    #[test]
    fn comparison_operators_are_not_tags() {
        let text = format!("If x < y and 3 > 2 then print(x) — {PROSE}");
        let out = cleaned(&text);
        assert!(out.contains("x < y"), "{out}");
        assert!(out.contains("3 > 2"), "{out}");
    }

    #[test]
    fn rejects_empty_short_and_non_prose_documents() {
        assert_eq!(clean("   \n\t ", &policy()), Err(Rejection::Empty));
        assert!(matches!(
            clean("too short to train on", &policy()),
            Err(Rejection::TooShort { .. })
        ));
        // A dictionary-sized entry is long enough to keep under the default policy.
        assert!(
            clean(
                "able (adjective): having the necessary means or skill to do something; \
            able to swim, able to program a computer.",
                &policy()
            )
            .is_ok()
        );
        let table: String = (0..80)
            .map(|i| format!("| {} | {} | {} |", i, i * 3, i * 7))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(matches!(
            clean(&table, &policy()),
            Err(Rejection::NotProse { .. })
        ));
    }

    #[test]
    fn cleaning_is_idempotent_on_hostile_input() {
        let samples = [
            format!("<div><p>&quot;{PROSE}&quot;</p></div>"),
            format!("<table>\n<tr><td>{PROSE}</td></tr>\n</table>"),
            format!("{PROSE}\n{PROSE}\n{PROSE}"),
            format!("Jump to content\n\n{PROSE}\n\nRetrieved from \"https://x/y\""),
            format!("&lt;div&gt;{PROSE}&lt;/div&gt;"),
            format!("{PROSE}\n\n==============================\n\n{PROSE}"),
        ];
        for sample in samples {
            let once = clean(&sample, &policy()).expect("first pass accepts").text;
            let twice = clean(&once, &policy()).expect("second pass accepts").text;
            assert_eq!(once, twice, "not idempotent for {sample:.60}");
        }
    }

    #[test]
    fn cleaning_is_deterministic() {
        let html = format!("<p>{PROSE}</p>\n<p>{PROSE}</p>");
        let a = clean(&html, &policy()).unwrap();
        let b = clean(&html, &policy()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn deduper_removes_exact_documents_and_repeated_paragraphs() {
        let mut d = Deduper::new();
        assert!(d.keep_document(PROSE));
        assert!(!d.keep_document(PROSE));
        assert_eq!(d.documents_removed, 1);

        let long_a = PROSE.to_string();
        let long_b = format!("{PROSE} {PROSE}");
        let body = format!("{long_a}\n\n{long_b}");
        let once = d.dedupe_paragraphs(&body, 200);
        let twice = d.dedupe_paragraphs(&body, 200);
        assert_eq!(twice.trim(), "");
        assert!(once.contains("arctic tern"));
        assert!(d.paragraphs_removed >= 2);
    }
}

/// Exact-document and paragraph deduplication across a corpus.
///
/// Document dedup uses a 64-bit hash of the cleaned text; paragraph dedup removes paragraphs
/// that already appeared anywhere in the corpus *and* are long enough to be boilerplate rather
/// than a legitimate repeated short phrase.
#[derive(Default)]
pub struct Deduper {
    documents: HashSet<u64>,
    paragraphs: HashSet<u64>,
    pub documents_removed: usize,
    pub paragraphs_removed: usize,
}

impl Deduper {
    pub fn new() -> Self {
        Self::default()
    }

    /// `false` when an identical document was already kept.
    pub fn keep_document(&mut self, text: &str) -> bool {
        if self.documents.insert(hash_key(0, text)) {
            true
        } else {
            self.documents_removed += 1;
            false
        }
    }

    /// Remove paragraphs seen before. Only paragraphs of at least `min_chars` are deduplicated,
    /// so short lines like "Introduction" or "See also" are left to the cleaner.
    pub fn dedupe_paragraphs(&mut self, text: &str, min_chars: usize) -> String {
        let mut out = String::with_capacity(text.len());
        for para in text.split("\n\n") {
            if para.chars().count() >= min_chars && !self.paragraphs.insert(hash_key(1, para)) {
                self.paragraphs_removed += 1;
                continue;
            }
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(para);
        }
        out
    }
}
