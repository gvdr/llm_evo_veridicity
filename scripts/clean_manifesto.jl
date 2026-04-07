#!/usr/bin/env julia
#
# clean_manifesto.jl
#
# Reads each raw Communist Manifesto HTML file, strips HTML tags, entities,
# navigation/headers/footers from marxists.org template, footnotes,
# section headings, lowercases, strips diacritics, removes non-letter
# non-whitespace characters, collapses whitespace, and writes cleaned versions.

using Unicode

const DATA_DIR = joinpath(@__DIR__, "..", "data", "manifesto")

# ----------------------------------------------------------------
# Encoding handling
# ----------------------------------------------------------------

"""
    read_html_file(path::String) -> String

Read an HTML file, detecting its encoding from <meta> charset declarations.
Falls back to Latin-1 if the file contains non-UTF-8 bytes and no charset
is declared (many marxists.org pages use ISO-8859-1 or Windows-1250).
"""
function read_html_file(path::String)
    raw = read(path)

    # Check if it's valid UTF-8
    is_valid_utf8 = true
    try
        s = String(copy(raw))
        # Try to iterate all chars to check validity
        for c in s
            if !isvalid(c)
                is_valid_utf8 = false
                break
            end
        end
    catch
        is_valid_utf8 = false
    end

    if is_valid_utf8
        text = String(copy(raw))
        # Even if valid UTF-8, check for regex-breaking invalid chars
        has_invalid = false
        for c in text
            if !isvalid(c)
                has_invalid = true
                break
            end
        end
        if !has_invalid
            return text
        end
    end

    # Not valid UTF-8 -- transcode assuming Latin-1 / Windows-1252
    # (each byte maps directly to a Unicode codepoint)
    buf = IOBuffer()
    for b in raw
        write(buf, Char(b))
    end
    return String(take!(buf))
end

const FILES = [
    ("manifesto_german_raw.txt",     "manifesto_german.txt"),
    ("manifesto_english_raw.txt",    "manifesto_english.txt"),
    ("manifesto_french_raw.txt",     "manifesto_french.txt"),
    ("manifesto_italian_raw.txt",    "manifesto_italian.txt"),
    ("manifesto_spanish_raw.txt",    "manifesto_spanish.txt"),
    ("manifesto_portuguese_raw.txt", "manifesto_portuguese.txt"),
    ("manifesto_polish_raw.txt",     "manifesto_polish.txt"),
    ("manifesto_finnish_raw.txt",    "manifesto_finnish.txt"),
    ("manifesto_dutch_raw.txt",      "manifesto_dutch.txt"),
    ("manifesto_czech_raw.txt",      "manifesto_czech.txt"),
]

# ----------------------------------------------------------------
# HTML entity decoding
# ----------------------------------------------------------------

const NAMED_ENTITIES = Dict{String, String}(
    "amp"    => "&",
    "lt"     => "<",
    "gt"     => ">",
    "quot"   => "\"",
    "apos"   => "'",
    "nbsp"   => " ",
    "ndash"  => "-",
    "mdash"  => "-",
    "laquo"  => "\"",
    "raquo"  => "\"",
    "ldquo"  => "\"",
    "rdquo"  => "\"",
    "lsquo"  => "'",
    "rsquo"  => "'",
    "hellip" => "...",
    "bull"   => " ",
    "middot" => " ",
    "copy"   => "",
    "reg"    => "",
    "trade"  => "",
    "times"  => "x",
    "divide" => "/",
    "shy"    => "",
    "ensp"   => " ",
    "emsp"   => " ",
    "thinsp" => " ",
    "zwj"    => "",
    "zwnj"   => "",
    # Accented characters - these will be handled by strip_diacritics later,
    # but we decode them to the actual Unicode character first.
    "agrave" => "\u00e0", "aacute" => "\u00e1", "acirc" => "\u00e2",
    "atilde" => "\u00e3", "auml"   => "\u00e4", "aring" => "\u00e5",
    "aelig"  => "\u00e6",
    "ccedil" => "\u00e7",
    "egrave" => "\u00e8", "eacute" => "\u00e9", "ecirc" => "\u00ea",
    "euml"   => "\u00eb",
    "igrave" => "\u00ec", "iacute" => "\u00ed", "icirc" => "\u00ee",
    "iuml"   => "\u00ef",
    "eth"    => "\u00f0",
    "ntilde" => "\u00f1",
    "ograve" => "\u00f2", "oacute" => "\u00f3", "ocirc" => "\u00f4",
    "otilde" => "\u00f5", "ouml"   => "\u00f6", "oslash" => "\u00f8",
    "ugrave" => "\u00f9", "uacute" => "\u00fa", "ucirc" => "\u00fb",
    "uuml"   => "\u00fc",
    "yacute" => "\u00fd",
    "thorn"  => "\u00fe",
    "szlig"  => "\u00df",
    "Agrave" => "\u00c0", "Aacute" => "\u00c1", "Acirc" => "\u00c2",
    "Atilde" => "\u00c3", "Auml"   => "\u00c4", "Aring" => "\u00c5",
    "AElig"  => "\u00c6",
    "Ccedil" => "\u00c7",
    "Egrave" => "\u00c8", "Eacute" => "\u00c9", "Ecirc" => "\u00ca",
    "Euml"   => "\u00cb",
    "Igrave" => "\u00cc", "Iacute" => "\u00cd", "Icirc" => "\u00ce",
    "Iuml"   => "\u00cf",
    "ETH"    => "\u00d0",
    "Ntilde" => "\u00d1",
    "Ograve" => "\u00d2", "Oacute" => "\u00d3", "Ocirc" => "\u00d4",
    "Otilde" => "\u00d5", "Ouml"   => "\u00d6", "Oslash" => "\u00d8",
    "Ugrave" => "\u00d9", "Uacute" => "\u00da", "Ucirc" => "\u00db",
    "Uuml"   => "\u00dc",
    "Yacute" => "\u00dd",
    "THORN"  => "\u00de",
)

"""
    decode_html_entities(text::String) -> String

Replace HTML entities (named, decimal, hex) with their character equivalents.
"""
function decode_html_entities(text::String)
    # Numeric entities: &#8212; or &#x2014;
    result = replace(text, r"&#x([0-9a-fA-F]+);" => function(m)
        hex = match(r"&#x([0-9a-fA-F]+);", m)
        if hex !== nothing
            code = parse(Int, hex.captures[1], base=16)
            return string(Char(code))
        end
        return m
    end)
    result = replace(result, r"&#(\d+);" => function(m)
        dec = match(r"&#(\d+);", m)
        if dec !== nothing
            code = parse(Int, dec.captures[1])
            if code > 0 && code <= 0x10FFFF
                return string(Char(code))
            end
        end
        return m
    end)
    # Named entities: &amp; &nbsp; etc.
    result = replace(result, r"&([a-zA-Z]+);" => function(m)
        name_match = match(r"&([a-zA-Z]+);", m)
        if name_match !== nothing
            name = name_match.captures[1]
            if haskey(NAMED_ENTITIES, name)
                return NAMED_ENTITIES[name]
            end
            # Try lowercase lookup
            if haskey(NAMED_ENTITIES, lowercase(name))
                return NAMED_ENTITIES[lowercase(name)]
            end
        end
        return " "  # unknown entity -> space
    end)
    return result
end

# ----------------------------------------------------------------
# HTML stripping
# ----------------------------------------------------------------

"""
    strip_html(text::String) -> String

Remove all HTML from the raw file:
1. Remove <head>, <script>, <style> blocks entirely (well-formed closing tags)
2. Remove footnote markers (<sup> tags)
3. Remove footnote div sections by id
4. Insert newlines at block boundaries
5. Strip all remaining HTML tags
6. Decode HTML entities

NOTE: We do NOT try to match <p class="X">...</p> patterns because HTML <p>
tags are often not properly closed, causing catastrophic regex matching.
Instead, boilerplate is removed at the text level after stripping.
"""
function strip_html(text::String)
    # Remove entire <head>...</head> blocks (metadata, CSS, etc.)
    text = replace(text, r"<head\b[^>]*>.*?</head>"si => " ")

    # Remove <script>...</script> blocks
    text = replace(text, r"<script\b[^>]*>.*?</script>"si => " ")

    # Remove <style>...</style> blocks
    text = replace(text, r"<style\b[^>]*>.*?</style>"si => " ")

    # Remove HTML/XML declarations and DOCTYPE
    text = replace(text, r"<\?xml[^>]*\?>" => "")
    text = replace(text, r"<!DOCTYPE[^>]*>" => "")

    # Remove CDATA blocks
    text = replace(text, r"<!\[CDATA\[.*?\]\]>"s => "")

    # Remove HTML comments
    text = replace(text, r"<!--.*?-->"s => " ")

    # Remove footnote markers: <sup class="enote">...</sup> and similar
    text = replace(text, r"<sup\b[^>]*class\s*=\s*\"enote\"[^>]*>.*?</sup>"si => "")
    text = replace(text, r"<sup\b[^>]*>.*?</sup>"si => "")

    # Remove footnote anchor links embedded in <a> tags (self-contained)
    text = replace(text, r"<a\s+class\s*=\s*\"sdfootnoteanc\"[^>]*>\[?\d+\]?</a>"si => "")
    text = replace(text, r"<a\s+[^>]*name\s*=\s*\"sdfootnote\d+anc\"[^>]*>[^<]*</a>"si => "")

    # Remove entire footnote div sections at the bottom of pages
    text = replace(text, r"<div\s+id\s*=\s*\"sdfootnote\d+\"[^>]*>.*?</div>"si => "")

    # Remove <hr> elements (section separators)
    text = replace(text, r"<hr\b[^>]*/?\s*>"si => "\n")

    # Insert newlines before block-level elements
    text = replace(text, r"<(p|div|blockquote|table|tr|ul|ol|li|h[1-6]|dt|dd)\b"si => function(m)
        return "\n" * m
    end)

    # Insert newlines for <br> tags
    text = replace(text, r"<br\s*/?\s*>"si => "\n")

    # Strip all remaining HTML tags
    text = replace(text, r"<[^>]+>" => " ")

    # Decode HTML entities
    text = decode_html_entities(text)

    return text
end

# ----------------------------------------------------------------
# Preface removal
# ----------------------------------------------------------------

"""
    remove_prefaces(text::String) -> String

Remove preface sections that appear before the actual manifesto text.
The manifesto starts with the famous "A spectre..." preamble in each language.
We detect this and discard everything before it.
"""
function remove_prefaces(text::String)
    # The preamble opening in each language
    preamble_patterns = [
        r"Ein Gespenst geht um"i,                    # German
        r"A spectre is haunting"i,                    # English
        r"Un spectre hante"i,                         # French
        r"Uno spettro s.aggira"i,                     # Italian
        r"Un espectro se cierne"i,                    # Spanish
        r"Anda um espectro"i,                         # Portuguese (new translation)
        r"Um espectro ronda"i,                        # Portuguese (alt)
        r"Widmo kr..y po Europie"i,                   # Polish
        r"Aave kummittelee"i,                         # Finnish
        r"Een spook waart"i,                          # Dutch
        r"Evropou obch.z. stra"i,                     # Czech
    ]

    for pat in preamble_patterns
        m = match(pat, text)
        if m !== nothing
            # Find the start of the line containing the preamble
            idx = m.offset
            while idx > 1 && text[idx - 1] != '\n'
                idx -= 1
            end
            return text[idx:end]
        end
    end

    # If no preamble found, return as-is
    return text
end

# ----------------------------------------------------------------
# Section heading removal
# ----------------------------------------------------------------

"""
    remove_section_headings(text::String) -> String

Remove chapter/section headings like:
- "I. Bourgeois and Proletarians"
- "II. Proletarier und Kommunisten"
- Roman numeral section headers
- Chapter titles in various languages
"""
function remove_section_headings(text::String)
    lines = split(text, '\n')
    filtered = String[]
    for line in lines
        stripped = strip(line)
        # Skip lines that are purely Roman numeral section headers
        # Pattern: optional "Chapter" or equivalent, then Roman numerals, then title
        if occursin(r"^\s*(chapter|chapitre|capitolo|capitulo|cap[ií]tulo|kapitel|hoofdstuk|luku|rozd[zí][ae]l|kapitola)\s+"i, stripped)
            continue
        end
        # Lines that are just Roman numerals + period + optional title
        if occursin(r"^\s*[IVXivx]+\s*[\.\):\-]\s*\S"i, stripped) && length(stripped) < 200
            # Check it's actually a heading (short line with Roman numeral prefix)
            if length(stripped) < 150
                continue
            end
        end
        # Lines that are just Roman numerals alone
        if occursin(r"^\s*[IVX]+\.?\s*$", stripped)
            continue
        end
        # Numbered subsection headers: "1. Der reaktionare Sozialismus" etc.
        if occursin(r"^\s*\d+\.\s+[A-Z]", stripped) && length(stripped) < 120
            # Only skip if it looks like a short heading, not a paragraph starting with a number
            if !occursin(r"\.\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+", stripped)
                continue
            end
        end
        # Lettered subsection headers: "a) Der feudale Sozialismus"
        if occursin(r"^\s*[a-c]\)\s+[A-Z]", stripped) && length(stripped) < 120
            continue
        end
        push!(filtered, string(line))
    end
    return join(filtered, '\n')
end

# ----------------------------------------------------------------
# Footnote section removal
# ----------------------------------------------------------------

"""
    remove_footnote_sections(text::String) -> String

Remove footnote sections and inline footnote markers.
Strategy:
1. Find the end of the manifesto proper (the "Workers of all countries, unite!" line)
2. Truncate everything after the final manifesto line (+ a small buffer)
3. Remove inline footnote markers [1], (1), [a] and editorial annotations
"""
function remove_footnote_sections(text::String)
    # The manifesto always ends with a variation of
    # "Proletarier aller Lander, vereinigt euch!" / "Workers of all countries, unite!"
    closing_patterns = [
        r"Proletarier aller L.nder,?\s*vereinigt\s+[Ee]uch"i,       # German
        r"Working\s+[Mm]en\s+of\s+[Aa]ll\s+[Cc]ountries,?\s*[Uu]nite"i, # English
        r"Prol.taires\s+de\s+tous\s+les\s+pays,?\s*unissez.vous"i,  # French
        r"Proletari\s+di\s+tutti\s+i\s+paesi,?\s*unitevi"i,         # Italian
        r"Proletarios\s+de\s+todos\s+los\s+pa.ses,?\s*un.os"i,      # Spanish
        r"Prolet.rios\s+de\s+[Tt]odos\s+os\s+pa.ses,?\s*[Uu]ni.vos"i, # Portuguese
        r"Proletariusze\s+wszystkich\s+kraj.w,?\s*..czcie\s+si"i,   # Polish
        r"Kaikkien\s+maiden\s+proletaarit,?\s*liittyk..\s+yhteen"i, # Finnish
        r"Proletari.rs\s+aller\s+landen,?\s*verenigt\s+[Uu]"i,      # Dutch
        r"Prolet..i\s+v.ech\s+zem.,?\s*spojte\s+se"i,               # Czech
    ]

    last_closing_pos = 0
    for pat in closing_patterns
        for m in eachmatch(pat, text)
            # m.offset is byte-based, so use sizeof for byte length of match
            pos = m.offset + sizeof(m.match)
            if pos > last_closing_pos
                last_closing_pos = pos
            end
        end
    end

    if last_closing_pos > 0
        # Find end of the line containing the closing call
        # NOTE: m.offset and match lengths are byte-based, use sizeof for bounds
        text_bytes = sizeof(text)
        end_pos = last_closing_pos
        while end_pos <= text_bytes
            try
                if text[end_pos] == '\n'
                    break
                end
                end_pos = nextind(text, end_pos)
            catch
                break
            end
        end
        # Keep up to a couple lines after (in case of minor formatting)
        extra_newlines = 0
        while end_pos <= text_bytes && extra_newlines < 3
            try
                c = text[end_pos]
                if c == '\n'
                    extra_newlines += 1
                elseif !isspace(c)
                    break
                end
                end_pos = nextind(text, end_pos)
            catch
                break
            end
        end
        text = text[1:min(end_pos - 1, text_bytes)]
    end

    # Remove inline footnote markers [1], (1) etc.
    text = replace(text, r"\[\d+\]" => "")
    text = replace(text, r"\(\d+\)" => "")

    # Remove editorial annotations in brackets
    text = replace(text, r"\[Engels[^\]]*\]" => "")
    text = replace(text, r"\[Note by [^\]]*\]" => "")
    text = replace(text, r"\[Marx[^\]]*\]" => "")

    # Remove per-chapter footnote sections (between chapters in concatenated files)
    lines = split(text, '\n')
    result = String[]
    in_footnotes = false

    for line in lines
        stripped = strip(line)

        # Detect start of footnote section header
        if occursin(r"^\s*(Notes?|Footnotes?|Anmerkungen|Pozn[aá]mk[iy]|Notas?\s+de\s+fim)\s*:?\s*$"i, stripped)
            in_footnotes = true
            continue
        end

        # Lines starting with footnote markers like [1], (1), [a], (a)
        if occursin(r"^\s*[\[\(]\s*(\d+|[a-z])\s*[\]\)]", stripped) && length(stripped) < 300
            in_footnotes = true
            continue
        end

        if in_footnotes
            if isempty(stripped)
                # Blank line in footnotes -- check if next content looks like main text
                continue
            end
            # If the line looks like the start of a new chapter (main text), exit footnotes
            if length(stripped) > 200 || occursin(r"^[A-Z][a-z]+ [a-z]+ [a-z]+ [a-z]+ [a-z]+"i, stripped)
                in_footnotes = false
                push!(result, string(line))
            end
            # Otherwise skip (still in footnotes)
            continue
        end

        push!(result, string(line))
    end

    return join(result, '\n')
end

# ----------------------------------------------------------------
# Marxists.org template removal
# ----------------------------------------------------------------

"""
    remove_site_boilerplate(text::String) -> String

Remove marxists.org site navigation, metadata blocks, and editorial notes.
Works on the text after HTML tag stripping.
"""
function remove_site_boilerplate(text::String)
    lines = split(text, '\n')
    filtered = String[]

    for line in lines
        stripped = strip(line)

        # Skip navigation breadcrumbs
        occursin(r"^MIA\s*>"i, stripped) && continue
        occursin(r"^MIA\s*:?\s*(Marxists|Deutsch|Fran)"i, stripped) && continue
        occursin(r"Marxists\s*Internet\s*Archive"i, stripped) && continue
        occursin(r"marxists\.org"i, stripped) && continue
        occursin(r"Marxistick.*archiv"i, stripped) && continue
        occursin(r"Archivio Marx"i, stripped) && continue
        occursin(r"Arquivo Marxista"i, stripped) && continue
        occursin(r"Archivo.*castellano"i, stripped) && continue

        # Skip meta tag content that leaked through
        occursin(r"^meta\s+name"i, stripped) && continue
        occursin(r"^content\s*="i, stripped) && continue

        # Skip lines that are just CSS class names or HTML artifacts
        occursin(r"^\s*class\s*=\s*\""i, stripped) && length(stripped) < 50 && continue

        # Skip metadata lines
        occursin(r"^\s*(Written|First Published|Source|Translated|Transcription|Proofed|Copyleft|Download|Geschreven|Eerste publicatie|Bron|Proeflezen|Transcriptie)\s*:"i, stripped) && continue
        occursin(r"^\s*(Kirjoitettu|Julkaistu|Suomentaja|L.hde|Skannaus|HTML)\s*:"i, stripped) && continue
        occursin(r"^\s*(Escrito|Publicado|Origem|Tradu..o|Direitos)\s*:"i, stripped) && continue
        occursin(r"^\s*(Napsali|Poprv|Podle)\s+"i, stripped) && continue

        # Skip "last updated" lines
        occursin(r"(Zuletzt aktualisiert|Last updated|Dernière modification|Ultima modifica|Laatst bijgewerkt|Inclus.o|atualiza..o)"i, stripped) && continue

        # Skip "beginning of page" / "top of page" links
        occursin(r"(Anfang der Seite|Inizio pagina|Inicio de|D.but de page|Begin pagina)"i, stripped) && continue

        # Skip page navigation links like "[German Original]"
        occursin(r"^\s*\[.*Original\]\s*$"i, stripped) && continue

        # Skip table of contents lines
        occursin(r"^\s*(Contents|Inhalt|Inhoudsopgave|Sis.llysluettelo|Obsah|.ndice|Sommaire|Indice)\s*$"i, stripped) && continue

        # Skip "* * *" separator lines
        occursin(r"^\s*(\*\s+)+\*\s*$", stripped) && continue
        occursin(r"^_{3,}$", stripped) && continue

        # Skip editorial notes about translations
        occursin(r"Engels,\s*(English|German)\s+Edition"i, stripped) && continue
        occursin(r"famous final phrase"i, stripped) && continue
        occursin(r"more correct translation"i, stripped) && continue
        occursin(r"Workers of the World"i, stripped) && continue
        occursin(r"is a popularisation"i, stripped) && continue
        occursin(r"approved by Engels"i, stripped) && continue

        # Skip empty "photograph" references
        occursin(r"Photograph of a page"i, stripped) && continue

        # Skip "Table of Contents: Manifesto" references
        occursin(r"Table of Contents.*Manifesto"i, stripped) && continue
        occursin(r"Marx/Engels\s*(Library|Archive|Archief|archiv)"i, stripped) && continue

        # Skip lines that are just titles/headers for the work itself
        occursin(r"^\s*Manifest\s+(der|komunistick|do Partido)"i, stripped) && length(stripped) < 80 && continue
        occursin(r"^\s*Manifesto\s+(of the|del Partido|do Partido|del Partito)"i, stripped) && length(stripped) < 80 && continue
        occursin(r"^\s*Communistisch\s+Manifest\s*$"i, stripped) && continue
        occursin(r"^\s*Kommunistisen\s+puolueen\s+manifesti\s*$"i, stripped) && continue
        occursin(r"^\s*Le\s+manifeste\s+du\s+Parti\s+communiste\s*$"i, stripped) && continue

        # Skip author attribution lines at top
        occursin(r"^\s*Karl\s+Marx\s+(u\.|en|und|and|e|et|a|ja|i)\s+Friedrich\s+Engels\s*$"i, stripped) && continue
        occursin(r"^\s*Karel\s+Marx\s+a\s+Bed.ich\s+Engels\s*$"i, stripped) && continue
        occursin(r"^\s*Karl\s+Marx\s*&\s*Friedrich\s+Engels\s*$"i, stripped) && continue

        # Skip year lines
        occursin(r"^\s*\(?\s*(Februar\s+)?1848\s*\)?\s*$"i, stripped) && continue
        occursin(r"^\s*\(?1848\)?\s*$", stripped) && continue

        # Skip publication/editorial info blocks
        occursin(r"Dietz Verlag"i, stripped) && continue
        occursin(r"^Dieser Text wurde"i, stripped) && continue
        occursin(r"HTML-Markierung"i, stripped) && continue
        occursin(r"Progress Publishers"i, stripped) && continue
        occursin(r"Samuel Moore"i, stripped) && continue
        occursin(r"^Kustannusliike"i, stripped) && continue
        occursin(r"^Editorial.*Avante"i, stripped) && continue

        # Skip preface/prefazioni references in TOC and preface titles
        occursin(r"^\s*(Prefaz|Pr[eé]face|Esipuhe|Przedmowa|P[rř]edmluv|Voorwoord|Preface)\b"i, stripped) && length(stripped) < 120 && continue

        # Skip prologue/preface headers (various editions)
        occursin(r"^\s*Pr[oó]logo\s+(de|a\s+la)"i, stripped) && continue
        occursin(r"^\s*Edici[oó]n\s+(alemana|inglesa|rusa|italiana|polaca|francesa)"i, stripped) && continue

        # Skip study guide references
        occursin(r"Study Guide"i, stripped) && continue

        # Skip "Over deze versie" (about this version) blocks
        occursin(r"^(Over deze versie|Deze versie)"i, stripped) && continue
        occursin(r"Maarten Vanheuverswyn"i, stripped) && continue
        occursin(r"copyleft.*Overname"i, stripped) && continue

        # Skip audio/PDF download lines
        occursin(r"(PDF-bestand|EPUB-bestand|Mobi-bestand|Audiolivro|eBook|audiobooks)"i, stripped) && continue

        # Skip "communist confession" type links
        occursin(r"Communist Confession"i, stripped) && continue
        occursin(r"Principles of Communism"i, stripped) && continue
        occursin(r"Demands of Communist"i, stripped) && continue

        # Skip volunteer/admin references
        occursin(r"volunteer.*MIA"i, stripped) && continue
        occursin(r"Admin Committee"i, stripped) && continue

        # Skip Creative Commons references
        occursin(r"Creative Commons"i, stripped) && continue

        # Skip "See Note in" references
        occursin(r"^See Note in"i, stripped) && continue

        # Skip alternative translation references
        occursin(r"Alternatieve vertaling"i, stripped) && continue

        # Skip "Volver al Archivo" (Spanish navigation)
        occursin(r"^Volver al"i, stripped) && continue

        # Skip "Digitalizado para" (Spanish transcription credits)
        occursin(r"^Digitalizado para"i, stripped) && continue
        occursin(r"^Retranscrito para"i, stripped) && continue

        # Skip Italian index reference
        occursin(r"^\s*\[\s*Indice\s+de"i, stripped) && continue

        # Skip "K. Marx" short author lines
        occursin(r"^\s*K\.\s*Marx"i, stripped) && length(stripped) < 80 && continue
        occursin(r"^\s*F\.\s*Engels"i, stripped) && length(stripped) < 80 && continue

        # Skip "Manifiesto del Partido" (Spanish title)
        occursin(r"^\s*Manifiesto\s+del\s+Partido"i, stripped) && length(stripped) < 80 && continue

        # Skip short year-in-parentheses lines
        occursin(r"^\s*\(\s*\d{4}\s*\)\s*$", stripped) && continue

        # Skip "Suomenkielinen" (Finnish archive reference)
        occursin(r"Suomenkielinen"i, stripped) && continue

        # Skip inter-chapter navigation (from multi-page concatenation)
        occursin(r"^\s*Marxisten\s*$"i, stripped) && continue
        occursin(r"^\s*Deutsch\s*$"i, stripped) && continue
        occursin(r"^\s*Marx/Engels\s*$"i, stripped) && continue
        occursin(r"^\s*Archief\s*$"i, stripped) && continue

        # Skip "Inleiding" (Dutch: Introduction as a heading)
        occursin(r"^\s*Inleiding\s*$"i, stripped) && continue

        # Skip Czech archive links
        occursin(r"^Obsah \d+\. d"i, stripped) && continue
        occursin(r"^Hnut. roku"i, stripped) && continue
        occursin(r"^.e. o svobod"i, stripped) && continue

        # Skip "Biblioteca" and short navigation words
        occursin(r"^\s*Biblioteca\s*$"i, stripped) && continue
        occursin(r"^\s*Novidades\s*$"i, stripped) && continue
        occursin(r"^\s*Marx.?Engels\s*$"i, stripped) && continue

        # Skip edition-related preface text for Spanish
        occursin(r"^PROLOGOS DE MARX"i, stripped) && continue
        occursin(r"^EDICIONES DEL MANIFIESTO"i, stripped) && continue

        # Skip browser audio instructions (Dutch)
        occursin(r"wereldberoemde manifest"i, stripped) && continue
        occursin(r"browser.*Edge"i, stripped) && continue
        occursin(r"Read Aloud"i, stripped) && continue
        occursin(r"Add-on"i, stripped) && continue

        push!(filtered, string(line))
    end

    return join(filtered, '\n')
end

# ----------------------------------------------------------------
# Diacritics stripping
# ----------------------------------------------------------------

"""
    strip_diacritics(c::Char) -> String

Decompose a character via NFD and keep only the ASCII base letter(s),
discarding combining marks. Returns lowercase ASCII.
Special handling for ligatures: ae, oe, ss.
"""
function strip_diacritics(c::Char)
    c_lower = lowercase(c)
    # Handle common ligatures
    c_lower == '\u00e6' && return "ae"    # ae
    c_lower == '\u0153' && return "oe"    # oe
    c_lower == '\u00df' && return "ss"    # sharp s
    c_lower == '\u00f0' && return "d"     # eth
    c_lower == '\u00fe' && return "th"    # thorn
    c_lower == '\u0142' && return "l"     # Polish l with stroke
    c_lower == '\u0111' && return "d"     # d with stroke
    c_lower == '\u0127' && return "h"     # h with stroke
    c_lower == '\u0131' && return "i"     # dotless i

    # NFD decomposition: split into base + combining marks
    decomposed = Unicode.normalize(string(c_lower), :NFD)
    result = ""
    for dc in decomposed
        # Keep only ASCII letters (skip combining diacritical marks U+0300-U+036F)
        if 'a' <= dc <= 'z'
            result = result * string(dc)
        end
    end
    return result
end

# ----------------------------------------------------------------
# Main text cleaning
# ----------------------------------------------------------------

"""
    clean_text(text::String) -> String

Lowercase, strip diacritics, remove non-letter/non-whitespace characters,
and normalize whitespace while preserving paragraph breaks.
"""
function clean_text(text::String)
    out = IOBuffer()
    for c in text
        if c == '\n'
            write(out, '\n')
        elseif c == ' ' || c == '\t'
            write(out, ' ')
        elseif isletter(c)
            write(out, strip_diacritics(c))
        end
        # All other characters (digits, punctuation, etc.) are dropped
    end
    raw_cleaned = String(take!(out))

    # Normalize whitespace within lines
    lines = split(raw_cleaned, '\n')
    cleaned_lines = String[]
    for line in lines
        stripped = strip(replace(string(line), r" +" => " "))
        push!(cleaned_lines, string(stripped))
    end

    # Collapse runs of 3+ blank lines to 2
    result = join(cleaned_lines, '\n')
    result = replace(result, r"\n{3,}" => "\n\n")

    # Trim leading/trailing whitespace
    return strip(result)
end

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

function main()
    println("Cleaning Communist Manifesto texts...")
    println("Data directory: " * DATA_DIR)
    println()

    for (raw_name, clean_name) in FILES
        raw_path = joinpath(DATA_DIR, raw_name)
        clean_path = joinpath(DATA_DIR, clean_name)

        if !isfile(raw_path)
            println("WARNING: File not found, skipping: " * raw_path)
            continue
        end

        println("Processing: " * raw_name * " -> " * clean_name)

        # Read raw file (handle various encodings: UTF-8, Latin-1, Windows-1252)
        text = read_html_file(raw_path)

        # Step 1: Strip HTML
        text = strip_html(text)

        # Step 2: Remove prefaces (before the actual manifesto text)
        text = remove_prefaces(text)

        # Step 3: Remove site boilerplate
        text = remove_site_boilerplate(text)

        # Step 4: Remove section headings
        text = remove_section_headings(text)

        # Step 5: Remove footnote sections
        text = remove_footnote_sections(text)

        # Step 6: Clean text (lowercase, strip diacritics, remove non-letters)
        text = clean_text(text)

        # Write cleaned file
        write(clean_path, text * "\n")

        # Report stats
        char_count = length(text)
        word_count = length(split(text))
        line_count = count(==('\n'), text) + 1
        println("  Characters: " * string(char_count))
        println("  Words: " * string(word_count))
        println("  Lines: " * string(line_count))
        println("  First 5 non-empty lines:")
        nonempty = filter(!isempty, split(text, '\n'))
        for line in first(nonempty, 5)
            println("    " * string(first(line, 100)))
        end
        println()
    end

    # Summary table
    println("=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    for (raw_name, clean_name) in FILES
        clean_path = joinpath(DATA_DIR, clean_name)
        if isfile(clean_path)
            text = read(clean_path, String)
            chars = length(strip(text))
            lang = replace(replace(clean_name, "manifesto_" => ""), ".txt" => "")
            padded_lang = rpad(lang, 15)
            println("  " * padded_lang * string(chars) * " characters")
        end
    end
end

main()
