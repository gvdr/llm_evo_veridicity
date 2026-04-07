#!/usr/bin/env julia
#
# clean_dante.jl
#
# Cleans raw Divine Comedy translations (Gutenberg + Wikisource) into
# normalised plain-text suitable for cross-lingual comparison experiments.
#
# Steps per file:
#   1. Strip Gutenberg headers/footers (*** START / *** END markers)
#   2. Strip Wikisource markup ({{...}}, [[...]], <ref>...</ref>, etc.)
#   3. Remove canto/chapter headings (CANTO I, CHANT PREMIER, Gesang, etc.)
#   4. Remove section headings (INFERNO, PURGATORIO, PARADISO, etc.)
#   5. Remove illustration markers, image links
#   6. Remove translator prefaces, notes, appendices, TOC
#   7. Lowercase everything
#   8. Strip diacritics via Unicode NFD (e-acute -> e, u-umlaut -> u, etc.)
#      with special handling for ligatures (ae, oe, ss)
#   9. Remove all non-letter, non-whitespace characters
#  10. Collapse multiple blank lines
#  11. Save cleaned files as dante_LANG.txt

using Unicode

const DATA_DIR = joinpath(@__DIR__, "..", "data", "dante")

# Map of (raw_filename, cleaned_filename, source_type)
# source_type: :gutenberg, :gutenberg_concat, :wikisource_wikitext, :wikisource_html
const FILES = [
    ("dante_italian_raw.txt",    "dante_italian.txt",    :gutenberg),
    ("dante_english_raw.txt",    "dante_english.txt",    :gutenberg),
    ("dante_german_raw.txt",     "dante_german.txt",     :gutenberg),
    ("dante_finnish_raw.txt",    "dante_finnish.txt",    :gutenberg),
    ("dante_spanish_raw.txt",    "dante_spanish.txt",    :gutenberg),
    ("dante_french_raw.txt",     "dante_french.txt",     :gutenberg_concat),
    ("dante_portuguese_raw.txt", "dante_portuguese.txt", :wikisource_wikitext),
    ("dante_polish_raw.txt",     "dante_polish.txt",     :wikisource_html),
]

# ---------------------------------------------------------------------------
# Gutenberg header/footer stripping
# ---------------------------------------------------------------------------

"""
    strip_gutenberg(text) -> String

Remove everything before the first "*** START OF THE PROJECT GUTENBERG EBOOK"
line and everything from the last "*** END OF THE PROJECT GUTENBERG EBOOK"
onward. For concatenated files (French), handles multiple START/END pairs
by keeping the text between the first START and the last END, stripping
internal Gutenberg boilerplate between pairs.
"""
function strip_gutenberg(text::String; concatenated::Bool=false)
    lines = split(text, '\n')

    if concatenated
        # Find all START and END markers
        starts = Int[]
        ends = Int[]
        for (i, line) in enumerate(lines)
            if occursin(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK"i, line)
                push!(starts, i)
            end
            if occursin(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"i, line)
                push!(ends, i)
            end
        end

        if isempty(starts) || isempty(ends)
            @warn "No START/END markers found in concatenated file"
            return text
        end

        # Build text from segments between each START..END pair
        segments = String[]
        for k in 1:min(length(starts), length(ends))
            seg_start = starts[k] + 1
            seg_end = ends[k] - 1
            if seg_start <= seg_end
                push!(segments, join(lines[seg_start:seg_end], '\n'))
            end
        end
        return join(segments, "\n\n")
    else
        # Single START/END pair
        start_idx = 0
        end_idx = length(lines)

        for (i, line) in enumerate(lines)
            if occursin(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK"i, line)
                start_idx = i
            end
            if occursin(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"i, line)
                end_idx = i - 1
                break
            end
        end

        if start_idx == 0
            @warn "No START marker found, using entire text"
            start_idx = 0
        end

        return join(lines[(start_idx + 1):end_idx], '\n')
    end
end

# ---------------------------------------------------------------------------
# Wikisource markup stripping
# ---------------------------------------------------------------------------

"""
    strip_wikisource_wikitext(text) -> String

Remove Wikisource wikitext markup: templates, image links, refs, categories,
poem tags, etc. Keep the actual poem text.
"""
function strip_wikisource_wikitext(text::String)
    out = text

    # Remove {{template...}} blocks (possibly multiline)
    # Greedy removal of balanced {{ }}
    out = replace(out, r"\{\{[^{}]*\}\}" => "")
    # Second pass for nested templates
    out = replace(out, r"\{\{[^{}]*\}\}" => "")
    # Third pass
    out = replace(out, r"\{\{[^{}]*\}\}" => "")

    # Remove ''italic text'' (used for canto summaries/arguments)
    out = replace(out, r"''([^']*)''" => "")

    # Remove [[Imagem:...]], [[Image:...]], [[File:...]] (possibly multiline)
    out = replace(out, r"\[\[(Imagem|Image|Arquivo|File|Ficheiro):[^\]]*\]\]"i => "")

    # Remove [[Categoria:...]], [[Category:...]]
    out = replace(out, r"\[\[(Categoria|Category|Kategoria):[^\]]*\]\]"i => "")

    # Convert [[link|display text]] to display text
    out = replace(out, r"\[\[[^\]]*\|([^\]]*)\]\]" => s"\1")
    # Convert [[link]] to link text
    out = replace(out, r"\[\[([^\]]*)\]\]" => s"\1")

    # Remove <ref>...</ref> (possibly multiline)
    out = replace(out, r"<ref[^>]*>.*?</ref>"s => "")
    out = replace(out, r"<ref[^>]*/>" => "")

    # Remove <references .../> and <references>...</references>
    out = replace(out, r"<references\s*/>" => "")
    out = replace(out, r"<references>.*?</references>"s => "")

    # Remove <poem> and </poem> tags (keep content)
    out = replace(out, r"</?poem>" => "")

    # Remove other HTML tags
    out = replace(out, r"<[^>]+>" => "")

    # Remove &mdash; &ndash; and other HTML entities
    out = replace(out, r"&[a-zA-Z]+;" => " ")
    out = replace(out, r"&#\d+;" => " ")

    # Remove == Section == headings
    out = replace(out, r"^==+[^=]+=+\s*$"m => "")

    # Remove lines that are just '*' bullet list items from TOC
    out = replace(out, r"^\*\s.*$"m => "")

    # Remove lines starting with | (template parameters)
    out = replace(out, r"^\|.*$"m => "")

    return out
end

"""
    strip_wikisource_html(text) -> String

Clean rendered Wikisource HTML that has been converted to plain text.
Strips navigation headers, page numbers, translator attributions, etc.
Note: HTML tags and entities have already been stripped by the download script.
"""
function strip_wikisource_html(text::String)
    lines = split(text, '\n')
    filtered = String[]
    for line in lines
        stripped = strip(line)

        # Skip navigation arrows and breadcrumb lines
        occursin(r"^.*Boska Komedia.*Dante Alighieri"i, stripped) && continue
        occursin(r"^.*Boska Komedia\s*$"i, stripped) && continue

        # Skip navigation arrows: lines like "←Piekło" or "Pieśń II→"
        occursin(r"^.*Pie.{1,3}\s+[IVXLCDM]+.*$"i, stripped) && length(stripped) < 30 && continue
        occursin(r"^.*Piek.o$"i, stripped) && length(stripped) < 20 && continue
        occursin(r"^.*Czy.ciec$"i, stripped) && length(stripped) < 20 && continue
        occursin(r"^.*Raj$"i, stripped) && length(stripped) < 10 && continue

        # Skip translator attribution
        occursin(r"^Przek.ad:.*Stanis.awski"i, stripped) && continue

        # Skip page numbers like [1], [2], etc. at start of lines
        line_clean = replace(stripped, r"^\[\d+\]" => "")

        # Skip lines that are just page numbers
        occursin(r"^\[\d+\]\s*$", stripped) && continue

        # Skip PIESN headings (they'll also be caught by strip_headings)
        occursin(r"^PIE.{1,3}\s+(PIERWSZA|DRUGA|TRZECIA|CZWARTA|PIĄTA|SZÓSTA|SIÓDMA|ÓSMA|DZIEWIĄTA|DZIESIĄTA)"i, stripped) && continue
        occursin(r"^PIE.{1,3}\s+\d+"i, stripped) && continue

        push!(filtered, string(line_clean))
    end
    return join(filtered, '\n')
end

# ---------------------------------------------------------------------------
# Preface / TOC / Notes removal
# ---------------------------------------------------------------------------

"""
    strip_preface_and_toc(text, lang) -> String

Remove translator prefaces, table of contents, and end-matter notes.
Language-specific patterns.
"""
function strip_preface_and_toc(text::String, lang::Symbol)
    lines = split(text, '\n')

    # --- Italian: skip TOC (Contents section) and title block ---
    if lang == :italian
        body_start = 1
        # Skip everything before first actual verse line
        # The TOC has lines like " Canto I. " and section headers
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # First canto starts with "Nel mezzo del cammin"
            if occursin(r"^Nel mezzo del cammin"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]
    end

    # --- English: skip TOC and title block ---
    if lang == :english
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            if occursin(r"^Midway upon the journey of our life"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]
    end

    # --- German: skip preamble, version notes, TOC ---
    if lang == :german
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # First canto starts with "Auf halbem Weg"
            if occursin(r"^Auf halbem Weg des Menschenlebens"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]
    end

    # --- Finnish: skip preface and remove VIITESELITYKSET (notes) sections ---
    if lang == :finnish
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # First verse: "Elomme vaelluksen keskitiessa"
            if occursin(r"^Elomme vaelluksen"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]

        # Remove VIITESELITYKSET (footnotes/endnotes) sections
        # These appear between each cantica. Also remove edition notes.
        filtered = String[]
        in_notes = false
        for line in lines
            stripped = strip(line)

            # Enter notes section
            if occursin(r"^VIITESELITYKSET"i, stripped)
                in_notes = true
                continue
            end

            # Exit notes when we hit a section header or verse text
            if in_notes
                # Edition notes like "Ensimmainen painos ilmestyi 1913"
                if occursin(r"^Ensimm.inen painos"i, stripped)
                    continue
                end
                # Section headers bring us back to poem
                if occursin(r"^JUMALAINEN"i, stripped)
                    in_notes = false
                    # Skip this header line (will be caught by strip_headings)
                    push!(filtered, string(line))
                    continue
                end
                # Numbered note lines [1], [2], ...
                if occursin(r"^\[\d+\]", stripped)
                    continue
                end
                # Laulu heading (individual song header like "1. laulu")
                if occursin(r"^\d+\.\s*laulu"i, stripped)
                    continue
                end
                # Blank lines in notes
                if isempty(stripped)
                    continue
                end
                # Continuation of note text
                continue
            end

            push!(filtered, string(line))
        end
        lines = filtered
    end

    # --- Spanish: skip "Nota del Transcriptor", intro essay, etc. ---
    if lang == :spanish
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # First verse: "A la mitad del viaje de nuestra vida"
            if occursin(r"^A la mitad del viaje"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]

        # Remove end matter: FIN, _INDICE_ section at the end
        body_end = length(lines)
        for i in length(lines):-1:max(1, length(lines) - 500)
            stripped = strip(lines[i])
            if occursin(r"^_?INDICE_?$"i, stripped) ||
               occursin(r"^_?ÍNDICE_?$"i, stripped) ||
               occursin(r"^FIN$"i, stripped) ||
               occursin(r"^NOTAS$"i, stripped)
                body_end = i - 1
                break
            end
        end
        # Also strip trailing blank lines
        while body_end > 1 && isempty(strip(lines[body_end]))
            body_end -= 1
        end
        lines = lines[1:body_end]
    end

    # --- French: skip translator preface, find first verse ---
    # Also remove interleaved NOTES sections (footnote expansions after each canto),
    # ARGUMENT blocks, and TABLE DES MATIERES.
    if lang == :french
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # Rivarol translation starts: "J'etais au milieu de ma course"
            if occursin(r"^J'.tais au milieu"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]

        # State machine to strip non-poem sections
        # States: :poem, :notes, :argument, :table
        filtered = String[]
        state = :poem
        for line in lines
            stripped = strip(line)

            # Transition into NOTES section
            if occursin(r"^\s*NOTES\s*$", stripped)
                state = :notes
                continue
            end

            # Transition into TABLE DES MATIERES
            if occursin(r"^\s*TABLE DES"i, stripped)
                state = :table
                continue
            end

            # Transition into ARGUMENT block
            if occursin(r"^\s*ARGUMENT\s*$"i, stripped)
                state = :argument
                continue
            end

            # In notes/table/argument: skip everything until we detect
            # the start of actual poem text. Poem text resumes after
            # the ARGUMENT's indented summary, at the first non-indented
            # line that doesn't look like a footnote or heading.
            if state == :notes || state == :table
                # Stay in notes/table -- skip all content
                # But check if we hit a CHANT heading (which precedes ARGUMENT)
                if occursin(r"^CHANT\s"i, stripped)
                    # This is a canto heading, skip it, stay in non-poem state
                    continue
                end
                continue
            end

            if state == :argument
                # Argument blocks are indented summary text
                # They end when we hit regular (non-indented) text
                if isempty(stripped)
                    continue
                end
                # Argument text is typically indented with 2+ spaces
                if startswith(line, "  ") || startswith(line, "\t")
                    # Still in argument
                    continue
                end
                # Also skip heading-like lines
                if occursin(r"^CHANT\s"i, stripped) || occursin(r"^SUR LE"i, stripped)
                    continue
                end
                # We've reached actual poem text
                state = :poem
                push!(filtered, string(line))
                continue
            end

            # In poem state: keep the line
            if state == :poem
                push!(filtered, string(line))
            end
        end
        lines = filtered

        # Strip printer colophon at end
        body_end = length(lines)
        for i in length(lines):-1:max(1, length(lines) - 50)
            stripped = strip(lines[i])
            if occursin(r"^Paris.*Imprimerie"i, stripped) ||
               occursin(r"^Imprimerie"i, stripped)
                body_end = i - 1
                break
            end
        end
        lines = lines[1:body_end]
    end

    # --- Portuguese: skip to first verse ---
    if lang == :portuguese
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # First verse: "Da nossa vida, em meio da jornada"
            if occursin(r"^Da nossa vida"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]

        # Remove "== Notas ==" sections at end of each canto
        filtered = String[]
        in_notes = false
        for line in lines
            stripped = strip(line)
            if occursin(r"^Notas\s*$"i, stripped)
                in_notes = true
                continue
            end
            if in_notes
                # Stay in notes until we hit the next canto's content
                # (which starts with <poem> content after navegar template is stripped)
                if !isempty(stripped) && !startswith(stripped, "[") &&
                   !occursin(r"^Categoria"i, stripped) &&
                   !occursin(r"^Inferno|^Purgat|^Para"i, stripped) &&
                   length(stripped) > 40 &&
                   !occursin(r"^\d+", stripped)
                    in_notes = false
                    push!(filtered, string(line))
                end
                continue
            end
            push!(filtered, string(line))
        end
        lines = filtered
    end

    # --- Polish: skip navigation headers to first verse ---
    if lang == :polish
        body_start = 1
        for (i, line) in enumerate(lines)
            stripped = strip(line)
            # First verse: "W polowie drogi naszego zywota"
            if occursin(r"^W po.owie drogi"i, stripped)
                body_start = i
                break
            end
        end
        lines = lines[body_start:end]
    end

    return join(lines, '\n')
end

# ---------------------------------------------------------------------------
# Canto / section heading removal
# ---------------------------------------------------------------------------

"""
    strip_headings(text) -> String

Remove canto headings, section headers (INFERNO, etc.), and similar
structural markers across all languages.
"""
function strip_headings(text::String)
    lines = split(text, '\n')
    filtered = String[]
    for line in lines
        stripped = strip(line)

        # Skip empty lines (keep them -- will be collapsed later)
        if isempty(stripped)
            push!(filtered, "")
            continue
        end

        # --- Canto headings ---
        # Italian: "Canto I", "Canto XXXIII"
        occursin(r"^Canto\s+[IVXLCDM]+\.?\s*$"i, stripped) && continue
        # Also "Inferno\nCanto I" pattern: just "Inferno" or "Purgatorio" or "Paradiso" alone
        occursin(r"^(Inferno|Purgatorio|Paradiso)\s*$"i, stripped) && continue

        # English: "Canto I", "CANTO I"
        # (already caught by Italian pattern above)

        # French: "CHANT PREMIER.--..." or "CHANT I." to "CHANT XXXIV."
        # These are long lines with descriptions; match the CHANT prefix
        occursin(r"^CHANT\s+(PREMIER|DEUXIÈME|TROISIÈME|[IVXLCDM]+)"i, stripped) && continue

        # German: "Erster Gesang", "Zweiter Gesang", ..., "Dreiunddreißigster Gesang"
        occursin(r"Gesang$"i, stripped) && continue
        # Also match numbered patterns like "1. Gesang" just in case
        occursin(r"^\d+\.\s*Gesang"i, stripped) && continue

        # Finnish: "Ensimmainen laulu", "Toinen laulu", etc.
        occursin(r"laulu\s*$"i, stripped) && continue

        # Spanish: "_CANTO PRIMERO_", "_CANTO SEGUNDO_", etc.
        occursin(r"^_?CANTO\s+"i, stripped) && continue

        # Portuguese: "Canto I", "CANTO I" (in Wikisource content)
        # (already caught by Italian pattern)

        # Polish: "PIESN PIERWSZA.", "PIESN DRUGA." etc. (all ordinal forms)
        occursin(r"^PIE.N\s"i, stripped) && continue

        # --- Section headings ---
        # Italian/English: INFERNO, PURGATORIO, PARADISO (alone on a line)
        occursin(r"^(INFERNO|PURGATORIO|PARADISO)\s*$"i, stripped) && continue

        # Spanish: _INFIERNO_, _PURGATORIO_, _PARAISO_
        occursin(r"^_?(INFIERNO|PURGATORIO|PARAISO|PARAÍSO)_?\s*$"i, stripped) && continue

        # German: Die Hoelle, Das Fegefeuer, Das Paradies
        occursin(r"^(Die Hölle|Das Fegefeuer|Das Paradies)\s*$"i, stripped) && continue

        # Finnish: JUMALAINEN NAYTELMAN: HELVETTI / KIIRASTULI / PARATIISI
        occursin(r"^JUMALAINEN"i, stripped) && continue

        # French: L'ENFER (as section header)
        occursin(r"^L'ENFER\s*$"i, stripped) && continue

        # Portuguese: Inferno, Purgatório, Paraíso as section headers
        # (already caught above)

        # Polish: PIEKLO, CZYSCIEC, RAJ
        occursin(r"^(PIEKŁO|CZYŚCIEC|RAJ)\s*$"i, stripped) && continue

        # --- Illustration markers ---
        occursin(r"^\[Illustra"i, stripped) && continue
        occursin(r"^\[Grabado"i, stripped) && continue
        occursin(r"^\[Gravure"i, stripped) && continue
        occursin(r"^\[Ilustraci"i, stripped) && continue

        # --- Gutenberg "Produced by" credit lines ---
        occursin(r"^Produced by"i, stripped) && continue
        occursin(r"^E-text prepared by"i, stripped) && continue

        # --- Title lines (book title repeated) ---
        occursin(r"^LA DIVINA COMMEDIA\s*$"i, stripped) && continue
        occursin(r"^The Divine Comedy\s*$"i, stripped) && continue
        occursin(r"^Die Göttliche Komödie\s*$"i, stripped) && continue
        occursin(r"^La Divina Comedia\s*$"i, stripped) && continue
        occursin(r"^L'ENFER$"i, stripped) && continue
        occursin(r"^BOSKA KOMEDJA\s*$"i, stripped) && continue
        occursin(r"^A DIVINA COMÉDIA\s*$"i, stripped) && continue

        # --- Author attribution lines ---
        occursin(r"^di Dante Alighieri\s*$"i, stripped) && continue
        occursin(r"^of Dante Alighieri\s*$"i, stripped) && continue
        occursin(r"^Dante Alighieri\s*$"i, stripped) && continue
        occursin(r"^HENRY WADSWORTH LONGFELLOW\s*$"i, stripped) && continue
        occursin(r"^Translated by\s*$"i, stripped) && continue
        occursin(r"^Kirjoittanut Dante\s*$"i, stripped) && continue
        occursin(r"^Suomentanut Eino Leino\s*$"i, stripped) && continue

        # --- Misc structure ---
        # "Contents" line
        occursin(r"^Contents\s*$"i, stripped) && continue
        occursin(r"^Inhalt:?\s*$"i, stripped) && continue

        push!(filtered, string(line))
    end
    return join(filtered, '\n')
end

# ---------------------------------------------------------------------------
# Diacritics stripping
# ---------------------------------------------------------------------------

"""
    strip_diacritics(c::Char) -> String

Decompose a character via NFD and keep only ASCII base letters,
discarding combining marks. Returns lowercase ASCII.
Special handling for ligatures.
"""
function strip_diacritics(c::Char)
    c_lower = lowercase(c)
    # Handle common ligatures
    c_lower == '\u00e6' && return "ae"  # ae ligature
    c_lower == '\u0153' && return "oe"  # oe ligature
    c_lower == '\u00df' && return "ss"  # sharp s
    c_lower == '\u00f0' && return "d"   # eth
    c_lower == '\u00fe' && return "th"  # thorn
    c_lower == '\u0142' && return "l"   # Polish l-stroke
    c_lower == '\u0111' && return "d"   # d-stroke
    c_lower == '\u0131' && return "i"   # dotless i

    # NFD decomposition
    decomposed = Unicode.normalize(string(c_lower), :NFD)
    result = ""
    for dc in decomposed
        if 'a' <= dc <= 'z'
            result = result * string(dc)
        end
    end
    return result
end

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

"""
    clean_text(text) -> String

Lowercase, strip diacritics, remove non-letter/non-whitespace,
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
        # All other characters (digits, punctuation, etc.) dropped
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

    return strip(result)
end

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

const LANG_MAP = Dict(
    "italian" => :italian,
    "english" => :english,
    "german" => :german,
    "finnish" => :finnish,
    "spanish" => :spanish,
    "french" => :french,
    "portuguese" => :portuguese,
    "polish" => :polish,
)

function detect_lang(filename::String)
    for (key, sym) in LANG_MAP
        if occursin(key, filename)
            return sym
        end
    end
    return :unknown
end

function main()
    println("Cleaning Dante Divine Comedy translations")
    println("Data directory: " * DATA_DIR)
    println()

    for (raw_name, clean_name, source_type) in FILES
        raw_path = joinpath(DATA_DIR, raw_name)
        clean_path = joinpath(DATA_DIR, clean_name)

        if !isfile(raw_path)
            println("SKIP (not found): " * raw_name)
            println()
            continue
        end

        raw_size = filesize(raw_path)
        # Skip files that are too small (likely incomplete Wikisource downloads)
        if source_type == :wikisource_wikitext && raw_size < 50_000
            println("SKIP (too small, likely incomplete): " * raw_name * " (" * string(raw_size) * " bytes)")
            println("  Run: bash scripts/download_dante.sh to download Wikisource texts")
            println()
            continue
        end
        if source_type == :wikisource_html && raw_size < 50_000
            println("SKIP (too small, likely incomplete): " * raw_name * " (" * string(raw_size) * " bytes)")
            println("  Run: bash scripts/download_dante.sh to download Wikisource texts")
            println()
            continue
        end

        println("Processing: " * raw_name * " -> " * clean_name)

        text = read(raw_path, String)
        lang = detect_lang(raw_name)

        # Step 1: Source-specific stripping
        if source_type == :gutenberg
            text = strip_gutenberg(text; concatenated=false)
        elseif source_type == :gutenberg_concat
            text = strip_gutenberg(text; concatenated=true)
        elseif source_type == :wikisource_wikitext
            text = strip_wikisource_wikitext(text)
        elseif source_type == :wikisource_html
            text = strip_wikisource_html(text)
        end

        # Step 2: Strip prefaces, TOC, and end-matter
        text = strip_preface_and_toc(text, lang)

        # Step 3: Strip headings
        text = strip_headings(text)

        # Step 4: Clean text (lowercase, strip diacritics, etc.)
        text = clean_text(text)

        # Write cleaned file
        write(clean_path, text * "\n")

        # Report stats
        char_count = length(text)
        line_count = count(==('\n'), text) + 1
        word_count = length(split(text))
        println("  Characters: " * string(char_count))
        println("  Lines:      " * string(line_count))
        println("  Words:      " * string(word_count))
        println("  First 3 non-empty lines:")
        nonempty = filter(!isempty, split(text, '\n'))
        for line in first(nonempty, 3)
            println("    " * string(first(string(line), 80)))
        end
        println()
    end

    # Print summary table
    println("=" ^ 60)
    println("Summary of cleaned files:")
    println("-" ^ 60)
    for (_, clean_name, _) in FILES
        clean_path = joinpath(DATA_DIR, clean_name)
        if isfile(clean_path)
            sz = filesize(clean_path)
            text = read(clean_path, String)
            chars = length(text)
            println("  " * rpad(clean_name, 30) * string(chars) * " chars")
        else
            println("  " * rpad(clean_name, 30) * "(not created)")
        end
    end
    println("=" ^ 60)
end

main()
