#!/usr/bin/env julia
#
# reclean_alice.jl
#
# Re-cleans the alice_LANG.txt files to remove surviving Gutenberg artifacts:
# front matter, illustration markers, chapter headings, chapter titles,
# dedicatory poems, end markers, and other single-word artifact lines.
# Collapses multiple blank lines into single blank lines.

const DATA_DIR = joinpath(@__DIR__, "..", "data", "alice")

# ── Chapter titles per language (used both for TOC removal and in-body removal) ──

const EN_TITLES = [
    "down the rabbithole",
    "the pool of tears",
    "a caucusrace and a long tale",
    "the rabbit sends in a little bill",
    "advice from a caterpillar",
    "pig and pepper",
    "a mad teaparty",
    "the queens croquetground",
    "the mock turtles story",
    "the lobster quadrille",
    "who stole the tarts",
    "alices evidence",
]

const FR_TITLES = [
    "au fond du terrier",
    "la mare aux larmes",
    "la course cocasse",
    "lhabitation du lapin blanc",
    "conseils dune chenille",
    "porc et poivre",
    "un the de fous",
    "le croquet de la reine",
    "histoire de la faussetortue",
    "le quadrille de homards",
    "qui a vole les tartes",
    "deposition dalice",
]

const DE_TITLES = [
    "hinunter in den kaninchenbau",
    "der thranenpfuhl",
    "caucusrennen und was daraus wird",
    "die wohnung des kaninchens",
    "guter rath von einer raupe",
    "ferkel und pfeffer",
    "die tolle theegesellschaft",
    "das croquetfeld der konigin",
    "die geschichte der falschen schildkrote",
    "das hummerballet",
    "wer hat die kuchen gestohlen",
    "alice ist die klugste",
]

const IT_TITLES = [
    "giu nella conigliera",
    "lo stagno di lagrime",
    "corsa arruffata e racconto con la coda",
    "la casettina del coniglio",
    "consigli dun bruco",
    "porco e pepe",
    "un te di matti",
    "il croquet della regina",
    "storia della falsatestuggine",
    "la contraddanza de gamberi",
    "chi ha rubato le torte",
    "testimonianza dalice",
]

const FI_TITLES = [
    "alhaalla kaninpesassa",
    "kyynellammikko",
    "vaalikilpajuoksu",
    "kanin kodissa",
    "kaalimato antaa hyvan neuvon",
    "porsas ja pippuri",
    "hullu teeseura",
    "kuningattaren krokettikentta",
    "valekilpikonnan tarina",
    "meriayriaisen hyppy",
    "kuka varasti leivokset",
    "liisan todistus",
]

# ── Single-word / short artifact lines to remove ──

const ARTIFACT_LINES = Set([
    "contents",
    "table",
    "inhalt",
    "indice",
    "sisallys",
    "the end",
    "fin",
    "ende",
    "fine",
    "chapitre page",
    "cap page",
    "originally published in",
])

# ── Illustration markers ──

function is_illustration(line::AbstractString)
    s = strip(line)
    return s == "illustration" || s == "illustrazione"
end

# ── Chapter heading patterns ──

function is_chapter_heading(line::AbstractString, lang::AbstractString)
    s = strip(line)
    if lang == "english"
        return occursin(r"^chapter [ivxlc]+$", s)
    elseif lang == "french"
        return occursin(r"^chapitre ", s)
    elseif lang == "german"
        return occursin(r"kapitel$", s)
    elseif lang == "italian"
        return occursin(r"^capitolo [ivxlc]+$", s)
    end
    return false
end

# ── Build set of chapter title strings for a given language ──

function chapter_title_set(lang::AbstractString)
    if lang == "english"
        return Set(EN_TITLES)
    elseif lang == "french"
        return Set(FR_TITLES)
    elseif lang == "german"
        return Set(DE_TITLES)
    elseif lang == "italian"
        return Set(IT_TITLES)
    elseif lang == "finnish"
        return Set(FI_TITLES)
    end
    return Set{String}()
end

# ── Find the first line of actual story text ──
# Strategy: find the first chapter heading (not in the TOC area),
# skip it and the following title line, then start from the next non-empty line.

function find_story_start(lines::Vector{<:AbstractString}, lang::AbstractString)
    titles = chapter_title_set(lang)
    n = length(lines)

    if lang == "finnish"
        # Finnish: no chapter headings per se, just chapter title lines.
        # The TOC lists all titles in sequence (lines 11-22).
        # The first chapter title "alhaalla kaninpesassa" appears again at
        # the start of the actual story. We want the *second* occurrence.
        first_title = "alhaalla kaninpesassa"
        count = 0
        for i in 1:n
            if strip(lines[i]) == first_title
                count += 1
                if count == 2
                    # Skip this title line and any blank lines after it
                    j = i + 1
                    while j <= n && isempty(strip(lines[j]))
                        j += 1
                    end
                    return j
                end
            end
        end
        # Fallback: skip front matter heuristically
        return 26
    end

    if lang == "french"
        # French: "chapitre premier" is the first real chapter heading.
        # Skip everything before it, then skip the heading + title + blanks.
        for i in 1:n
            if strip(lines[i]) == "chapitre premier"
                j = i + 1
                # Skip blank lines
                while j <= n && isempty(strip(lines[j]))
                    j += 1
                end
                # Skip chapter title line if present
                if j <= n && strip(lines[j]) in titles
                    j += 1
                end
                # Skip blank lines after title
                while j <= n && isempty(strip(lines[j]))
                    j += 1
                end
                return j
            end
        end
    end

    # English, German, Italian: find the first chapter heading
    for i in 1:n
        if is_chapter_heading(lines[i], lang)
            j = i + 1
            # Skip blank lines
            while j <= n && isempty(strip(lines[j]))
                j += 1
            end
            # Skip chapter title line if present
            if j <= n && strip(lines[j]) in titles
                j += 1
            end
            # Skip blank lines after title
            while j <= n && isempty(strip(lines[j]))
                j += 1
            end
            return j
        end
    end

    return 1  # fallback: no front matter detected
end

# ── Check if a line is a TOC entry ──
# TOC entries in French look like: "i au fond du terrier", "ii la mare aux larmes"
# These are roman numeral prefixed title lines.

function is_toc_entry(line::AbstractString)
    s = strip(line)
    return occursin(r"^[ivxlc]+ ", s) && length(s) < 80
end

# ── Main reclean function for one file ──

function reclean(filepath::AbstractString, lang::AbstractString)
    text = read(filepath, String)
    original_lines = split(text, '\n')
    original_count = length(original_lines)

    lines = collect(original_lines)
    n = length(lines)

    # Step 1: Strip front matter
    start = find_story_start(lines, lang)
    lines = lines[start:end]

    # Step 2-6: Filter lines
    titles = chapter_title_set(lang)
    filtered = String[]
    i = 1
    while i <= length(lines)
        line = lines[i]
        s = strip(line)

        # Remove illustration markers
        if is_illustration(line)
            i += 1
            continue
        end

        # Remove chapter headings and their following title lines
        if is_chapter_heading(line, lang)
            i += 1
            # Skip blank lines after heading
            while i <= length(lines) && isempty(strip(lines[i]))
                i += 1
            end
            # Skip chapter title line if it matches known titles
            if i <= length(lines) && strip(lines[i]) in titles
                i += 1
            end
            continue
        end

        # For Finnish: remove standalone chapter title lines (they serve as headings)
        if lang == "finnish" && s in titles
            i += 1
            continue
        end

        # Remove chapter title lines that appear standalone
        # (in case they weren't caught by the heading+title pair logic)
        if s in titles
            i += 1
            continue
        end

        # Remove artifact lines
        if s in ARTIFACT_LINES
            i += 1
            continue
        end

        # Remove end markers more broadly
        if s in Set(["the end", "fin", "ende", "fine"])
            i += 1
            continue
        end

        push!(filtered, string(line))
        i += 1
    end

    # Step 7: Collapse multiple blank lines into single blank lines
    result_text = join(filtered, "\n")
    result_text = replace(result_text, r"\n{3,}" => "\n\n")

    # Trim leading/trailing whitespace
    result_text = strip(result_text)

    # Write back
    write(filepath, result_text * "\n")

    new_lines = split(result_text, '\n')
    new_count = length(new_lines)
    new_chars = length(result_text)
    removed = original_count - new_count

    println("  Original lines: " * string(original_count))
    println("  New lines:      " * string(new_count))
    println("  Lines removed:  " * string(removed))
    println("  New characters: " * string(new_chars))
    println("  First 5 lines:")
    for j in 1:min(5, new_count)
        println("    " * string(new_lines[j]))
    end
    println()
end

# ── Main ──

function main()
    langs = [
        ("alice_english.txt", "english"),
        ("alice_french.txt",  "french"),
        ("alice_german.txt",  "german"),
        ("alice_italian.txt", "italian"),
        ("alice_finnish.txt", "finnish"),
    ]

    for (filename, lang) in langs
        filepath = joinpath(DATA_DIR, filename)
        if !isfile(filepath)
            println("WARNING: File not found, skipping: " * filepath)
            continue
        end
        println("Re-cleaning: " * filename * " (" * lang * ")")
        reclean(filepath, lang)
    end
end

main()
