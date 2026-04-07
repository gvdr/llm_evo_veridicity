#!/usr/bin/env julia
#
# clean_alice.jl
#
# Reads each raw Alice in Wonderland file, strips Gutenberg header/footer,
# lowercases, strips diacritics to pure ASCII a-z, removes non-letter/non-space
# characters, preserves paragraph structure, and writes cleaned versions.

using Unicode

const DATA_DIR = joinpath(@__DIR__, "..", "data", "alice")

const FILES = [
    ("alice_english_raw.txt", "alice_english.txt"),
    ("alice_french_raw.txt",  "alice_french.txt"),
    ("alice_german_raw.txt",  "alice_german.txt"),
    ("alice_italian_raw.txt", "alice_italian.txt"),
    ("alice_finnish_raw.txt", "alice_finnish.txt"),
]

"""
    strip_gutenberg(text::String) -> String

Remove Project Gutenberg header (everything up to and including the
"*** START OF THE PROJECT GUTENBERG EBOOK ..." line) and footer
(everything from "*** END OF THE PROJECT GUTENBERG EBOOK ..." onward).
Also removes the secondary "End of Project Gutenberg's ..." line that
some files include just before the *** END marker, and strips
"Produced by ..." credit lines that appear right after the START marker.
"""
function strip_gutenberg(text::String)
    lines = split(text, '\n')
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

    # Trim from the end: remove "End of Project Gutenberg's ..." lines
    while end_idx > start_idx + 1
        stripped_line = strip(lines[end_idx])
        if isempty(stripped_line) || occursin(r"^End of (the )?Project Gutenberg"i, stripped_line)
            end_idx -= 1
        else
            break
        end
    end

    # Trim editorial/colophon sections from the end.
    # Search backward from end for these markers; truncate at the earliest one found.
    # Only search in the last ~200 lines to avoid false positives in the body.
    editorial_markers = [
        r"^\s*Nota del Trascrittore\s*$"i,       # Italian transcriber's note
        r"^\s*Liste des modifications"i,           # French editorial corrections
        r"^\s*\*\s+\*\s+\*\s+\*\s+\*\s*$",       # separator line (* * * * *)
        r"IMPRIMERIE|STAMPATORI"i,                 # printer colophon line
    ]
    search_from = max(start_idx + 1, end_idx - 200)
    for marker in editorial_markers
        for i in search_from:end_idx
            if occursin(marker, lines[i])
                end_idx = i - 1
                while end_idx > start_idx + 1 && isempty(strip(lines[end_idx]))
                    end_idx -= 1
                end
                break
            end
        end
    end

    # Trim from the start: skip credit blocks ("Produced by", "E-text prepared by")
    # and "Note: Project Gutenberg" blocks that appear after the START marker
    body_start = start_idx + 1

    # Repeatedly skip known preamble blocks
    changed = true
    while changed
        changed = false
        # Skip blank lines
        while body_start <= end_idx && isempty(strip(lines[body_start]))
            body_start += 1
        end
        # Check for credit lines
        if body_start <= end_idx && occursin(r"^(Produced by|E-text prepared by)"i, strip(lines[body_start]))
            while body_start <= end_idx && !isempty(strip(lines[body_start]))
                body_start += 1
            end
            changed = true
        end
        # Check for "Note: Project Gutenberg" blocks
        if body_start <= end_idx && occursin(r"^Note:\s*Project Gutenberg"i, strip(lines[body_start]))
            while body_start <= end_idx && !isempty(strip(lines[body_start]))
                body_start += 1
            end
            changed = true
        end
        # Check for "Images of the original" blocks
        if body_start <= end_idx && occursin(r"^Images of the original"i, strip(lines[body_start]))
            while body_start <= end_idx && !isempty(strip(lines[body_start]))
                body_start += 1
            end
            changed = true
        end
    end

    return join(lines[body_start:end_idx], '\n')
end

"""
    strip_diacritics(c::Char) -> String

Decompose a character via NFD and keep only the ASCII base letter(s),
discarding combining marks. Returns lowercase ASCII.
"""
function strip_diacritics(c::Char)
    # Special-case some characters that NFD decomposition doesn't handle
    # the way we want (ligatures, etc.)
    c_lower = lowercase(c)
    # Handle common ligatures
    c_lower == 'æ' && return "ae"
    c_lower == 'œ' && return "oe"
    c_lower == 'ß' && return "ss"
    c_lower == 'ð' && return "d"
    c_lower == 'þ' && return "th"

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

"""
    clean_text(text::String) -> String

Lowercase, strip diacritics, remove non-letter/non-whitespace characters,
and normalize whitespace while preserving paragraph breaks (blank lines).
"""
function clean_text(text::String)
    # Process character by character
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

    # Normalize whitespace within lines: collapse multiple spaces to one
    # Preserve blank lines (paragraph structure)
    lines = split(raw_cleaned, '\n')
    cleaned_lines = String[]
    for line in lines
        # Collapse multiple spaces, strip leading/trailing spaces
        stripped = strip(replace(line, r" +" => " "))
        push!(cleaned_lines, string(stripped))
    end

    # Collapse runs of 3+ blank lines to 2 (one blank line between paragraphs)
    result = join(cleaned_lines, '\n')
    result = replace(result, r"\n{3,}" => "\n\n")

    # Trim leading/trailing whitespace
    return strip(result)
end

function main()
    for (raw_name, clean_name) in FILES
        raw_path = joinpath(DATA_DIR, raw_name)
        clean_path = joinpath(DATA_DIR, clean_name)

        if !isfile(raw_path)
            @warn "File not found, skipping: " * raw_path
            continue
        end

        println("Processing: " * raw_name * " -> " * clean_name)

        # Read raw file
        text = read(raw_path, String)

        # Strip Gutenberg boilerplate
        text = strip_gutenberg(text)

        # Clean the text
        text = clean_text(text)

        # Write cleaned file
        write(clean_path, text * "\n")

        # Report stats
        char_count = length(text)
        line_count = count(==('\n'), text) + 1
        println("  Characters: " * string(char_count))
        println("  Lines: " * string(line_count))
        println("  First 5 non-empty lines:")
        nonempty = filter(!isempty, split(text, '\n'))
        for line in first(nonempty, 5)
            println("    " * string(first(line, 80)))
        end
        println()
    end
end

main()
