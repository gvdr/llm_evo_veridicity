# clean_voynich.jl
# Reads the raw IVTFF Voynich manuscript EVA transcription and extracts
# clean character-level text from Takahashi's (;H>) complete transcription.
#
# Input:  data/voynich/voynich_raw.txt   (IVTFF format)
# Output: data/voynich/voynich_eva.txt   (pure lowercase a-z + spaces/newlines)

const BASE_DIR = joinpath(@__DIR__, "..", "data", "voynich")
const RAW_PATH = joinpath(BASE_DIR, "voynich_raw.txt")
const OUT_PATH = joinpath(BASE_DIR, "voynich_eva.txt")

function clean_voynich(raw_path::String, out_path::String)
    # Read as raw bytes then decode as Latin-1 to handle any non-UTF8 bytes
    raw_bytes = read(raw_path)
    raw_text = String(map(b -> Char(b), raw_bytes))
    lines = split(raw_text, '\n')

    cleaned_lines = String[]

    for line in lines
        sline = String(line)

        # Skip comment lines (start with #)
        startswith(sline, "#") && continue

        # Skip empty or whitespace-only lines
        isempty(strip(sline)) && continue

        # We only want Takahashi's transcription lines (code ;H>)
        # Data lines have a locator like <f1r.1,@P0;H> at the start
        occursin(";H>", sline) || continue

        # Extract the text after the locator: everything after the first ">"
        # that closes the locator tag. The locator is always at column 1.
        idx = findfirst('>', sline)
        idx === nothing && continue
        text = sline[idx+1:end]

        # Remove inline comments: {...} blocks
        text = replace(text, r"\{[^}]*\}" => "")

        # Remove inline markup: <...> blocks (e.g., <$>, <!@o'>, etc.)
        text = replace(text, r"<[^>]*>" => "")

        # Convert EVA word breaks (. and ,) to spaces
        text = replace(text, '.' => ' ')
        text = replace(text, ',' => ' ')

        # Remove filler and special characters: ! ? % - = ' @
        text = replace(text, r"[!?%\-=@'\*]" => "")

        # Keep only lowercase a-z and spaces
        text = String([c for c in text if (c >= 'a' && c <= 'z') || c == ' '])

        # Collapse multiple spaces into one
        text = replace(text, r" +" => " ")

        # Trim leading/trailing whitespace
        text = strip(text)

        # Skip if nothing left
        isempty(text) && continue

        push!(cleaned_lines, text)
    end

    # Join all lines with newlines
    output = join(cleaned_lines, "\n") * "\n"

    open(out_path, "w") do f
        write(f, output)
    end

    # Report statistics
    total_chars = length(output) - count(==('\n'), output)  # exclude newlines
    char_counts = Dict{Char,Int}()
    for c in output
        c == '\n' && continue
        char_counts[c] = get(char_counts, c, 0) + 1
    end
    unique_chars = sort(collect(keys(char_counts)))

    println("Cleaned Voynich EVA transcription written to: " * out_path)
    println("Total lines: " * string(length(cleaned_lines)))
    println("Total characters (excl. newlines): " * string(total_chars))
    println("Unique characters: " * string(length(unique_chars)))
    println("Characters used: " * join(unique_chars, " "))
    println()
    println("Character frequencies (sorted by count):")
    sorted_chars = sort(collect(char_counts), by=x -> -x[2])
    for (c, n) in sorted_chars
        label = c == ' ' ? "<space>" : string(c)
        println("  " * label * ": " * string(n))
    end
    println()
    println("First 10 lines of cleaned text:")
    for i in 1:min(10, length(cleaned_lines))
        println("  " * cleaned_lines[i])
    end
end

clean_voynich(RAW_PATH, OUT_PATH)
