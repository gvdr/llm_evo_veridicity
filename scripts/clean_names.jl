"""
clean_names.jl

Reads each names file in data/names/, converts to lowercase ASCII
(stripping diacritics), removes non a-z characters, removes duplicates
and empty lines, overwrites each file, and reports statistics.
"""

using Unicode

const NAMES_DIR = joinpath(@__DIR__, "..", "data", "names")

# Manual fallback mapping for common diacritical characters
# that might not decompose cleanly via NFKD
const MANUAL_MAP = Dict{Char, String}(
    'ä' => "a", 'Ä' => "a",
    'ö' => "o", 'Ö' => "o",
    'ü' => "u", 'Ü' => "u",
    'ß' => "ss",
    'æ' => "ae", 'Æ' => "ae",
    'ø' => "o", 'Ø' => "o",
    'å' => "a", 'Å' => "a",
    'ð' => "d", 'Ð' => "d",
    'þ' => "th", 'Þ' => "th",
    'ł' => "l", 'Ł' => "l",
    'đ' => "d", 'Đ' => "d",
    'ŧ' => "t", 'Ŧ' => "t",
    'ħ' => "h", 'Ħ' => "h",
    'ı' => "i",
    'ĸ' => "k",
    'ŉ' => "n",
    'ŋ' => "n", 'Ŋ' => "n",
    'œ' => "oe", 'Œ' => "oe",
)

"""
    strip_diacritics(s::AbstractString) -> String

Convert a string to lowercase ASCII by:
1. Applying manual mapping for special characters
2. NFKD decomposition to separate base chars from combining marks
3. Stripping combining characters (Unicode category M)
4. Removing any remaining non a-z characters
"""
function strip_diacritics(s::AbstractString)
    # First, apply manual mappings
    buf = IOBuffer()
    for ch in s
        if haskey(MANUAL_MAP, ch)
            write(buf, MANUAL_MAP[ch])
        else
            write(buf, ch)
        end
    end
    mapped = String(take!(buf))

    # Lowercase
    mapped = lowercase(mapped)

    # NFKD decomposition: splits e.g. é into e + combining acute
    decomposed = Unicode.normalize(mapped, :NFKD)

    # After NFKD decomposition, combining characters are separated.
    # We simply keep only a-z characters, which naturally discards
    # combining marks and any other non-ASCII characters.
    buf2 = IOBuffer()
    for ch in decomposed
        if 'a' <= ch <= 'z'
            write(buf2, ch)
        end
    end
    return String(take!(buf2))
end

function clean_file(filepath::String)
    raw_lines = readlines(filepath)

    # Clean each line
    cleaned = String[]
    for line in raw_lines
        name = strip_diacritics(strip(line))
        if !isempty(name)
            push!(cleaned, name)
        end
    end

    # Remove duplicates preserving order
    seen = Set{String}()
    unique_names = String[]
    for name in cleaned
        if !(name in seen)
            push!(seen, name)
            push!(unique_names, name)
        end
    end

    # Overwrite file
    open(filepath, "w") do io
        for (i, name) in enumerate(unique_names)
            write(io, name)
            if i < length(unique_names)
                write(io, "\n")
            end
        end
        write(io, "\n")
    end

    return unique_names
end

function char_frequency(names::Vector{String})
    freq = Dict{Char, Int}()
    for name in names
        for ch in name
            freq[ch] = get(freq, ch, 0) + 1
        end
    end
    return sort(collect(freq), by=x -> -x[2])
end

function main()
    files = sort(filter(f -> endswith(f, ".txt"), readdir(NAMES_DIR)))

    total_names = 0

    println("=" ^ 70)
    println("Cleaning names files in: " * NAMES_DIR)
    println("=" ^ 70)

    for fname in files
        filepath = joinpath(NAMES_DIR, fname)
        names = clean_file(filepath)
        n = length(names)
        total_names += n

        # Extract language from filename: names_arabic.txt -> arabic
        lang = replace(replace(fname, "names_" => ""), ".txt" => "")

        # Sample names (first 5)
        sample = if n >= 5
            join(names[1:5], ", ")
        else
            join(names, ", ")
        end

        # Character frequency
        freq = char_frequency(names)
        freq_str = join([string(p[1]) * ":" * string(p[2]) for p in freq[1:min(10, length(freq))]], " ")

        println()
        println("Language: " * lang)
        println("  Count: " * string(n) * " names")
        println("  Sample: " * sample)
        println("  Top-10 char freq: " * freq_str)
    end

    println()
    println("=" ^ 70)
    println("Total names across all languages: " * string(total_names))
    println("=" ^ 70)
end

main()
