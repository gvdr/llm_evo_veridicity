#!/usr/bin/env julia
# download_wikisource_cantos.jl
#
# Downloads Portuguese and Polish Divine Comedy cantos from Wikisource
# and assembles them into single raw text files.
# Respects rate limits with proper User-Agent and retry logic.

using Downloads

const DATA_DIR = joinpath(@__DIR__, "..", "data", "dante")
const USER_AGENT = "DanteCleaner/1.0 (academic research; contact: gvdr@pm.me)"

function roman(n::Int)
    result = ""
    for (val, sym) in [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        while n >= val
            result = result * sym
            n -= val
        end
    end
    return result
end

function url_encode_wiki(s::String)
    s = replace(s, " " => "_")
    out = IOBuffer()
    for c in s
        if isascii(c) && (isletter(c) || isdigit(c) || c in ['-', '_', '.', '~', '/', '(', ')'])
            write(out, c)
        else
            for b in codeunits(string(c))
                write(out, "%" * uppercase(string(b, base=16, pad=2)))
            end
        end
    end
    return String(take!(out))
end

function fetch_with_retry(url::String; max_retries::Int=3)
    for attempt in 1:max_retries
        try
            buf = IOBuffer()
            Downloads.download(url, buf; headers=["User-Agent" => USER_AGENT])
            return String(take!(buf))
        catch e
            err_str = string(e)
            if occursin("429", err_str)
                wait_time = 10 * attempt
                println("    Rate limited, waiting " * string(wait_time) * "s (attempt " * string(attempt) * "/" * string(max_retries) * ")...")
                sleep(wait_time)
            else
                if attempt == max_retries
                    println("    FAILED after " * string(max_retries) * " attempts: " * first(err_str, 200))
                    return nothing
                end
                sleep(2)
            end
        end
    end
    return nothing
end

# --- Portuguese ---
println("=== Downloading Portuguese cantos ===")
pt_base = "https://pt.wikisource.org/w/index.php?action=raw&title="
pt_sections = [
    ("Inferno", 34),
    ("Purgat\u00f3rio", 33),
    ("Para\u00edso", 33),
]

pt_parts = String[]
for (section, count) in pt_sections
    for i in 1:count
        r = roman(i)
        title = "A Divina Com\u00e9dia (Xavier Pinheiro)/grafia atualizada/" * section * "/" * r
        url = pt_base * url_encode_wiki(title)
        print("  PT " * section * "/" * r * "... ")
        data = fetch_with_retry(url)
        if data !== nothing
            push!(pt_parts, data)
            println("OK (" * string(length(data)) * " chars)")
        end
        sleep(1.0)  # Be polite to Wikisource
    end
end

pt_out = joinpath(DATA_DIR, "dante_portuguese_raw.txt")
open(pt_out, "w") do f
    write(f, join(pt_parts, "\n\n"))
end
println("Portuguese: " * string(filesize(pt_out)) * " bytes, " * string(length(pt_parts)) * " cantos")

# --- Polish (rendered HTML -> plain text) ---
println("\n=== Downloading Polish cantos (rendered HTML) ===")
pl_base = "https://pl.wikisource.org/w/index.php?action=render&title="
pl_sections = [
    ("Piek\u0142o", 34),
    ("Czy\u015bciec", 33),
    ("Raj", 33),
]

pl_parts = String[]
for (section, count) in pl_sections
    for i in 1:count
        r = roman(i)
        title = "Boska Komedia (Stanis\u0142awski)/" * section * " - Pie\u015b\u0144 " * r
        url = pl_base * url_encode_wiki(title)
        print("  PL " * section * "/" * r * "... ")
        html = fetch_with_retry(url)
        if html !== nothing
            # Convert HTML to plain text
            text = replace(html, r"<br\s*/?>" => "\n")
            text = replace(text, r"<[^>]+>" => "")
            text = replace(text, "&amp;" => "&")
            text = replace(text, "&lt;" => "<")
            text = replace(text, "&gt;" => ">")
            text = replace(text, "&nbsp;" => " ")
            text = replace(text, "&mdash;" => " ")
            text = replace(text, "&ndash;" => " ")
            text = replace(text, r"&#\d+;" => " ")
            text = replace(text, r"&[a-zA-Z]+;" => " ")
            push!(pl_parts, text)
            println("OK (" * string(length(text)) * " chars)")
        end
        sleep(1.0)
    end
end

pl_out = joinpath(DATA_DIR, "dante_polish_raw.txt")
open(pl_out, "w") do f
    write(f, join(pl_parts, "\n\n"))
end
println("Polish: " * string(filesize(pl_out)) * " bytes, " * string(length(pl_parts)) * " cantos")

println("\nDone!")
