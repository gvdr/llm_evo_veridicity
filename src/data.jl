# ── Data loading and tokenization for multilingual experiments ─────────────────
#
# Supports two data formats:
#   1. Names: one name per line (short sequences, BOS-delimited)
#   2. Continuous text: paragraph-level chunks from books (longer sequences)
# Each "world state" is a language, represented as a text file.
# Tokenization is character-level with a BOS token.

"""
    load_names(path) -> Vector{String}

Load a names file (one name per line, lowercase ASCII).
Strips whitespace, filters empty lines.
"""
function load_names(path::String)
    lines = readlines(path)
    names = String[]
    for line in lines
        s = strip(line)
        isempty(s) && continue
        push!(names, lowercase(s))
    end
    names
end

"""
    build_charset(names_by_lang) -> Vector{Char}

Build sorted unique character set across all languages.
"""
function build_charset(names_by_lang::Dict{String, Vector{String}})
    chars = Set{Char}()
    for (_, names) in names_by_lang
        for name in names
            for c in name
                push!(chars, c)
            end
        end
    end
    sort(collect(chars))
end

"""
    Tokenizer

Character-level tokenizer with BOS token.
"""
struct Tokenizer
    chars    :: Vector{Char}
    char2id  :: Dict{Char, Int}
    bos      :: Int
    vocab_size :: Int
end

function Tokenizer(chars::Vector{Char})
    char2id = Dict(c => i for (i, c) in enumerate(chars))
    bos = length(chars) + 1
    Tokenizer(chars, char2id, bos, bos)
end

"""
    tokenize(tok, name) -> Vector{Int}

Tokenize a name into [BOS, c1, c2, ..., cn, BOS].
"""
function tokenize(tok::Tokenizer, name::String)
    tokens = [tok.bos]
    for c in name
        if haskey(tok.char2id, c)
            push!(tokens, tok.char2id[c])
        end
    end
    push!(tokens, tok.bos)
    tokens
end

"""
    train_test_split(names, ratio; rng)

Split names into (train, test) with the given train ratio.
"""
function train_test_split(names::Vector{String}, ratio::Float64;
                          rng::AbstractRNG=Random.default_rng())
    n = length(names)
    perm = randperm(rng, n)
    split_idx = round(Int, n * ratio)
    train = names[perm[1:split_idx]]
    test  = names[perm[(split_idx + 1):end]]
    (train, test)
end

"""
    load_ecology(data_dir, languages) -> (names_by_lang, tokenizer)

Load all language files and build a shared tokenizer.
Expects files named `names_<language>.txt` in data_dir.
"""
function load_ecology(data_dir::String, languages::Vector{String})
    names_by_lang = Dict{String, Vector{String}}()
    for lang in languages
        path = joinpath(data_dir, "names_" * lang * ".txt")
        if !isfile(path)
            error("Missing data file: " * path)
        end
        names_by_lang[lang] = load_names(path)
    end
    charset = build_charset(names_by_lang)
    tok = Tokenizer(charset)
    (names_by_lang, tok)
end

"""
    make_training_batches(names_by_lang, tok, n_per_lang; rng)

Create a shuffled list of token sequences, balanced across languages.
Returns (tokens_list, lang_labels) where lang_labels[i] is the language
of tokens_list[i].
"""
function make_training_batches(names_by_lang::Dict{String, Vector{String}},
                               tok::Tokenizer, n_per_lang::Int;
                               rng::AbstractRNG=Random.default_rng())
    tokens_list = Vector{Int}[]
    lang_labels = String[]
    for (lang, names) in names_by_lang
        n = min(n_per_lang, length(names))
        sampled = names[randperm(rng, length(names))[1:n]]
        for name in sampled
            push!(tokens_list, tokenize(tok, name))
            push!(lang_labels, lang)
        end
    end
    perm = randperm(rng, length(tokens_list))
    (tokens_list[perm], lang_labels[perm])
end

# ── Continuous text loading (for Alice, Voynich, etc.) ────────────────────────

"""
    load_text(path) -> String

Load a text file as a single lowercase ASCII string.
Only keeps a-z and spaces; collapses whitespace.
"""
function load_text(path::String)
    raw = read(path, String)
    # Keep only a-z and whitespace, lowercase
    cleaned = Char[]
    sizehint!(cleaned, length(raw))
    for c in lowercase(raw)
        if 'a' <= c <= 'z'
            push!(cleaned, c)
        elseif c == ' ' || c == '\n' || c == '\t'
            # Collapse whitespace to single space
            if !isempty(cleaned) && cleaned[end] != ' '
                push!(cleaned, ' ')
            end
        end
    end
    String(cleaned)
end

"""
    chunk_text(text, chunk_size; overlap=0) -> Vector{String}

Split continuous text into overlapping chunks of approximately chunk_size
characters, breaking at word boundaries.
"""
function chunk_text(text::String, chunk_size::Int; overlap::Int=0)
    chunks = String[]
    pos = 1
    n = length(text)
    while pos <= n
        # Find end of chunk
        hard_end = min(pos + chunk_size - 1, n)
        end_pos = hard_end
        # Try to break at a space, but only if the resulting chunk
        # is long enough to guarantee forward progress past the overlap
        if end_pos < n
            space_pos = findprev(' ', text, end_pos)
            if space_pos !== nothing && space_pos > pos &&
               (space_pos - pos + 1) > overlap
                end_pos = space_pos
            end
        end
        chunk = text[pos:end_pos]
        if length(chunk) >= 3
            push!(chunks, chunk)
        end
        # Guarantee forward progress: advance by at least 1.
        # Near the end, skip overlap to avoid tiny trailing fragments.
        remaining = n - end_pos
        eff_overlap = remaining > overlap ? overlap : 0
        next_pos = end_pos + 1 - eff_overlap
        pos = max(next_pos, pos + 1)
    end
    chunks
end

"""
    tokenize_chunk(tok, chunk) -> Vector{Int}

Tokenize a text chunk into [BOS, c1, c2, ..., cn, BOS].
Spaces are skipped (not tokenized) — the model sees only letter sequences.
"""
function tokenize_chunk(tok::Tokenizer, chunk::String)
    tokens = Int[tok.bos]
    sizehint!(tokens, length(chunk) + 2)
    for c in chunk
        if haskey(tok.char2id, c)
            push!(tokens, tok.char2id[c])
        end
    end
    push!(tokens, tok.bos)
    tokens
end

"""
    load_text_ecology(data_dir, file_map) -> (chunks_by_lang, tokenizer)

Load continuous text files for multiple languages.
file_map is a Dict mapping language name to filename.
Chunks each text into sequences suitable for training.

Example:
    load_text_ecology("data/alice",
        Dict("english" => "alice_english.txt",
             "french"  => "alice_french.txt"))
"""
function load_text_ecology(data_dir::String, file_map::Dict{String,String};
                           chunk_size::Int=60, overlap::Int=10)
    # First pass: build charset
    all_chars = Set{Char}()
    texts = Dict{String, String}()
    for (lang, fname) in file_map
        path = joinpath(data_dir, fname)
        if !isfile(path)
            error("Missing data file: " * path)
        end
        text = load_text(path)
        texts[lang] = text
        for c in text
            if 'a' <= c <= 'z'
                push!(all_chars, c)
            end
        end
    end

    charset = sort(collect(all_chars))
    tok = Tokenizer(charset)

    # Second pass: chunk and return
    chunks_by_lang = Dict{String, Vector{String}}()
    for (lang, text) in texts
        chunks_by_lang[lang] = chunk_text(text, chunk_size; overlap=overlap)
    end

    (chunks_by_lang, tok)
end

"""
    text_to_token_sequences(chunks, tok; max_seq_len=0) -> Vector{Vector{Int}}

Convert text chunks to token sequences. If max_seq_len > 0, truncate
sequences longer than max_seq_len tokens.
"""
function text_to_token_sequences(chunks::Vector{String}, tok::Tokenizer;
                                  max_seq_len::Int=0)
    seqs = Vector{Int}[]
    for chunk in chunks
        seq = tokenize_chunk(tok, chunk)
        if max_seq_len > 0 && length(seq) > max_seq_len
            seq = seq[1:max_seq_len]
        end
        length(seq) >= 3 && push!(seqs, seq)
    end
    seqs
end
