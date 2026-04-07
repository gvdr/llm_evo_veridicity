#!/bin/bash
# download_dante.sh
#
# Downloads Divine Comedy translations from Project Gutenberg and Wikisource.
# Run from any directory; all paths are absolute.
set -euo pipefail

DATADIR="/home/gvdr/reps/veridicity/llm/data/dante"
mkdir -p "$DATADIR"

echo "=== Gutenberg downloads ==="

echo "  Italian..."
curl -sL "https://www.gutenberg.org/cache/epub/1000/pg1000.txt" \
    -o "$DATADIR/dante_italian_raw.txt"

echo "  English..."
curl -sL "https://www.gutenberg.org/cache/epub/1004/pg1004.txt" \
    -o "$DATADIR/dante_english_raw.txt"

echo "  German..."
curl -sL "https://www.gutenberg.org/cache/epub/8085/pg8085.txt" \
    -o "$DATADIR/dante_german_raw.txt"

echo "  Finnish..."
curl -sL "https://www.gutenberg.org/cache/epub/12546/pg12546.txt" \
    -o "$DATADIR/dante_finnish_raw.txt"

echo "  Spanish..."
curl -sL "https://www.gutenberg.org/cache/epub/57303/pg57303.txt" \
    -o "$DATADIR/dante_spanish_raw.txt"

echo "  French (Part 1)..."
curl -sL "https://www.gutenberg.org/cache/epub/22768/pg22768.txt" \
    -o "$DATADIR/dante_french_raw.txt"
echo "  French (Part 2, appending)..."
curl -sL "https://www.gutenberg.org/cache/epub/22769/pg22769.txt" \
    >> "$DATADIR/dante_french_raw.txt"

echo ""
echo "=== Portuguese (pt.wikisource.org, raw wikitext) ==="
PT_BASE="https://pt.wikisource.org/w/index.php?action=raw&title="
> "$DATADIR/dante_portuguese_raw.txt"

for SECTION in Inferno "Purgatório" "Paraíso"; do
    SECTION_ENC=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$SECTION'))")
    if [ "$SECTION" = "Inferno" ]; then MAX=34; else MAX=33; fi
    for i in $(seq 1 $MAX); do
        # integer to Roman numeral
        R=""
        n=$i
        while [ $n -ge 10 ]; do R="${R}X"; n=$((n-10)); done
        if [ $n -ge 9 ]; then R="${R}IX"; n=$((n-9)); fi
        if [ $n -ge 5 ]; then R="${R}V"; n=$((n-5)); fi
        if [ $n -ge 4 ]; then R="${R}IV"; n=$((n-4)); fi
        while [ $n -ge 1 ]; do R="${R}I"; n=$((n-1)); done

        TITLE_ENC=$(python3 -c "import urllib.parse; print(urllib.parse.quote('A Divina Comédia (Xavier Pinheiro)/grafia atualizada/$SECTION/$R'))")
        echo -n "  PT $SECTION/$R... "
        curl -sL "${PT_BASE}${TITLE_ENC}" >> "$DATADIR/dante_portuguese_raw.txt"
        printf "\n\n" >> "$DATADIR/dante_portuguese_raw.txt"
        echo "OK"
        sleep 0.2
    done
done

echo ""
echo "=== Polish (pl.wikisource.org, rendered HTML) ==="
PL_BASE="https://pl.wikisource.org/w/index.php?action=render&title="
> "$DATADIR/dante_polish_raw.html"

for SECTION in "Piekło" "Czyściec" "Raj"; do
    if [ "$SECTION" = "Piekło" ]; then MAX=34; else MAX=33; fi
    for i in $(seq 1 $MAX); do
        R=""
        n=$i
        while [ $n -ge 10 ]; do R="${R}X"; n=$((n-10)); done
        if [ $n -ge 9 ]; then R="${R}IX"; n=$((n-9)); fi
        if [ $n -ge 5 ]; then R="${R}V"; n=$((n-5)); fi
        if [ $n -ge 4 ]; then R="${R}IV"; n=$((n-4)); fi
        while [ $n -ge 1 ]; do R="${R}I"; n=$((n-1)); done

        TITLE_ENC=$(python3 -c "import urllib.parse; print(urllib.parse.quote('Boska Komedia (Stanisławski)/$SECTION - Pieśń $R'))")
        echo -n "  PL $SECTION/$R... "
        curl -sL "${PL_BASE}${TITLE_ENC}" >> "$DATADIR/dante_polish_raw.html"
        printf "\n\n" >> "$DATADIR/dante_polish_raw.html"
        echo "OK"
        sleep 0.2
    done
done

# Convert Polish HTML to plain text using Python
echo ""
echo "  Converting Polish HTML to plain text..."
python3 -c "
import re, html
with open('$DATADIR/dante_polish_raw.html', 'r', encoding='utf-8') as f:
    text = f.read()
# Remove HTML tags but keep text
text = re.sub(r'<br\s*/?>', '\n', text)
text = re.sub(r'<[^>]+>', '', text)
text = html.unescape(text)
with open('$DATADIR/dante_polish_raw.txt', 'w', encoding='utf-8') as f:
    f.write(text)
"

echo ""
echo "=== File sizes ==="
for f in "$DATADIR"/dante_*_raw.txt; do
    echo "  $(basename "$f"): $(wc -c < "$f") bytes"
done
echo "Done."
