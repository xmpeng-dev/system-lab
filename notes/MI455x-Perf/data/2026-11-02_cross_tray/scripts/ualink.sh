echo "host=$(hostname -s)"
for c in /sys/class/drm/card*/device; do
    [ -d "$c/ualink" ] || continue
    bdf=$(basename "$(readlink -f "$c")")
    echo "== $bdf"
    for f in "$c"/ualink/*; do
        [ -f "$f" ] && printf '  %s=%s\n' "$(basename "$f")" "$(tr '\n' ' ' < "$f" 2>/dev/null | head -c 120)"
    done
    for d in "$c"/ualink/*/; do
        [ -d "$d" ] || continue
        for f in "$d"*; do [ -f "$f" ] && printf '  %s/%s=%s\n' "$(basename "$d")" "$(basename "$f")" "$(tr '\n' ' ' < "$f" 2>/dev/null | head -c 120)"; done
    done
done 2>/dev/null | awk '!seen[$0]++' | head -80
echo ---afm
which afmctl afmd 2>&1 | grep -v "no afm"
systemctl list-units --all --no-pager 2>/dev/null | grep -iE "afm|fabric|ualink|ualoe" | head
ls /opt/amd*/bin 2>/dev/null | grep -iE "afm|fabric" | head
