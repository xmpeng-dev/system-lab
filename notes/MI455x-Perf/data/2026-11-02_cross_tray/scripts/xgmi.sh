echo "host=$(hostname -s)"
for c in /sys/class/drm/card*/device; do
    [ -e "$c/xgmi_hive_info" ] || continue
    bdf=$(basename "$(readlink -f "$c")")
    echo "$bdf hive=$(cat $c/xgmi_hive_info/xgmi_hive_id 2>/dev/null) node=$(cat $c/xgmi_device_id 2>/dev/null) phys=$(cat $c/xgmi_physical_id 2>/dev/null) err=$(cat $c/xgmi_error 2>/dev/null | head -c 40)"
done | sort -u
ls /sys/class/drm/card1/device/xgmi_hive_info/ 2>/dev/null | tr '\n' ' '; echo
cat /sys/class/drm/card1/device/xgmi_hive_info/* 2>/dev/null | head -5
echo ---topology
amd-smi topology 2>/dev/null | sed -n '1,40p'
echo ---xgmi
amd-smi xgmi 2>/dev/null | sed -n '1,40p'
echo ---kfd_iolinks
for n in /sys/class/kfd/kfd/topology/nodes/*; do
    [ -d "$n/io_links" ] || continue
    for l in $n/io_links/*; do
        echo "$(basename $n)->$(grep -E '^node_to' $l/properties | awk '{print $2}') type=$(grep -E '^type' $l/properties | awk '{print $2}') weight=$(grep weight $l/properties | awk '{print $2}') maxbw=$(grep max_bandwidth $l/properties | awk '{print $2}')"
    done
done | head -30
