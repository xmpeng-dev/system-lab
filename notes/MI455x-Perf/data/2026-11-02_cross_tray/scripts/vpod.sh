# usage: sudo bash vpod.sh VPOD_ID [SIZE ACCELS]   (ACCELS like 0,1,2,3,4,5,6,7)
ID=$1 SIZE=${2:-} ACC=${3:-}
for b in 0001:01:00.0 0002:01:00.0 0003:01:00.0 0004:01:00.0; do
    c=/sys/bus/pci/devices/$b/ualink/config
    echo -n "$ID" > $c/vpod_id || echo "$b vpod_id write failed"
    if [ -n "$SIZE" ]; then
        echo -n "$SIZE" > $c/vpod_size || echo "$b vpod_size write failed"
        echo -n "$ACC" > $c/vpod_active_accels || echo "$b vpod_active_accels write failed"
        echo -n source-aliasing > $c/addr_mode || echo "$b addr_mode write failed"
    fi
    if echo -n true > $c/commit; then echo "$b commit ok"; else echo "$b commit FAILED"; fi
    sleep 1
done
sleep 3
for b in 0001:01:00.0 0002:01:00.0 0003:01:00.0 0004:01:00.0; do
    i=/sys/bus/pci/devices/$b/ualink
    echo "$b accel=$(cat $i/accel_id) state=$(cat $i/accel_state) vpod=$(cat $i/vpod_id) size=$(cat $i/vpod_size) members=$(cat $i/vpod_active_accels | tr '\n' ' ')"
done
dmesg | grep -E "UALINK|ualink|vpod" | tail -24
