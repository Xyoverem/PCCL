"""Extract detailed profiling data from nsys sqlite database."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pccl_prof.sqlite"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get string table for resolving names
c.execute("SELECT id, value FROM StringIds")
strings = dict(c.fetchall())

print("=" * 100)
print("CUDA KERNEL EXECUTION TIMES (GPU-side)")
print("=" * 100)

# Get kernel execution data grouped by name
c.execute("""
    SELECT demangledName, COUNT(*) as cnt,
           AVG(end - start) as avg_ns,
           MIN(end - start) as min_ns,
           MAX(end - start) as max_ns,
           SUM(end - start) as total_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    GROUP BY demangledName
    ORDER BY total_ns DESC
""")
for row in c.fetchall():
    name = strings.get(row[0], f"id={row[0]}")
    # Truncate long names
    short_name = name[:80] if len(name) > 80 else name
    cnt, avg_ns, min_ns, max_ns, total_ns = row[1], row[2], row[3], row[4], row[5]
    print(f"  {short_name:<80s}")
    print(f"    count={cnt:4d}  avg={avg_ns/1e3:10.1f}us  min={min_ns/1e3:10.1f}us  max={max_ns/1e3:10.1f}us  total={total_ns/1e6:10.2f}ms")

print()
print("=" * 100)
print("CUDA EXECUTOR KERNEL - PER INSTANCE DETAILS")
print("=" * 100)

# Get individual cuda_executor_kernel instances sorted by duration
c.execute("""
    SELECT start, end, (end - start) as duration, deviceId, gridX, gridY, gridZ, blockX
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE demangledName IN (
        SELECT id FROM StringIds WHERE value LIKE '%cuda_executor_kernel%'
    )
    ORDER BY start
""")
rows = c.fetchall()
print(f"  Total instances: {len(rows)}")
durations = [(r[2]) for r in rows]
durations.sort()
if durations:
    print(f"  Duration distribution (us):")
    print(f"    min={durations[0]/1e3:.1f}  p10={durations[len(durations)//10]/1e3:.1f}  "
          f"p50={durations[len(durations)//2]/1e3:.1f}  p90={durations[9*len(durations)//10]/1e3:.1f}  "
          f"max={durations[-1]/1e3:.1f}")

    # Group by approximate size (based on duration clusters)
    small = [d for d in durations if d < 100000]  # <100us
    medium = [d for d in durations if 100000 <= d < 500000]  # 100-500us
    large = [d for d in durations if d >= 500000]  # >500us
    print(f"    <100us: {len(small)} instances, avg={sum(small)/len(small)/1e3:.1f}us" if small else "")
    print(f"    100-500us: {len(medium)} instances, avg={sum(medium)/len(medium)/1e3:.1f}us" if medium else "")
    print(f"    >500us: {len(large)} instances, avg={sum(large)/len(large)/1e3:.1f}us" if large else "")

print()
print("=" * 100)
print("CUDA MEMCPY OPERATIONS")
print("=" * 100)

c.execute("""
    SELECT copyKind, COUNT(*) as cnt,
           AVG(end - start) as avg_ns,
           MIN(end - start) as min_ns,
           MAX(end - start) as max_ns,
           SUM(end - start) as total_ns,
           AVG(bytes) as avg_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
    ORDER BY total_ns DESC
""")
# copyKind: 1=H2D, 2=D2H, 8=D2D
kind_names = {1: "Host-to-Device", 2: "Device-to-Host", 8: "Device-to-Device"}
for row in c.fetchall():
    kind = kind_names.get(row[0], f"kind={row[0]}")
    cnt, avg_ns, min_ns, max_ns, total_ns, avg_bytes = row[1], row[2], row[3], row[4], row[5], row[6]
    print(f"  {kind:<25s}  count={cnt:4d}  avg={avg_ns/1e3:8.1f}us  "
          f"min={min_ns/1e3:8.1f}us  max={max_ns/1e3:8.1f}us  total={total_ns/1e6:8.2f}ms  "
          f"avg_bytes={avg_bytes/1024:.1f}KB")

print()
print("=" * 100)
print("D2D MEMCPY SIZE DISTRIBUTION")
print("=" * 100)

c.execute("""
    SELECT bytes, (end - start) as duration
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE copyKind = 8
    ORDER BY bytes
""")
d2d_rows = c.fetchall()
if d2d_rows:
    # Group by size
    from collections import defaultdict
    size_groups = defaultdict(list)
    for bytes_val, dur in d2d_rows:
        if bytes_val < 1024:
            key = f"{bytes_val}B"
        elif bytes_val < 1024*1024:
            key = f"{bytes_val//1024}KB"
        else:
            key = f"{bytes_val//(1024*1024)}MB"
        size_groups[key].append(dur)

    for key in sorted(size_groups.keys(), key=lambda k: d2d_rows[0][0]):
        durs = size_groups[key]
        avg_us = sum(durs) / len(durs) / 1e3
        print(f"  {key:>10s}: {len(durs):4d} copies, avg={avg_us:8.1f}us")

print()
print("=" * 100)
print("NVTX PHASE SUMMARY (host-side API timing)")
print("=" * 100)

c.execute("""
    SELECT text, COUNT(*) as cnt,
           AVG(end - start) as avg_ns,
           MIN(end - start) as min_ns,
           MAX(end - start) as max_ns,
           SUM(end - start) as total_ns
    FROM NVTX_EVENTS
    WHERE eventType = 59
    GROUP BY text
    ORDER BY avg_ns DESC
""")
for row in c.fetchall():
    name = strings.get(row[0], f"id={row[0]}")
    cnt, avg_ns, min_ns, max_ns, total_ns = row[1], row[2], row[3], row[4], row[5]
    print(f"  {name:<20s}  count={cnt:4d}  avg={avg_ns/1e3:10.1f}us  "
          f"min={min_ns/1e3:8.1f}us  max={max_ns/1e3:8.1f}us")

conn.close()
