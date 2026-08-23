import sys,csv
rows=list(csv.DictReader(open(sys.argv[1])))
base=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
def col(n):
    v=[]
    for r in rows:
        try: v.append(float(r[n]))
        except: pass
    return v
m=col('mem_used_mib'); u=col('util_gpu'); p=col('power_w')
if not m: print("no samples"); sys.exit()
act=[x for x,y in zip(p,u) if y>20] or p
print(f"samples={len(rows)}")
print(f"VRAM  peak={max(m):.0f} MiB   (minus desktop {base:.0f} -> {max(m)-base:.0f} MiB)")
print(f"GPUutil mean={sum(u)/len(u):.1f}%  peak={max(u):.0f}%")
print(f"Power  peak={max(p):.1f} W  mean(all)={sum(p)/len(p):.1f} W  mean(util>20%)={sum(act)/len(act):.1f} W")
