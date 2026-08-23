# Qwen3.8-27B on one RTX 5080

Everything I measured getting a 27B model to run usefully on a single 16 GB consumer GPU:
where the VRAM goes, how much context actually fits, what quantising the KV cache costs,
and a null result about 3-bit damage that I expected to go the other way.

Published guidance for this pairing says **"8K to 16K context, or 4-bit with RAM offload."**
The daily profile here runs **90,112 tokens of context at 130.8 tok/s, entirely on the GPU**,
with a bisected ceiling of 159,744. Every number below was measured on the machine in
[Rig](#rig), with `-fit off`, a real prefill, and a 5 Hz VRAM sampler attached.

```bash
./serve.sh          # 90,112 ctx, q4_0 KV, MTP draft 3, ~130 tok/s
CTX=131072 ./serve.sh   # more context, MTP no longer fits, ~34 tok/s at full depth
```

---

## Rig

| | |
|---|---|
| GPU | RTX 5080 16 GB, Blackwell, sm_120 |
| Driver / CUDA | 595.84 / 13.3 |
| CPU / RAM | Ryzen 7 9800X3D / 31 GB |
| OS | Ubuntu 26.04, kernel 7.0.0 |
| llama.cpp | b10588 (`70adb1b4c`), built `-DCMAKE_CUDA_ARCHITECTURES=120a-real` |
| Model | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-Q3_K_XL`, 12.24 GiB, 27.32B params |

The display was moved to the motherboard's iGPU. That took the desktop's GPU memory from
292 MiB to 15 MiB, and more usefully it stopped the ceiling moving every time a browser
opened.

## Only 16 of 64 layers have a KV cache

This is the whole reason a 27B model does long context on a small card, and no size-based
rule of thumb captures it.

```
general.architecture           = qwen35
qwen35.block_count             = 65      # 64 base layers + 1 MTP head
qwen35.full_attention_interval = 4
qwen35.attention.head_count_kv = 4
qwen35.attention.key_length    = 256
qwen35.nextn_predict_layers    = 1
```

Reading the tensor list confirms what that implies: `attn_k`/`attn_v` tensors exist on
layers 3, 7, 11 … 63 and on the MTP head at 64, and nowhere else. The other 48 layers are
DeltaNet, linear attention whose recurrent state is a constant **149 MiB** at any context
length.

So the per-token KV cost is set by 16 layers:

```
16 layers x 4 kv-heads x 256 dim x 2 (K+V) = 32,768 elements/token

  f16   64.0 KiB/token
  q8_0  34.0 KiB/token
  q5_1  24.0 KiB/token     no CUDA flash-attn kernel
  q5_0  22.0 KiB/token     no CUDA flash-attn kernel
  q4_0  18.0 KiB/token
```

Cross-check: llama.cpp's own estimator, asked for the full 262,144 context, reports
`context = 16,533 MiB`. That is 262,144 × 64 KiB + 149 MiB, to the megabyte.

## Where the 16,303 MiB goes

Three overheads are unreclaimable and usually invisible in guides.

```
  16,303 MiB   nvidia-smi total
    -462 MiB   driver/firmware reserve      CUDA never sees this
  =15,841 MiB  CUDA-visible total
    -282 MiB   CUDA context                 per process, on cuInit
  =15,559 MiB  actually allocatable for tensors
```

The daily profile then spends it like this: 11,671 MiB model on GPU, 149 MiB DeltaNet
state, ~872 MiB compute buffer at 90K, 978 MiB for MTP at draft depth 3, and 1,584 MiB of
q4_0 KV cache for 90,112 tokens, leaving about 290 MiB of headroom.

Note that 520 MiB of the model stays **host-resident** even at `-ngl 99`. That is the
embedding table, and llama.cpp leaves it on the CPU deliberately: an embedding lookup reads
one row per token and is not bandwidth-sensitive.

**At q4_0 KV, every 1 GiB you free buys 58,254 tokens of context.** Model quant is the
context lever. Not the KV type, and definitely not closing browser tabs.

## Measured context ceilings

Bisected by real allocation with `-fit off`, not calculated.

| KV type | max ctx, no MTP | max ctx, MTP draft 3 | MTP costs |
|---|---|---|---|
| f16 | ~40,960 | | |
| q8_0 | 94,208 | 61,440 | −32,768 tokens |
| **q4_0** | **159,744** | 98,304 | −61,440 tokens |

## Only three KV types are usable

Identical context (n_ctx 1664), flash attention on, one variable changed:

| KV type | prefill tok/s | decode tok/s | verdict |
|---|---|---|---|
| f16 | 2186.96 | 58.53 | fast |
| q8_0 | 2173.80 | 57.97 | fast |
| q4_0 | 2170.97 | 57.94 | fast |
| q4_1 | 227.50 | 43.94 | 9.6× slower prefill |
| q5_0 | **93.75** | 35.16 | **23.3× slower prefill** |
| q5_1 | **94.49** | 35.97 | **23.1× slower prefill** |

CUDA flash-attention kernels exist for f16, q8_0 and q4_0 only. Everything else falls back
to a generic path with **no warning printed**. A `q5_1` run at depth 32,768 looks like a
hang; it is 32K tokens arriving at 94 tok/s, about six minutes.

Quantised KV also **requires** `-fa on`. With flash attention off, context creation fails
outright.

## q4_0 KV costs 0.09% perplexity

Measured with `llama-perplexity` over 104,694 tokens of llama.cpp's own C++ source, so the
corpus is real code rather than wiki text.

| context | f16 | q8_0 | q4_0 | q4_0 delta | error bar |
|---|---|---|---|---|---|
| 8,192 | 1.6057 | 1.6059 | 1.6063 | +0.037% | ±0.0097 |
| 32,768 | 1.4909 | | 1.4923 | +0.094% | ±0.0059 |

The gap grows with context and stays 4–16× smaller than the error bar. This is the finding
that turns a 40K-context card into a 160K one.

The f16 row at 65,536 is missing because f16 KV does not fit that context on this card. That
is the result, not a failed run.

## Speed at real depth

`llama-bench -d N` prefills N tokens before timing, so these are filled contexts. Allocating
32K and sending a 12-token prompt measures nothing.

| depth | prefill tok/s | decode tok/s | cold time to first token |
|---|---|---|---|
| 0 | 2186.0 | 57.10 | |
| 16,384 | 1883.0 | 52.48 | 8 s |
| 32,768 | 1593.9 | 48.80 | 20 s |
| 65,536 | 1258.7 | 42.76 | 52 s |
| 98,304 | 1037.2 | 37.77 | 94 s |
| 131,072 | 881.8 | 33.70 | 148 s |

Decode falls 41% across 0 → 131K. Peak VRAM at 131,072 was 15,533 MiB with no OOM and no
allocator warnings. Sustained power was 352 W mean, 365 W peak, at 98.9% utilisation.

The real ceiling is not VRAM. It is that 148-second cold prefill, and the fact that 131,072
tokens is **11,623 lines of C++** (measured: 11.27 tokens per line). llama.cpp's own `src/`
is 576,264 tokens. Context is not how you load a codebase.

## MTP is worth ~1 GB, and its best depth depends on content

Multi-token prediction ships inside this GGUF as layer 64. Generation is
memory-bandwidth-bound, so verifying several drafted tokens inside one weight read is close
to free throughput. Output is **bit-identical**: MTP off, draft 3 and draft 4 at temperature
0 produce the same MD5.

Cost is linear: **528 MiB base + 150 MiB per draft token**. A widely repeated "~1.9 GB
overhead" figure turned out to be an artifact of benchmarking with `--fit` left on.

Decode tok/s, `-n 256`, temperature 0, seed 42, five different prompts:

| prompt | off | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|
| code | 57.7 | 84.1 | 89.0 | 91.2 | **99.0** |
| chain-of-thought | 57.6 | 91.0 | 100.0 | 103.6 | **130.8** |
| creative prose | 57.6 | 75.5 | **86.1** | 83.3 | 74.1 |
| technical exposition | 57.6 | 81.7 | 89.0 | **92.7** | 87.7 |
| Turkish | 57.7 | 85.4 | 95.3 | **98.3** | 85.9 |
| mean | 57.6 | 83.5 | 91.9 | 93.8 | 95.5 |
| worst case | | 75.5 | **86.1** | 83.3 | 74.1 |

Draft 4 wins on mean and loses on worst case: it collapses on prose, where drafts get
rejected. Draft 3 is the balanced default here; draft 4 if the workload is only code.

`--spec-draft-type-k` / `-ctkd` make no measurable difference for `draft-mtp` (13,678 MiB
with f16 draft cache versus 13,694 with q8_0). Those flags can come off the command line.

## The prompt cache is the biggest daily-use win

OpenAI-compatible endpoint, streaming, three runs each, cold runs prefixed with a random
nonce so they miss the cache.

| prompt tokens | cold TTFT | warm TTFT | cached_tokens | decode tok/s |
|---|---|---|---|---|
| 80 | 0.173 s | 0.178 s | 76 | 92.5 |
| 7,080 | 3.606 s | **0.089 s** | 7,076 | 90.5 |
| 18,080 | 9.568 s | **0.112 s** | 18,076 | 82.3 |
| 36,080 | 20.743 s | **0.151 s** | 36,076 | 77.3 |

A 36K-token conversation re-prefills in 0.151 s instead of 20.7 s, 137× faster, as long as
the prefix stays stable. It is on by default and rarely mentioned.

A prompt longer than `n_ctx` returns HTTP 400 rather than being silently truncated.

## This model reasons by default

For "Say hi in exactly three words":

| | completion tokens |
|---|---|
| default | **175** (503 characters of reasoning) |
| `reasoning_effort: "none"` | **4** |
| `chat_template_kwargs: {enable_thinking: false}` | **4** |
| `/no_think` suffix | 115 (does nothing; old Qwen3 syntax) |

Reasoning deltas arrive in **`reasoning_content`, not `content`**. A client reading only
`content` sees an empty reply and measures TTFT as infinite. This cost me one broken
benchmark before I noticed.

## The null result: q4_0 KV versus f16 KV

Perplexity is teacher-forced. It measures surprise at text the model did not write, and says
nothing about staying consistent with 8,000 tokens of its own output. During generation the
model attends over its own tokens, and those live in the KV cache, so a 4-bit cache means a
lossy memory of what it wrote 400 tokens ago.

That predicts a specific failure shape: correct setup written, then a consumer written as
though the setup were not there. I found that shape everywhere in the q4_0 artifacts.
Flappy Bird computes a frame delta, clamps it, passes it to `update(dt)`, and never
multiplies by it. The Mandelbrot renderer writes 0–1 floats into a `Uint8ClampedArray` while
a swatch builder 35 lines away does the `×255` correctly. And the heptagon prompt collapsed
into 153 near-identical `MAX_SPEED_BALL_BALL_…_WALL` declarations, spent all 14,000 tokens
on constants, and wrote none of the simulation.

Then I ran the control. Same model, same five prompts, same seed, same temperature, same
draft depth, `-ctk`/`-ctv` changed from `q4_0` to `f16`.

| | q4_0 KV | f16 KV |
|---|---|---|
| confirmed defects | **24** | **24** |
| artifacts completed | 4 / 5 | 5 / 5 |
| same state as the other arm | | 4 of 5 |

An exact tie, and the failure shape is at least as common in the control. Both arms produced
the same defect at the same site: Flappy Bird computes a frame delta and never applies it
under q4_0, and declares `let last = 0` plus an unread `rAF` timestamp under f16.

One real difference survives. Under f16 the heptagon wrote a complete simulation in 3,775
tokens with an impulse solver, mass weighting and substepping. That is 1 of 5 versus 0 of 5,
Fisher exact p = 1.0. On counts near 24, the Poisson interval is about ±10.

**Verdict: insufficient evidence.** The write-up in [`writeup/post.md`](writeup/post.md) has
the full argument, including what I did not check.

## Gotchas worth the price of admission

- **`--fit` is on by default and silently shrinks `-c`.** A run that does not error is not a
  run that got the context you asked for. This invalidated my own first session.
- **`q5_0` / `q5_1` / `q4_1` KV have no CUDA flash-attention kernel** and fall back to a
  ~23× slower prefill with no warning.
- **Quantised KV requires `-fa on`**; context creation fails outright otherwise.
- **`reasoning_content`, not `content`.** See above.
- **Identical prompts hit the prompt cache.** Prefix a nonce or you are benchmarking the cache.
- **`--temp 0 --seed 42` gives a 0.1 tok/s spread across repeats.** That is not evidence of
  reliability; it recomputes the same generation. Vary the *prompt* when the metric depends
  on content. A single-prompt conclusion about draft depth inverted here once five were tested.
- **Kill leftover processes first.** An idle `llama-cli` holding 14.8 GB skewed this
  project's first measurements.
- **`-hf` and resolved blob paths make the server report a hash as the model name.** Point
  `-m` at the snapshot symlink and set `-a`.

## Layout

```
serve.sh              the daily profile; MODEL=, CTX=, NMAX=, HOST=, PORT= all override
benchmarks/RESULTS.md the full measurement log, including runs not summarised here
benchmarks/raw/       nvidia-smi samples behind the VRAM and power figures
oneshot/              the one-shot test harness: prompts, runner, gallery builder
oneshot/results/      every artifact, index.json (timings), defects.json (what is wrong)
writeup/post.md       the article about the null result
```

For measuring a *different* model or card, the generic version of this tooling is in
[`gguf-envelope`](https://github.com/mberatsanli/gguf-envelope). It reads a GGUF header
over an HTTP range request if you point it at a URL, so inspecting a 9 GiB model on
HuggingFace takes two seconds and 30 MB, and it bisects the real ceiling with `-fit off`.

That is worth doing before a download. On this card, at q4_0 KV: Gemma 3 12B Q6_K is
3.2 GiB *smaller* than the model here and gets **57,900** tokens of context against
**158,300**, because all 48 of its layers carry a KV cache and only 16 of these 65 do.

## Corrections welcome

Numbers from other cards, other quants, or a properly powered version of the KV experiment
are the point. Please include your llama.cpp build hash, driver version, the exact flags,
and whether the GPU was also driving a display.

## Licence

MIT.
