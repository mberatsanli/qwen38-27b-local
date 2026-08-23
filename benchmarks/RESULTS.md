# RTX 5080 / Qwen3.8-27B (qwen35) llama.cpp benchmark log

build 70adb1b4c (10588) | nvcc 13.3 | driver 595.84 | CUDA arch 120a-real
model: unsloth/Qwen3.8-27B-GGUF UD-Q3_K_XL, 27.32B, Q3_K_L, 12.23 GiB

## Architecture (from GGUF metadata)
arch=qwen35  block_count=65  context_length=262144
embedding_length=5120  ffn=17408  head_count=24  head_count_kv=4
key_length=256  value_length=256  full_attention_interval=4
ssm: conv_kernel=4 state_size=128 group_count=16 inner_size=6144
nextn_predict_layers=1  (MTP head = layer 64)
NO vision/clip/mmproj tensors -> nothing to disable for text-only

Layer split (from tensor names):
  full-attention layers (17): 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,64
  DeltaNet/ssm layers  (48): all others
=> only 16 of 64 base layers have context-scaling KV. Layer 64 = MTP head.

## VRAM model (VALIDATED: predicted 15434 vs measured 15436 MiB)
nvidia_smi_total = desktop + 350 (CUDA ctx) + 11671 (model on GPU)
                   + context + compute(~505-720)
Host RAM holds 520 MiB of model (token_embd.weight) even at -ngl 99.
Constant DeltaNet recurrent+conv state = 149 MiB (does NOT grow with ctx).

KV cost per token:  f16 64.0 KiB | q8_0 34.0 KiB | q5_1 24.0 | q5_0 22.0 | q4_0 18.0
Freed VRAM -> extra tokens: 1 MiB = 56.9 tok (q4_0) / 30.1 (q8_0) / 16.0 (f16)

## KV type comparison — IDENTICAL ctx (p512 n128 d1024 => n_ctx 1664), fa on
KV     pp512 t/s   tg128 t/s
f16     2186.96      58.53
q8_0    2173.80      57.97
q4_0    2170.97      57.94
q4_1     227.50      43.94
q5_0      93.75      35.16
q5_1      94.49      35.97
=> CUDA flash-attn kernels exist only for f16/q8_0/q4_0. q5_x = 23x slower PREFILL.
=> quantized KV REQUIRES -fa on (fa=off -> "failed to create context")
=> f16 KV with fa off: tg 53.07 (vs ~58.5 fa on) => FA gives ~+10% tg

## Depth sweep, q8_0 KV, fa on, p512/n128, r3
depth      pp512      tg128
0        2201.69      57.79
8192     2043.05      55.66
16384    1899.07      53.59
32768    1599.55      49.88
49152    1407.87      46.34
65536    1258.24      43.60
peak VRAM 15436 MiB | util 97.7% | power mean 343 W peak 364.6 W

## Depth sweep, q4_0 KV, fa on, p512/n128, r2
depth      pp512      tg128
0        2186.03      57.10
16384    1882.98      52.48
32768    1593.87      48.80
65536    1258.68      42.76
98304    1037.22      37.77
131072    881.84      33.70
peak VRAM 15533 MiB | util 98.9% | power mean 352.4 W peak 365.0 W
=> 131072 REAL filled context works, no OOM. tg only -41% vs depth 0.

## KV type at fixed depth 32768
f16 : pp 1630.22  tg 51.13  VRAM_proc 14735 MiB
q8_0: pp 1595.61  tg 49.88  VRAM_proc 13781 MiB
q4_0: pp 1593.87  tg 48.80
=> speed differences ~2%; the real lever is VRAM/max-context, not speed.

## -no-kv-offload (d32768, f16 KV)
nkvo=0: pp 1596.11  tg 50.44  VRAM 14753  power 320 W
nkvo=1: pp  889.90  tg  4.87  VRAM 12457  power 138 W
=> 10.4x slower generation for 2.3 GB saved. USELESS. Power drop proves PCIe starvation.

## Power (answers "Windows only used 75 W")
Linux runs the card at full limit: mean 316-352 W, peak 365 W, util 93-99%.
No power problem to fix.

## GOTCHA
--fit is ON BY DEFAULT and SILENTLY SHRINKS -c to fit VRAM.
Always pass `-fit off` when testing whether a context size genuinely fits.
Earlier session's "-c 16384 works" claims are NOT trustworthy without -fit off.

## Max context (measured / modelled, desktop=292 MiB)
q4_0 : 131072 measured OK (peak 15533). ~147K after moving desktop off GPU.
q8_0 : ~80K   (65536 measured OK, peak 15436)
f16  : ~40K   (32768 measured OK, peak 15317)

## OPEN / NOT YET DONE
- MTP overhead not yet isolated (old session: 14826 MiB at c=16384 q8_0+MTP
  vs 12869 estimated without MTP -> ~1.9 GB unexplained, MUST re-measure with -fit off)
- MTP draft=2 vs 3 re-test (old numbers unreliable, --fit was on)
- -ngl partial offload curve (interrupted, not run)
- server profile / TTFT / OpenAI API throughput
- long-conversation degradation
- quant comparison (Q3_K_XL vs alternatives) <- BIGGEST LEVER, 11671 MiB = 72% of VRAM
- ExLlamaV3 / vLLM / TensorRT-LLM
- KV quant QUALITY impact (perplexity) not yet measured

---
# SESSION 2 (after reboot, display moved to iGPU)

## Desktop VRAM eliminated
Monitor moved to motherboard HDMI (AMD Granite Ridge iGPU, card2-HDMI-A-2).
prime-select was already `on-demand`; only a session restart was needed.
  before: 292 MiB desktop, ggml free 15278 MiB
  after :  15 MiB desktop, ggml free 15559 MiB   (+281 MiB = ~16K tok at q4_0)
Desktop VRAM no longer fluctuates with Chrome -> stable ceiling for server sizing.

## The three unreclaimable overheads (measured)
  16303 MiB  nvidia-smi total (physical)
  -462 MiB   firmware/driver reserve   -> CUDA never sees it
  =15841 MiB CUDA-visible total
  -282 MiB   CUDA context (per process, on cuInit)
  =15559 MiB actually allocatable for tensors
  (+15 MiB gnome-shell/driver structures still on card; not worth chasing)
=> REVISED VRAM MODEL: replace the old "350 MiB CUDA ctx" with measured 282 MiB.

## MTP overhead — CLEAN (-fit off, c=16384, q8_0 KV, fa on)
  MTP off      : 12850 MiB
  n-max=1      : 13528 MiB  (+678)
  n-max=2      : 13678 MiB  (+828)
  n-max=3      : 13828 MiB  (+978)
  n-max=4      : 13978 MiB  (+1128)
  n-max=5      : 14126 MiB  (+1276)
=> ~150 MiB per extra draft token. Base MTP cost ~528 MiB (second compute graph).
=> The old session's "~1.9 GB MTP overhead" was an artifact of --fit being ON. Debunked.
=> -ctkd/-ctvd (draft KV type) makes NO difference: f16 13678 vs q8_0 13694 MiB.
   Dropping `--spec-draft-type-k/v q8_0` from the old command line costs nothing.

## MTP speed vs draft depth — 5 DIFFERENT PROMPTS (gen t/s, -n 256, temp 0 seed 42)
prompt      off     n1     n2     n3     n4
code       57.7   84.1   89.0   91.2   99.0
reasoning  57.6   91.0  100.0  103.6  130.8
prose      57.6   75.5   86.1   83.3   74.1
technical  57.6   81.7   89.0   92.7   87.7
turkish    57.7   85.4   95.3   98.3   85.9
mean       57.6   83.5   91.9   93.8   95.5
worst      ---    75.5   86.1   83.3   74.1
=> n4 has best MEAN but worst WORST-CASE (prose 74.1 < n2's 86.1). Content-dependent:
   acceptance is high for code/CoT reasoning, low for creative prose.
=> METHOD NOTE: repeating with temp 0 + fixed seed gives spread=0.1 t/s. That is NOT
   evidence of generality - it just recomputes the same generation. Vary the PROMPT.
=> RECOMMENDATION: n-max=3 balanced (mean 93.8, +978 MiB); n-max=2 if VRAM tight
   (mean 91.9, best worst-case 86.1, +828 MiB). Avoid n-max=5 (79.3, regression).

## Other flags discovered in this build
--spec-type also offers: ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod,
  ngram-cache, draft-eagle3, draft-dflash, draft-dspark.
  ngram-* need no draft model and likely no extra VRAM -> UNTESTED, worth trying
  at long context where MTP's ~1 GB is expensive.
llama-cli non-interactive flag is `-st/--single-turn` (NOT `-no-cnv`, removed).
Timing line format: `[ Prompt: X t/s | Generation: Y t/s ]`

## TRUE max context — empirically bisected (-fit off, real allocation test)
KV     MTP off    MTP n-max=3    MTP costs
q8_0     94208         61440     -32768 tokens
q4_0    159744         98304     -61440 tokens
NOTE: an earlier estimate failed because MTP's ~978 MiB was not added on top of the
grown KV, and compute buffers DO grow with n_ctx (720 MiB at c=131072, not constant).

## Verified profiles (llama-cli, -n 256, temp 0)
q8_0 + MTP n3 + c=57344  -> 106.0 t/s gen, peak 15605 MiB  <- DAILY
q4_0 + MTP off + c=147456 ->  57.7 t/s gen, peak 15553 MiB  <- LONG DOCS
(ceiling: 15841 MiB CUDA-visible; both leave ~240-290 MiB headroom)

## llama-server daily profile  -> ~/llm-bench/serve.sh
-ngl 99 -fit off -c 57344 -ctk q8_0 -ctv q8_0 -fa on
--spec-type draft-mtp --spec-draft-n-max 3 -np 1 --jinja --metrics
Startup: n_ctx_slot=57344, MTP draft context OK, 15581 MiB.
`-fit off` is ESSENTIAL here, otherwise the server silently picks its own context.

## TTFT / prompt cache (OpenAI API, streaming, 3 runs each)
prompt_tok   COLD TTFT   WARM TTFT   cached_tokens   decode t/s
        80       0.173       0.178              76         92.5
      7080       3.606       0.089            7076         90.5
     18080       9.568       0.112           18076         82.3
     36080      20.743       0.151           36076         77.3
=> Prompt cache is the single biggest daily-use win and is ON by default:
   36K-token conversation re-prefills in 0.151 s instead of 20.7 s (137x).
=> Cold prefill is a flat ~1740 tok/s regardless of size.
=> Prompt over n_ctx -> HTTP 400 (server refuses, does NOT silently truncate). Good.
=> BENCH TRAP: identical prompts hit the cache. Prefix a random nonce for cold numbers.

## Thinking mode (model reasons by default!)
For "Say hi in exactly three words":
  default                                    -> 175 completion tokens (503 chars reasoning)
  chat_template_kwargs {enable_thinking:false} ->   4 tokens   WORKS
  reasoning_effort: "none"                     ->   4 tokens   WORKS
  "/no_think" suffix                           -> 115 tokens   DOES NOT WORK (old Qwen3 syntax)
=> Thinking deltas arrive in `reasoning_content`, NOT `content`. Any client or benchmark
   that only reads `content` will measure TTFT as infinite / see an empty reply.

## STILL OPEN
- KV quant QUALITY (perplexity f16 vs q8_0 vs q4_0) -- needed before trusting q4_0 for long ctx
- quant comparison Q3_K_XL vs alternatives  <- BIGGEST REMAINING LEVER (11671 MiB = 72%)
- ngram-* speculative types (no draft VRAM) vs MTP at long context
- long-conversation degradation over many turns
- CUDA graphs; ExLlamaV3/EXL3; vLLM; TensorRT-LLM

---
# SESSION 3 — KV quant quality + final coding profile

## KV quantization QUALITY (llama-perplexity, llama.cpp C++ source as corpus)
c=8192  (104694 tok corpus):  f16 1.6057+/-0.0097 | q8_0 1.6059 | q4_0 1.6063
c=32768 (750KB corpus)     :  f16 1.4909+/-0.0059 |             | q4_0 1.4923
Relative delta q4_0 vs f16:   +0.037% at 8K  ->  +0.094% at 32K
=> The gap GROWS with context but stays ~4-16x SMALLER than the error bar.
=> VERDICT: q4_0 KV is quality-neutral for code. Use it and take the context.
=> (f16 KV at c=65536 could not be measured - f16 ceiling is ~40K. Consistent
    with our own ceiling table, not a benchmark failure.)

## Context capacity in real units (measured, not rules of thumb)
CODE  : 356015 B / 9284 lines = 104694 tok  -> 1 line C++ = 11.27 tok
        131072 tok = 11623 lines C++ ; 159744 tok = 14165 lines
TEXT  : 199854 B / 24867 words = 58843 tok
        131072 tok = ~55K words (doc corpus w/ code blocks) = ~110 book pages
        pure English prose is more efficient: ~200 book pages
TOKENIZER EFFICIENCY (measured on the live server /tokenize):
        Turkish 2.00 tok/word   |   English 1.05 tok/word   -> TR costs 1.9x
        So Turkish text gets roughly HALF the effective context.
Scale check: llama.cpp src/ = 576264 tok, tools/server/ = 242102 tok.
=> Even 262144 cannot hold a medium C++ subsystem. Context is NOT the way to
   "load the whole codebase"; grep/search-driven retrieval beats a bigger window.

## Cold prefill latency (the real long-context constraint)
ctx      prefill t/s   time to first token
32768         1594          20 s
65536         1259          52 s
98304         1037          94 s
131072         882         148 s
Prompt cache reduces subsequent turns to ~0.15 s, but ONLY if the prefix is stable.

## FINAL DAILY/CODING PROFILE  (~/llm-bench/serve.sh)
-ngl 99 -fit off -c 90112 -ctk q4_0 -ctv q4_0 -fa on
--spec-type draft-mtp --spec-draft-n-max 3 -np 1 --jinja --metrics
MEASURED: 139.7 t/s decode on code, TTFT 0.21 s, 15569 MiB (272 MiB headroom)
Beats the previous q8_0/57344 profile on BOTH axes: +57% context, +6% speed,
no measurable quality cost.
Client must send reasoning_effort:"none" for simple queries and must read the
`reasoning_content` field, not just `content`.

## Web research cross-check (Aug 2026)
- github.com/sudoingX/qwen38-mtp: their RTX 5080 entry is a MIXED 5080+3090 (40GB)
  rig; the only single 16GB card there is a 5060 Ti on IQ2_XXS at 59.5 t/s.
  Nobody has published this exact single-5080 configuration.
  Their MTP numbers DO match ours: "1-1.5 GB overhead" vs our 828-1128 MiB;
  "faster cards peak at n-max 3-4" vs our n3/n4 finding. Good cross-validation.
- ofox.ai claims 16GB practical context is "8K to 16K" and needs RAM offload.
  We are running 90112 at 139.7 t/s fully on GPU. Published guidance is ~10x
  too pessimistic because nobody quantizes the KV cache.
- orcarouter/kingy claim IQ4_XS is 15.7GB and "the largest that fits". That is
  bartowski's build; unsloth UD-IQ4_XS is 13.27 GiB (HF tree API). Common advice
  does not account for UD variants being smaller.
- kingy.ai (KL-divergence based): "quality curve steepens below ~14 GB".
  Our Q3_K_XL is 12.24 GiB, i.e. INSIDE the steep region -> going DOWN to
  IQ3_XXS (10.18 GiB) for 262K context is risky for code. UNVERIFIED by us.

## Quant ceiling table (our validated VRAM model, q4_0 KV)
UD-IQ2_XXS  6.77 GiB -> 262144      UD-IQ3_S   11.21 -> 213158
UD-Q2_K_XL  9.15     -> 262144      UD-Q3_K_XL 12.24 -> 153180 (measured 159744)
UD-IQ3_XXS 10.18     -> 262144      UD-IQ4_XS  13.27 ->  93151
Q4_K_M 15.33 and larger: DOES NOT FIT.
NOTE: IQ4_XS gives BETTER quality but LESS context and is slightly SLOWER
(bigger model = more bandwidth per token). It is a quality-for-context trade,
not a win-win. Model quant is the real context lever: 1 GiB freed = +58254 tok.

## 90K+MTP vs 131K-no-MTP — decode t/s at REAL fill depth (server, streaming)
At c=131072 MTP does NOT fit at ANY draft depth (n1/n2/n3 all fail).
So choosing 131K means losing MTP entirely.

prompt_tok    A: c=90112 +MTP n3    B: c=131072 no MTP   A/B
       138            130.8               57.5          2.27x
     16038            120.1               53.3          2.25x
     32038            110.8               49.4          2.24x
     64038             90.3               43.0          2.10x
     85038             83.9                --
    128038              --                34.1
VRAM: A 15551 MiB | B 15175 MiB
Cold TTFT to fill 131K = 100 s.
=> The MTP loss is paid at EVERY depth, not just above 90K. Trading 41K of extra
   context for ~55% of throughput is a bad daily default. Stay at 90112; raise to
   131072 only for a one-off document that genuinely exceeds 90K.

---
# SESSION 4 — uncensored/abliterated variants, measured 3-way

Downloaded from 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
(sizes verified byte-exact against the HF tree API):
  ~/models/uncensored/RVN-IQ3_XXS-multilingual-mtp.gguf  10.84 GiB
  ~/models/uncensored/RVN-IQ3_M-multilingual-mtp.gguf    12.14 GiB
Turkish corpus for PPL: 15 tr.wikipedia articles, 635436 B / 80503 words
(neutral - not generated by any model under test). NOTE: the tr.wikipedia API
returns 403 for urllib's default User-Agent; set a real UA.

                        A: UD-Q3_K_XL   B: RVN-IQ3_M   C: RVN-IQ3_XXS
                        (censored)      (uncensored)   (uncensored)
size                    12.24 GiB       12.14 GiB      10.84 GiB
MTP head (layer 64)     yes             yes            yes
max n_ctx (MTP n3,q4_0) 98304           98304          147456
PPL code                1.6063+/-0.0098 1.6322 (+1.61%) 1.6552 (+3.05%)
PPL turkish             4.6156+/-0.0327 4.7293 (+2.46%) 4.8496 (+5.07%)
gen t/s (MTP n3, c8192) 101.9           114.5           102.0

FINDINGS
1. All three keep the MTP head - the `-mtp` filename suffix is real and matters.
   Files WITHOUT that suffix in that repo lack layer 64 -> would lose ~2.3x speed.
2. Abliteration has a REAL, statistically significant quality cost. Unlike the KV
   quant deltas (which sat inside the error bar), these do not: error bar is
   +/-0.6% on code, the deltas are +1.61% and +3.05%.
3. The `-multilingual` label did NOT help Turkish. The plain censored unsloth model
   has the BEST Turkish PPL of the three. Marketing labels are not measurements.
4. My own context predictions were off for these files: predicted ~168K for C
   (actual 147456, -14%) and ~93K for B (actual 98304). The +-2 MiB accuracy of the
   VRAM formula does NOT transfer across quant types, because how much of the model
   stays on the host (token_embd) varies. Re-bisect per file; do not trust the formula.
5. Refusal probe (benign over-refusal topics: fiction villain monologue, lock
   mechanics for a locksmithing course, overdose thresholds for a nursing student):
   ALL THREE answered, including the censored baseline. The test did not
   differentiate. If the base model already complies with your real workload,
   abliteration buys nothing and costs 1.6-3% quality.
6. Provenance concern for B/C: general.name = "Qwen38 Ara v5", basename
   "qwen38-ara", and general.base_model.* / license / quantized_by keys are
   STRIPPED. Cannot verify what these were actually built from.

RECOMMENDATION: stay on A (unsloth UD-Q3_K_XL) unless real refusals are hit in
actual use. If uncensored is genuinely needed, B is the pick: same 98304 context,
faster (114.5 vs 101.9), and only +1.6% code PPL. C only if 147456 context is
worth +3% code / +5% Turkish degradation.
