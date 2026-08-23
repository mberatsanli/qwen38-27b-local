---
title: "A control run killed my best theory about 3-bit quantisation"
description: "I found a failure pattern in a 3-bit model that had a mechanism and a spectacular example. Then I ran the control, and the pattern turned out to be denser without the variable I blamed."
pubDate: 2026-08-23
slug: kv-cache-quantisation-null-result
tags: ["llm", "llama.cpp", "benchmarking", "gpu"]
draft: true
---

Asked for twenty balls bouncing in a spinning heptagon, my local model wrote this, 153 times:

```js
const MAX_SPEED_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_WALL = 3200;
```

Thirty-seven `_BALL`s in one identifier. It burned all 14,000 tokens of its budget on
constants and wrote none of the simulation. No canvas, no loop, nothing. The model is
Qwen3.8-27B quantised to 3 bits, running on one RTX 5080, and I thought I knew what this
meant.

I was wrong, and the run that showed me took twenty minutes.

## Only 16 of the 64 layers have a KV cache

The reason a 27B model fits any meaningful context in 16 GB is architectural. Its GGUF
metadata gives it away:

```
general.architecture     = qwen35
qwen35.block_count       = 65
qwen35.full_attention_interval = 4
qwen35.attention.head_count_kv = 4
qwen35.attention.key_length    = 256
```

`full_attention_interval = 4` means every fourth layer is real attention. Reading the tensor
list confirms it. `attn_k`/`attn_v` tensors exist on layers 3, 7, 11 … 63 and nowhere else.
The other 48 layers are DeltaNet, linear attention with a recurrent state that stays a
constant 149 MiB no matter how long the context gets.

So the KV cache is sized by 16 layers, not 65. At f16 that's 64.0 KiB per token; the
published guidance I'd been reading assumed a dense model and quoted "8K to 16K context" for
this card. I measured a ceiling of 159,744 tokens, entirely on the GPU, by bisecting real
allocation.

## Quantising the KV cache costs 0.09% perplexity

`q4_0` stores the cache at 18.0 KiB per token against f16's 64.0. That is 3.6× more context
for the same memory. The question is what it costs. Over 104,694 tokens of llama.cpp's own C++
source:

```
c=8192    f16 1.6057 ±0.0097   q8_0 1.6059   q4_0 1.6063
c=32768   f16 1.4909 ±0.0059                 q4_0 1.4923
```

The q4_0 column moves by +0.037% at 8K and +0.094% at 32K. Both are inside the error bar by
a factor of 4 to 16. On this evidence the cheap cache is free, and that single fact is what
turns a 40K-context card into a 160K one.

Then I stopped trusting it.

## Perplexity is a mean over text someone else wrote

Perplexity is teacher-forced. It measures how surprised the model is by a document it did
not produce. It says nothing about staying consistent with 8,000 tokens of its own output,
which is what code generation is.

And there's a mechanism to suspect. While generating, the model attends over its own
previous tokens, and those live in the KV cache. Quantise the cache to 4 bits and the
model's memory of what it wrote 400 tokens ago is lossy. If that mattered, it would show up
as a specific shape: correct setup written, then a consumer written as though the setup
weren't there.

So I ran the community's one-shot tests: pelican on a bicycle, Flappy Bird, a Mandelbrot
explorer, a solar system, the heptagon. Every artifact was reviewed for defects, with a
second pass whose only job was to refute the first.

## The shape was in every artifact

Flappy Bird's loop does the delta-time calculation correctly:

```js
function loop(t){
  let dt = (t - last) / 16.667;
  if(dt > 3) dt = 3;
  last = t;
  update(dt);
```

Then `update(dt)` never multiplies anything by `dt`. Not once. `grep` finds the variable
only in the loop above and as the unused parameter. Physics runs per-frame: `p.x -=
PIPE_SPEED`, `bird.vy += GRAVITY`. On my 60 Hz monitor the tuning is right. On a 144 Hz one
the same file is a different game.

![A Flappy Bird clone with pipes, a score counter and a bird mid-flight, rendered in a browser](/images/flappy-oneshot.webp)
*The game runs and is playable. Two pipes are on screen at once with 154 px of clear air between them, because PIPE_SPACING is 220 px on a canvas capped at 440 px.*

The Mandelbrot renderer writes palette stops as 0–1 floats into a `Uint8ClampedArray`, so
every colour rounds to 0 or 1. A swatch builder 35 lines away does the `*255` correctly. The state machine assigns `state = 'dead'` and three input handlers compare
against `'over'`, so keyboard restart is dead code. The solar system's eight orbital periods
are right to three decimals against real values, sitting inside an animation that is broken
in other ways.

Every one of these has the same signature: the correct thing is present in the same file,
often within 40 lines, and the consumer ignores it. Plus the heptagon, which never got as
far as contradicting a plan.

## The control tied 24 to 24

Same model file, same five prompts, same seed 42, same temperature 0, same speculative
draft depth. One variable changed: `-ctk`/`-ctv` from `q4_0` to `f16`.

| | q4_0 KV | f16 KV |
|---|---|---|
| confirmed defects | 24 | 24 |
| artifacts completed | 4 / 5 | 5 / 5 |
| same state as the other arm | | 4 of 5 |

An exact tie. And the shape I was chasing is at least as common in the control, which
supplied the purest examples I found anywhere: `legendRows` pushed as `{p: row}` and
destructured as `{p, row}`, so every legend update throws. `scale * factor` written with the
correct sense in the pinch handler and inverted in click, wheel and keyboard. A
ground-collision branch that inlines the first two statements of `gameOver()` and drops the
rest, softlocking the game on the most common death.

Both arms produced the same defect at the same site, too. Under q4_0, Flappy computes `dt`
and never applies it. Under f16 it declares `let last = 0`, accepts the `rAF` timestamp, and
reads neither. Same failure, same function, different cache precision.

Here is the image that nearly convinced me anyway:

![Two SVG pelicans on bicycles side by side. The left one has orange legs shooting off the top edge of the frame and a detached body; the right one has legs that connect the body to the pedals and a neck attached to the head](/images/pelican-kv-comparison.webp)
*Same prompt, same seed, KV dtype the only difference. The q4_0 bird's legs run off the top edge because one endpoint was written as `cy` on a `<line>`, an attribute `<line>` doesn't have, so it defaults to the origin. One attribute name.*

That is one artifact. The Mandelbrot is broken in both arms, by a temporal-dead-zone
`ReferenceError` under q4_0 and a duplicate `const rect` under f16. Both screenshots are a
blank gradient. Four of the five land in the same bucket.

## What survived is one prompt, and one prompt can't carry it

The heptagon is the only real difference. Under f16 the same prompt at the same seed wrote a
complete simulation in 3,775 tokens, with an impulse solver, mass weighting and substepping.

![Twenty numbered balls resting in a heap at the bottom of a dark heptagon outline](/images/heptagon-f16.webp)
*The f16 run produced a running simulation. The q4_0 run produced 153 constant declarations and no canvas. This is the entire measurable difference between the two arms.*

It's a genuine attention-to-own-output failure and it's the shape the theory predicts. It's
also 1 of 5 versus 0 of 5, which is Fisher exact p = 1.0. On counts near 24 the Poisson
interval is roughly ±10, so this design can't resolve anything smaller than a 40% change in
defect rate.

Temperature 0 doesn't rescue it either. Changing the KV dtype changes the logits, so one
argmax flip early forks everything downstream. These are two draws from adjacent
distributions, not one generation perturbed.

Verdict: insufficient evidence.

## What I did not check

- **Reviewer calibration.** Different agents graded each arm at different levels of detail.
  One tagged 11 of 24 defects as "knew better", the other 21 of 24. I'm reading that 46%
  versus 88% split as the reviewers, not the model, and I have no blind re-grade to prove it.
- **Whether q4_0's heptagon would have recovered.** The 14,000-token cap is my harness
  limit, and it only ever bound one arm.
- **Context size parity.** The q4_0 arm ran at `-c 90112` and the control at `-c 24576`,
  because f16 can't reach 90K on this card. Allocated context shouldn't change the maths for
  a 16K sequence, but I didn't verify that empirically.
- **q8_0 as a third point.** A real precision effect should order f16 ≥ q8_0 ≥ q4_0. A
  two-point comparison can't show monotonicity.
- **Whether any of this is about quantisation at all.** There's no unquantised baseline
  here, so I can't separate "3-bit weights" from "this model, one shot, no repair turn".

## The next run

Defect count needs a reviewer, and reviewers vary. The arms only ever differed on whether
the generation ran away, and that's binary and free to count. Thirty prompts per arm,
several seeds, a raised ceiling so a runaway gets measured instead of truncated. If you've
run something like this and got a different answer, I'd like the seed and the flags.
