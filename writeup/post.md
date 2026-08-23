---
title: "I blamed the KV cache, then found the same bugs without it"
description: "I got a 27B model running on my gaming GPU, asked it to draw a pelican and write Flappy Bird, and thought I'd worked out what the cheap settings were costing me. Then I checked."
pubDate: 2026-08-23
slug: kv-cache-quantisation-null-result
tags: ["llm", "llama.cpp", "gpu", "side-project"]
draft: true
---

I asked the model on my desktop to draw a pelican riding a bicycle. It's a silly test that
Simon Willison started, and the point is that the scene can't be in the training data, so
the model has to work it out. Mine gave me this:

![Two SVG pelicans on bicycles side by side. The left one has orange legs shooting off the top edge of the frame and a body floating above the bike; the right one has legs that reach the pedals and a neck attached to the head](/images/pelican-kv-comparison.webp)
*Left is the cheap setting I was suspicious of, right is the expensive one. Same prompt, same seed.*

The bicycle is fine both times. Spokes, chain, handlebars, a frame that makes sense. The
left bird's legs shoot off the top of the picture, and I spent a while convinced I knew why.

The actual reason is one word. Line 78 of that file draws a leg as
`<line x1="155" cy="195" x2="180" y2="265">`, and `<line>` has no `cy` attribute, so that
end of the leg defaults to the corner of the image. It does it twice. That is a typo, not a
memory problem, and it took me longer than it should have to go and look.

## Why I was poking at this at all

I'd been trying to get something big running on a 16 GB gaming card. Qwen3.8-27B, squashed
down to 3 bits so it fits in 12.24 GiB, on an RTX 5080.

Guides I found said 16 GB gets you 8K to 16K of context, or 4-bit weights with some of
the model spilling into system RAM. I ended up at **90,112 tokens**, all of it on the GPU: 130 tok/s
on a short prompt, about 84 once there's 85K of conversation behind it. Filling that cold
costs roughly a hundred seconds of chewing before the first word appears, which is the real
limit rather than the memory.

Here's the whole setup, so the numbers below mean something:

```bash
# RTX 5080 16 GB, driver 595.84, CUDA 13.3, Ubuntu 26.04
# llama.cpp b10588 (70adb1b4c), built with -DCMAKE_CUDA_ARCHITECTURES=120a-real
# model: unsloth/Qwen3.8-27B-GGUF, UD-Q3_K_XL, 12.24 GiB

llama-server -m Qwen3.8-27B-UD-Q3_K_XL.gguf -ngl 99 -fit off \
  -c 90112 -ctk q4_0 -ctv q4_0 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 3 --jinja
```

Every generation below used seed 42 and temperature 0. `-fit off` matters more than it
looks: llama.cpp will quietly shrink `-c` to whatever fits and not tell you, so a run that
doesn't error isn't a run that got the context you asked for. I lost a session's worth of
numbers to that before I noticed.

Most of that gap comes from one setting nobody seems to touch. The KV cache holds the
model's memory of the conversation so far, and llama.cpp will store it at 4 bits instead of
16. That's 18 KiB per token instead of 64, so **3.6 times the context for the same memory**.

What does that cost? I ran perplexity over a big pile of C++ and got
+0.037% at 8K context and +0.094% at 32K, both well inside the noise. So: free.

I didn't believe it.

## Perplexity grades the model on someone else's writing

Perplexity feeds the model a document it didn't write and measures how surprised it is. It
never asks the model to stay consistent with 8,000 tokens of its own output, which is the
whole job when it's writing a game.

And there's a reason to be suspicious. While it writes, the model looks back at what it
already wrote, and that lives in the KV cache. Squash the cache to 4 bits and its memory of
what it said 400 tokens ago gets fuzzy. If that mattered, I'd expect a particular kind of
bug: the model sets something up correctly, then forgets to use it.

So instead of perplexity I had it write things. Flappy Bird, a Mandelbrot explorer, a solar
system, twenty balls bouncing in a spinning heptagon, and the pelican. One shot each, no
follow-up messages, nothing fixed by hand.

## It kept setting things up and then ignoring them

Flappy Bird works. It's playable, the pipes have gaps, the score counts.

![A Flappy Bird clone in a browser with green pipes, a score of zero and a small bird mid-flight](/images/flappy-oneshot.webp)
*Playable, but the pipes are 220 px apart on a canvas 440 px wide, so there are always two on screen with about 154 px of gap. I never got past the first one.*

Inside, the game loop works out how much time passed since the last frame, clamps it, and
hands it to the update function:

```js
function loop(t){
  let dt = (t - last) / 16.667;
  if(dt > 3) dt = 3;
  last = t;
  update(dt);
```

And then `update(dt)` never uses `dt`. Not once. Gravity and pipe speed are applied per
frame, so on my 60 Hz monitor it's tuned right and on a 144 Hz one it'd be a different game.
The model clearly knew the correction was needed. It wrote it, passed it in, and never
multiplied by it.

The Mandelbrot does the same thing in a different costume. It writes its colours as numbers
between 0 and 1 into an array that only accepts 0 to 255, so every colour rounds to black.
Two hundred lines further down, the swatch builder does the same conversion with the `*255`
in place.

Then the heptagon, which is what really convinced me. Asked for twenty bouncing balls, it
wrote this 153 times:

```js
const MAX_SPEED_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_BALL_WALL = 3200;
```

Thirty-seven `_BALL`s in one name. It got as far as a `<canvas>` tag, a 2d drawing context
and eleven sensible physics constants, then spent the rest of its 14,000 tokens naming things
and never wrote the loop. A model losing track of what it wrote thirty lines ago is what a
fuzzy memory of its own output should look like.

## Turning it off changed nothing

I had a model read every file and list what was wrong with it, then had a second one try to
knock each finding down. A bug is one thing I'd have to go and fix. Different graders read
each run and clearly at different levels of fussiness, so read 24 to 24 as "about the same"
rather than an exact match.

Same model, same five prompts, same seed. Two things changed, not one: the cache, and `-c`.
The 4-bit run had 90,112 tokens of context and the 16-bit one 24,576, because a 16-bit cache
physically can't reach 90K on this card at 64 KiB per token. I don't think the allocated
size changes the maths for a 16,000 token generation, but I didn't check that.

|  | 4-bit cache | 16-bit cache |
|---|---|---|
| bugs found | 24 | 24 |
| files that came out complete | 4 of 5 | 5 of 5 |

Four of the five prompts ended up in the same state in both runs. Same number of bugs, and
the same *kind* of bugs. The 16-bit run has a legend that
builds rows one way and reads them another, so it crashes on every update. It has a zoom
factor written the right way round in one place and upside down in five others. It has a
death branch that copies the first two lines of the game-over function and drops the rest,
so falling on the ground softlocks the game.

Flappy Bird even broke the same way twice. The 4-bit run works out the frame time and never
uses it. The 16-bit run declares a variable to hold the frame time, takes the timestamp as
an argument, and reads neither.

So the pattern I'd been collecting isn't a fingerprint of the cheap cache. It happens as much
without it.

If you're wondering how the same seed and temperature 0 produced different files at all:
changing the cache changes the numbers coming out of the model slightly, and one different
choice near the start sends everything after it somewhere else. These are two neighbouring
runs, not the same run nudged.

## What survived is one run, and one run isn't much

The heptagon is the only place the two runs really differ. With the 16-bit cache, the same
prompt wrote a working simulation in 3,775 tokens.

![Twenty numbered balls piled at the bottom of a dark heptagon outline](/images/heptagon-f16.webp)
*The 16-bit run made this. The 4-bit run made 153 constants and no canvas. That's the whole difference between them.*

It's the exact failure I predicted, in the arm I predicted it in. It's also one prompt out
of five, from one seed. If I flipped a coin five times and got one head, I wouldn't conclude
much either.

Two things are worth keeping apart here, because I mashed them together the first time I
explained it to someone. "That pattern comes from the 4-bit cache" is dead, because the
pattern turns up without it. "The 4-bit cache makes long writing drift" is still standing,
untested, with one run pointing at it. I didn't disprove the idea. I disproved my reason for
believing it.

## What I actually run

Still the 4-bit cache. The quality cost I can measure is smaller than the noise, it buys me
3.6 times the context, and the one thing arguing against it is a single weird run. If a long
generation ever falls into a loop like the heptagon did, I'll switch the cache back to 16
bits and see, and then I'll have two data points instead of a theory.

## Things I didn't check

- Whether that heptagon would have recovered if I'd let it run past 14,000 tokens. That
  limit is mine, and it only ever stopped one of the two runs.
- Whether any of this is about the 4-bit weights, the 4-bit cache, or what happens when
  you ask for 8,000 lines of code in one shot with no chance to test it. I have no
  full-precision version to compare against.
- 8-bit as a middle point. If the cache really matters, 16-bit should beat 8-bit should beat
  4-bit, and two points can't show a line.
- Whether the perplexity number covers what I use it for. Both measurements are at 32K or
  below, I run at 90,112, and the cost roughly tripled between the two points I have. I
  couldn't take a third: a 16-bit cache can't reach 65K on this card to compare against.
- Anything on a second machine. This is one card, one model, one afternoon.

Every artifact from both runs and the timings are in
[qwen38-27b-local](https://github.com/mberatsanli/qwen38-27b-local), along with the prompts
and the script that ran them. The machine-readable list of what's wrong with each file
covers the 4-bit run only. The header-reading tool that works out a model's context ceiling before
you download it is in [gguf-envelope](https://github.com/mberatsanli/gguf-envelope).

My two runs only really differed on whether the model went off the rails, and counting that
needs no opinion about code quality. Thirty prompts each, a few seeds, a bigger token budget
so a runaway gets measured instead of cut off. If you've tried this and got a different
answer, I'd like your seed and your flags.
