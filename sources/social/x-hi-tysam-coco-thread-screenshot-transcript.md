# X thread screenshot transcript: Fern on CoCo augmentation

Source URL: https://x.com/hi_tysam/status/2059292395892523105?s=20

Author: Fern (`@hi_tysam`)

Date shown by X: May 26, 2026

Capture provenance: user-provided screenshots pasted into the assistant conversation on 2026-05-28. The root tweet text was also retrievable through public X embed/oEmbed-style endpoints. X did not expose the full thread to the unauthenticated fetch tools available during ingestion.

Limitations: this is a screenshot-derived transcript of visible thread text, not an authenticated X archive. The first visible screenshot post is partly obscured by an X profile popover, so it is marked as partially visible/reconstructed. Image diagrams are summarized rather than reproduced as raw image pixels because the screenshot files were not available in the local workspace.

## Root tweet

> A little over 2 years ago, I solved the SolidGoldMagikarp stability problem.
>
> Today, I am releasing the results of that work as a new technique to regularize training.
>
> More details below.

## Visible thread transcript

### Post 1: teacher forcing setup

Partially visible/reconstructed due to profile popover overlay:

> Autoregressive transformers have a core problem that limits their decoding performance over other generation techniques. This technique has been around for a while, and we train them massively in parallel. But it has a significant catch that has plagued transformers for years.

Attached image summary: an RNN/teacher-forcing diagram comparing inference without teacher forcing to training with teacher forcing. The lower panel shows ground-truth tokens feeding the model during training.

### Post 2: train/inference mismatch

> With teacher forcing, we input the “correct” input to the network at every step, instead of what the network predicts during inference. This creates behavior that works well in the infinite limit, but here’s the catch: we only train our networks for a finite amount of time!

Attached image summary: a plot titled “Teacher forcing only approaches the correct algorithm.” It shows a blue teacher-forcing trajectory and a red autoregressive-inference trajectory diverging over steps, with a final gap labeled epsilon sub T. Caption notes that teacher forcing matches the correct answer at every step, but the model only sees its own outputs during inference, so per-step errors compound and the terminal gap can stay positive even as training loss approaches zero.

### Post 3: permanent deficit

> This creates a permanent deficit. No matter what, we will always have a train-inference gap. Any token has a chance of being a “bad” token, and the model never really learns to recover from it. To my best knowledge, these tokens can destabilize inference.

### Post 4: Continuous Coding augmentation

> You can see this very clearly if you dump garbage tokens into a model — via ICL, the model picks up on the distribution, and tries to mimic the original mess. This is really bad.
>
> We can solve this with a new inductive bias. I call it the Continuous Coding augmentation (CoCo).

Attached image summary: a CoCo augmentation diagram showing input embeddings `x_i`, additional scaled random embeddings `alpha x_j`, and resulting augmented embeddings `x'_i`. Caption states that for each `i`, draw `j ~ Uniform({1,...,n})` independently and form `x'_i = x_i + alpha x_j`; `alpha` is fixed and a fresh `j` is drawn at every training step.

### Post 5: algorithm sketch

> The premise is simple: at every training step, for every input embedding, simply select another random input embedding, multiply it by a fixed value (e.g. .3), and add it to the original input embedding. That’s it.
>
> This single modification solves 2 separate problems at once. Why?

### Post 6: inductive priors

> Well, we can turn to the question of inductive priors. Under default teacher forcing, we have an (assumed) noiseless input, pointed at an (assumed) noiseless output.
>
> In practice, neither of these are true! This makes the network OOD right out of the gate.

### Post 7: on-manifold recovery

> But when we choose a uniformly-random token for the input at each step, we are selecting from an infinite (albeit unlikely) sequence space for corrupting our network inputs. So what does this do?
>
> It forces the network to _learn_ how to remain on-manifold during inference.

### Post 8: constant input distortion

> This is because, at every step during training, the network has a constant input distortion. Not only must the network look more thoroughly at surrounding tokens for context, but also, it now has a fixed “capacity” for trying to constantly return to being on-manifold every step.

Attached image summary: a 3D manifold diagram comparing “no augmentation (drifts off-manifold)” to “with CoCo (returns on-manifold)” from a shared start point. Caption: “Fig. 1 inference trajectories from a shared on-manifold token.”

### Post 9: rare embeddings

> Secondly, it gives the network constant exposure to rare embeddings. Because we add the embeddings to the input at every step, this means that the network now has access to _all of the embeddings contrasted against each other_.
>
> This is incredibly powerful. Why?

### Post 10: rare-token signal

> Because it gives the network signal for rare token embeddings. Instead of playing a gambling game, where you may see a rare token once every many, many batches, instead, you’re seeing the embedding for the token every batch. It no longer becomes rare or undefined behavior.

### Post 11: 125M recovery demo

> One of my absolute favorite examples showing this is an example I did a little over 2 years ago with a tiny, 125M network where I gave an astronomical prompt, added 60 garbage tokens, then watched it recover. The network actively fights its way upstream back onto the manifold.

### Post 12: SolidGoldMagikarp stability claim

> Most networks would fail. This one did not. It slowly recovered and returned to the original topic. No RL. No fancy inference tricks. Completely random sampling at temperature 1.
>
> This is a new kind of ICL. The stability of SolidGoldMagikarp has been solved.

Attached image summary: a terminal-like screenshot labeled `llb-gpt v0.6.0 prerelease` with annotations describing a 125M augmented teacher-forcing robustness inference demo at temperature 1, random sampling, and 60 random tokens. Colored callouts mark where the model starts on-topic, encounters garbage tokens, gradually recovers, and restores performance.

### Post 13: provenance note

> This is formerly unreleased work from about 2 years ago that was sponsored by my Patreon sponsors, as well as other generous sponsors. I am glad to be able to share it with you today.
