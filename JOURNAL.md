# Development Journal

A chronological log of the experiments, design decisions, setbacks, and
lessons behind this project.

## June 2026

### 2026-06-28

-   I started building the reconstruction phase for learning, drawing on
    both Ogma Corp's whitepaper and AOgmaNeo. As I worked through them,
    I realized that they differ in several important ways, so I decided
    to follow the whitepaper for now.
-   I also noticed that the model is biased toward the center,
    column-wise. Columns are not normalized against one another,
    although the cells within each column are. AOgmaNeo addresses this
    with Fuzzy ART, but I don't expect it to be a problem here because
    the whitepaper's learning rule operates locally, at the cell level
    within each column.

### 2026-06-29

-   Improved encoder performance.
-   Compared ERE and Fuzzy ART as encoder architectures. I initially
    tried implementing ERE, but ultimately switched to Fuzzy ART, mainly
    for its accuracy and performance.

## July 2026

### 2026-07-01

-   Significantly improved the encoder's forward-pass performance,
    measuring the changes with `perf`.
-   Decided to drop AOgmaNeo's multiple-visible-layer approach to keep
    the architecture simpler.

### 2026-07-03

-   Implemented encoder learning.
-   Precomputed the complete local receptive field and the learning
    receptive fields.
-   Found a major source of inefficiency that made the code extremely
    slow despite the micro-optimizations. Using `perf` to inspect cache
    misses, I found that roughly 30% of the encoder's cache misses came
    from dictionary indexing through `index_cell`, a pseudo-random index
    supplied as input.
-   Started considering cache blocking as a possible solution.

### 2026-07-07

-   I had already exhausted several approaches to improving encoder
    cache behavior, including loop inversion and cache blocking. None
    helped, so I decided to move on to the decoder and hope the model
    would be fast enough despite the cache issue.

### 2026-07-08

-   Switched the decoder's forward-pass softmax from regular `exp` to
    `exp2`. The approximation I'm using is effective over the range
    `[-128, 128)`, and I expected it to improve precision.
-   Considered replacing ReLU with Elliott's sigmoid as the dendritic
    activation function. This could simplify the algorithm: without
    exploding activations, I might no longer need online softmax or an
    expensive transcendental function. I would only need to change the
    forward activation and use its derivative during learning. I planned
    to test this after finishing the model.

### 2026-07-10

-   Started implementing the main `Sph` struct. I planned to write the
    whole thing first and split it into smaller commits afterward.

### 2026-07-11

-   While implementing exponential memory, I quickly realized that an
    efficient implementation would require dedicated processing logic
    for each individual layer, similar to AOgmaNeo.

### 2026-07-12

-   It turned out that a full visible-layer abstraction was not
    necessary for exponential memory after all. I wondered why AOgmaNeo
    used it. My understanding at the time was that it supported
    processing several sensory inputs simultaneously; since the first
    encoder did not need multiple input formats, I thought multiple
    visible layers were unnecessary for this SPH.
-   The `Sph` struct was becoming frustrating to implement. Keeping the
    sizes of each encoder-decoder pair synchronized while preserving
    exponential memory was a constant source of bugs. The assertions
    helped catch them, but the work was taking entire afternoons.

### 2026-07-15

-   I decided to take a break from coding. I had been feeling
    emotionally confused and increasingly stressed by school and judo. I
    needed some time to be a normal teenager: watch anime, spend time
    with friends and family, and enjoy things without turning everything
    into another project.
-   If anyone is reading this, I hope you understand. I was tired, and I
    didn't think I could keep up the pace I'd been maintaining.

### 2026-07-26

-   Back in the game. After taking a long break to deal with exam
    stress, a judo competition, and the emotional mess of being a
    teenager, I was ready to get back to work.
-   I began by reimplementing the SPH API struct because the logic I had
    been building on was fundamentally flawed.
-   I leaned on an LLM to help build the `step` function. It worked,
    although it still had some bugs, and it saved me a lot of time. This
    was the first time I had added AI-generated code to the project. I
    had wanted to build everything from scratch to develop the strongest
    possible mental model of the architecture, because there was
    genuinely a lot to understand at once. "Claude is my shepherd; I
    shall not want."
-   With the new `Sph` struct, the bugs were finally becoming
    manageable. They were not easy to solve, but at least they were no
    longer impossible.

### 2026-07-28

-   The SPH was finally working and learning effectively.
-   I tested the Elliott sigmoid idea. It improved performance
    noticeably, but introduced precision loss. After enough samples, the
    decoder began to forget, with a substantial loss of precision. I
    decided to keep the existing decoder because the performance gain
    was negligible compared with the precision cost over time.
-   After struggling with exponential memory, I began to question
    whether it was necessary for this model at all. Since the model
    would serve as the policy and value head of an MCTS chess engine,
    whether Lc0 or a custom implementation, it did not need temporal
    intelligence. It only needed to evaluate the current search node.

### 2026-07-31

-   During long simulation runs, the model began losing precision. Its
    smoothed average accuracy could peak around 90%, then fall to
    roughly 50--60%.
-   I suspected weight drift or catastrophic forgetting, but wasn't
    sure.
-   I confirmed that the problem was weight saturation. The decoder's
    integer weights use `i8` precision, with a representable range of
    `[-128, 128)`. I addressed this by scaling dendritic and activation
    values using a `decoder_scale` hyperparameter and the
    receptive-field count, borrowing the idea from AOgmaNeo. The model
    then converged to roughly 95% accuracy on a test set with 5% input
    noise.

## August 2026

### 2026-08-01

-   I tried building a training pipeline around Lc0's inputs and
    outputs. I initially thought it would be best to write a paper about
    the engine's performance and CPU inference first, but integrating
    the model with Lc0 or Ceres quickly became a wall of work. I had to
    dig through both engines' code to understand their input and output
    encodings, and the integration did not seem worth the effort.
-   I decided it would be better to build my own chess engine so that I
    could integrate the model and search tightly.

### 2026-08-02

-   I couldn't work on the project because I devoted the day to
    homework.
-   School was becoming increasingly boring and stressful. I was
    frustrated by spending so much time on assignments that felt
    disconnected from actual learning. I wished classes would challenge
    and stimulate me instead of measuring responsibility through hours
    of busywork.

### 2026-08-09

-   I hadn't written an update in a while. I had been trying to build a
    basic training pipeline to test whether the model could learn chess
    concepts, but the codebase was not ready for it.
-   The code was difficult to read and maintain, lacked the necessary
    chess pre-encoders and pre-decoders, and had no parallelization,
    which slowed training considerably. I spent these days refactoring
    the codebase and preparing it to run over a dataset.
-   I know SPHs are intended to learn during inference, but I wanted to
    find out whether this model could learn chess at all.

### 2026-08-11

-   Realized that the lack of precision in `fast_math::exp()` was
    hurting the model's accuracy. I switched to `f32::exp()` and
    simplified the softmax pass into a single function.

### 2026-08-14

-   I started reconsidering Lc0 integration. Now that I had read more of
    its code and understood how its input planes work and how to write a
    custom backend, it seemed much less difficult than I had initially
    thought. I might try that first and build my own MCTS only if it
    fails.
-   For now, I decided to focus on model training and parallelization
    before attempting Lc0 integration.
-   Flamegraphs showed that a significant amount of time was being spent
    in `rand_round`. I hadn't run a `perft` benchmark, but the
    `Decoder::forward` section had become noticeably smaller, so I
    counted that as a win.

### 2026-08-15

-   Started parallelizing the model. So far, I had only found noticeable
    speedups in the decoder's learning step, so I planned to parallelize
    that part first.
-   The likely explanation was that Rayon's scheduling overhead
    outweighed the gains elsewhere. Scheduling overhead also seemed to
    dominate the decoder's learning step, though, so I planned to keep
    an eye on it.

### 2026-08-16

-   Added validation-set testing with help from an LLM.
-   Added position transformations during pre-encoding to match Lc0's
    conventions, also with help from an LLM.
-   Fixed a bug that kept the policy head's loss just above
    random-guessing levels and its top-1 training accuracy around 15%.
    After spending the day investigating, I found a synchronization bug:
    the policy and value heads were receiving learning data from `t-1`
    instead of `t`.
-   After fixing it, the policy head stabilized at roughly 37% top-1
    accuracy on the validation set after training on around 16,250 games
    from the dataset.
-   The value head was still struggling and appeared to be guessing
    randomly. But if the policy head could learn, I believed the value
    head could too.

### 2026-08-21

-   I felt I had been going in circles, so I decided to create a
    dedicated training loop for the value head. I would freeze the rest
    of the model and tune only the value head to see how it behaved.

### 2026-08-22

-   Considered replacing training on raw PGNs with training on actual
    Lc0 chunks. I wanted to see whether this would help the value head
    learn something instead of guessing, and it might benefit the policy
    head as well.

## September 2026

### 2026-09-06

-   I wondered why I had slowed down so much. The truth was that I
    simply didn't feel like working on it. Coding is a hobby, and I only
    want to do it when I feel like it. I had slowed down to avoid
    growing to hate something I currently loved.
-   After testing on Lc0 chunks and training directly on probability
    distributions, I noticed that the model learned faster, but the
    value head remained a mystery.
-   I considered detaching body learning from head learning again. That
    would let me shuffle positions independently and might finally help
    the value head learn.

### 2026-09-07

-   I began questioning whether my validation method was appropriate.
    The dataset was so large that I could comfortably use only about 1%
    of it for validation. That subset might not represent the full
    variety of positions, causing validation loss to spike at certain
    points and look like overfitting.
-   I had assumed that 1% would be enough simply because the dataset was
    so large, but I was starting to consider alternatives.
-   I also wondered whether I had been interpreting the logs
    incorrectly. After making the model smaller and training it for
    longer, I saw the policy head improve smoothly while the value head
    remained noisy. I had interpreted that noise as overfitting or
    regression, but perhaps it was simply a noisy plateau.

### 2026-09-08

-   Built a quick-and-dirty Lc0 backend for the model. It was much more
    straightforward than I had expected.
-   When I tested the network, however, it always output `0 cp`. Very
    strange.

### 2026-09-12

-   After a lot of experimentation, I was told that the heads' receptive
    fields should cover their entire input. They need to turn the `t+1`
    prediction into a small set of categories.
-   I tried training the value head with a receptive-field radius of
    `r = 9`. It was remarkably slow, but it produced something better
    than `0 cp`.
-   My explanation was that a `3 × 3` local field forced the model to
    generalize too much complexity, so the best it could do was average
    everything out and call every position a draw.

### 2026-09-20

-   Instead of simply training the value head with `r = 9`, I
    reconsidered the decoder design. Since every cell needed to see
    every visible cell, the head's computation was massively
    parallelizable.
-   I implemented a specialized decoder called `Head`. It skips local
    receptive fields entirely, parallelizes the forward pass at the cell
    level rather than only at the column level, and supports masking.
    During training, this lets the policy head avoid computing logits
    for illegal moves.
-   This specialized decoder effectively solved the dead-value-head
    problem: the model finally produced values other than `0 cp`.
-   However, both the policy and value heads still plateaued very early,
    or learned so slowly and noisily that progress was difficult to
    recognize.
-   I suspected the encoders might be suffering from category
    proliferation and stopping their learning too early. But after
    logging the cells' commitment rate, which was around 10% after
    10,000 games, the encoder seemed less suspicious. This was a raw
    rate that also counted illegal or unreachable categories, but even
    so, only 10% did not seem alarming. The Fuzzy ART encoder might
    actually be generalizing.
-   I planned to experiment with learning-rate schedules for the heads,
    including reduce-on-plateau, 1cycle, and cosine annealing.

### 2026-09-26

-   I finally understood why visible layers were necessary. When the
    hidden state and feedback are concatenated as the decoder's input
    tensor, each hidden cell sees only one of the two, not both at once.
    This also makes the decoder's local receptive field much more
    scattered across the visible planes. So visible layers really are
    necessary.
-   The same issue applies to the encoder: because the auxiliary data
    occupies only one row of columns, not all hidden columns can see it.
-   I also realized that I was packing too many changes into single
    commits, leaving the repository without a useful development
    history. I needed to improve my commit discipline.

## October 2026

### 2026-10-07

-   Unfortunately, the hypothesis seemed to be false. My concern was
    right: in raw chess strength, this kind of model underperformed
    compared with alternatives such as ResNets and Transformers.
-   I suspected two main reasons:
    1.  **Local learning rules may be weaker at feature extraction.**
        Backpropagation tunes every parameter against the same
        objective, directly minimizing loss on the dataset. Local
        learning rules optimize according to other criteria, such as
        topological similarity in Fuzzy ART. Perhaps Gaussian ARTMAP
        could improve the encoder.
    2.  **Strict top-1 WTA and CSDR may discard too much information.**
        This is a trade-off for speed and preserves roughly 99%
        sparsity. It might be worth experimenting with top-4 CSDR
        representations, although that would increase computational cost
        and reduce sparsity to roughly 96%, according to an earlier
        estimate.
-   I wasn't going to abandon the project. The model might still
    improve, but it was becoming clear that beating backpropagation in
    raw dataset accuracy would be extremely difficult for a
    local-learning architecture.
-   Perhaps the model should serve a different purpose: online learning.
    But that was not what this paper was trying to measure.
-   I decided to create control groups so I could compare the model
    against other architectures under the same constraints.

### 2026-10-11

-   The first control group was a small CNN with 32 channels,
    implemented in PyTorch and trained with AdamW without a
    learning-rate scheduler.
-   Despite being significantly smaller, the CNN beat my SPH by a huge
    margin in training time, accuracy, and development time.
