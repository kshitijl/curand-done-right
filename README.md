# curand-done-right
Don't use stateful RNGs on the GPU!

It is very common for massively parallel computations to require large
amounts of random numbers. For example, monte carlo integration is
compute-heavy and embarrassingly parallel or nearly so, a natural fit
for the GPU. Examples of monte carlo integration include
physically-accurate rendering based on path tracing; diffusion monte
carlo for high-accuracy quantum chemistry calculations; statistical
machine learning in the Bayesian inference framework; and the pricing
of securities in finance.

The generation of random numbers may end up being the bottleneck for
these computations. "That's ok," you say, "I have an obscene amount of
floating-point hardware resources on my GPU." Unfortunately, the API
to most random number generation schemes imposes memory bandwidth and
latency costs that limit the utilization of GPU compute resources.

## What do I mean by API?

Most random number generation schemes can be divided into two parts:
state transition and output. There is some internal state `S` and two
functions `t(S)` and `o(S)`. Every time you ask for a random number
you use the output function `o(S)` to turn the current state `S` into
the output random number. Then, you apply the state transition
function `t(S)` to obtain the next state `S_(i+1)` from state `S_i`.

Many random number generators put all their intelligence in the state
transition function and have a trivial output function: the identity.
Others split the work into both functions. The kind that's best for
parallel applications puts all the hard work into the output function
and has a trivial transition function: incrementing a counter.

This stateful setup is good for serial machines: to generate a lot of
random numbers, a serial machine simply generates them one after
another. However, a parallel machine has to maintain `N` different
random number generators each with their own internal states, run
transition and output functions for each one independently and then
store the states back.

It seems like the work of the transition function has been wasted: if
we were only going to use the state once, why bother with a transition
at all? And there is the question of how to initialize `N` independent
internal states without running the transition function `N` times in
serial.

## Other people have done the hard work

Random123 and Fortuna

## cuRAND ships with all the necessary pieces

## How to use this library

### Uniform- and gaussian-distributed single-precision floats

So, for example

```
    auto randoms = curanddr::uniforms<2>(uint3{0,0,0},
                                         uint2{index, seed});
```

generates two uniform floats in `[0,1]` and return a fixed-size array
that can be indexed to retrieve the two random numbers.

### Don't repeat input bits!
The fundamental rule is that you must pass in a unique set of 5
integers each time you ask for some random numbers. Passing in the
same set of 5 integers will give you exactly the same output, and
incrementing them in sequence will recapitulate a sequence you have
seen before. This is both helpful, in the sense that deterministic
programs are easier to debug and maintain, but also a potential
pitfall.

So here are some rules of thumb that will make it easy to avoid this
problem and the sticky statistical issues that come with it.

### Generate many random numbers together, in multiples of 4

## Examples

### Estimating pi

### Verifying the mean and variance of gaussian randoms

### Miller-rabin primality test

### Brownian motion

## Bonus: a reason to deliberately reuse sequences of random numbers
