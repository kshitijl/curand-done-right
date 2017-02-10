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

Most random number generation schemes can be
divided into two parts: state transition and output.

## Other people have done the hard work

Random123 and Fortuna

## cuRAND ships with all the necessary pieces

## How to use this library correctly

The fundamental rule is that you must pass in a unique set of 5
integers each time you ask for some random numbers. Passing in the
same set of 5 integers will give you exactly the same output, and
incrementing them in sequence will recapitulate a sequence you have
seen before. This is both helpful, in the sense that deterministic
programs are easier to debug and maintain, but also a potential
pitfall.

So here are some rules of thumb that will make it easy to avoid this
problem and the sticky statistical issues that come with it.

## Bonus: a reason to deliberately reuse sequences of random numbers
