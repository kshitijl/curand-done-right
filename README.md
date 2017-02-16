# curand-done-right: A header-only library for fast parallel random numbers on the GPU
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

Even if we solve that problem, this API has doomed us to waste memory
bandwidth and latency writing and reading RNG state.

## Other people have done the hard work

The solution to this problem is to get rid of the transition function,
or rather to make it so trivial that it is easy to apply it `N` times
in constant time. An example of such a transition function is
counting: the internal state is a number, and it is incremented every
time a new output random number is desired.

But then what output function should we use? The key insight here is
to exploit the fruits of research into encryption. A good encryption
algorithm has the property that if we change any one bit in the input
(either in the plaintext or the key), then _every_ bit of the output
is equally likely to change. An algorithm that has this property makes
it difficult to try to discover the plaintext by perturbing inputs to
the algorithm and seeing if it gets closer to the known ciphertext.

Well, that property is very useful for making a random number
generator as well! Simply encrypt a counter with some key, and every
time we increment the counter (change some bits), _every_ bit of the
output is equally likely to change -- in other words, a new set of
random bits has been generated.

This idea has come up at least twice that I know of: Random123 and
Fortuna are two random number generators that follow this scheme. They
consist of specific choices of encryption algorithm, rounds of
encryption, block size and so forth that have been chosen to be
efficient and produce sufficiently random output.

Note that this kind of RNG implies a different API. There is no need
to talk of a state transition function, and no need for writing and
reading state. Simply ask for the `N`th random number. If you need
them one after another, ask for the `N`th, `N+1`th and so forth in
serial. If you need a million random numbers in parallel then have
each parallel unit (thread in CUDA) ask using a different counter.
Since each thread already has some index variables that distinguish it
from every other thread, this is a very natural setup for CUDA
programs.

Of course, the caveat is that if you ask for the `N`th random number
twice, you will get the same result twice. Instead of explicitly
managing state, you must implicitly ensure that the RNG is not called
twice with the same input bits within the same calculation.

## cuRAND ships with all the necessary pieces

The best part of this story is that NVIDIA ships the Random123 RNG
as one of the choices available in cuRAND. However, since all the
cuRAND RNGs share the same interface, the advantage of using this kind
of RNG is lost. It's silly to store and load state that consists of a
couple of counters!

So you must eschew the use of cuRAND APIs and directly call the
primitive encryption functions yourself. I grew tired of doing that
over and over and wrapped it up in a small header-only library with a
sensible interface.

## How to use this library

This is a header-only library for device code. Call either
`curanddr::uniforms` or `curanddr::gaussians` from within your CUDA
kernels. Pass in five `uint` numbers; together, these make up a 160
bit counter that results in a unique random number being generated.

The five `uint`s are divided into a `uint4` struct and a single
`uint`, which I call `counter` and `key`. However there is nothing
special about them: just pass in five numbers and if even one of those
160 bits is different, the output random numbers will be different. I
just think two arguments is better than five; one argument would have
been best but `uint5` is not a predefined type.

Both functions take a template argument: this is how many random
numbers will be returned. Fixing this at compile-time allows the CUDA
compiler to generate parsimonious code.

The return type is an array of floats, 

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

1. Use application-level indices as counters. So if your application
   has an index for each thread and an outer index for each iteration,
   then use both of these indices as two of the five `uint` counters
   you get to pass in.
2. Differentiate between calls on different source text lines using
   constants. Since you probably don't have five levels of nesting in
   your application, some of the five `uint` counters will be left
   constant. You can pass in different constants (I prefer 1,2,3...)
   for different locations in your code where the RNG is being called.
   This ensures that even if application-level indices are the same
   for two invocations, their results will be uncorrelated.

### Generate many random numbers together, in multiples of 4

While you can generate one random number at a time, the underlying
encryption algorithm works in blocks of 128 bits at a time. At least
four random `uint`s are generated every time you call one of the
functions (`uniforms` or `gaussians`), while returning to you the
amount you asked for. If you ask for 5 then 8 will be generated, of
which the first 5 will be returned. So if you can rearrange your
computation to use random numbers in blocks of 4 (or 8, or 12 ...) at
a time then you will achieve higher utilization of compute resources.

## Examples

### Estimating pi

### Verifying the mean and variance of gaussian randoms

### Miller-rabin primality test

### Brownian motion

## Bonus: a reason to deliberately reuse sequences of random numbers
