# Notes on ART and MoD-ART theory

These notes assume some level of familiarity with acoustic radiance transfer literature.

## Fundamental premise of ART: radiance, power, and etendue

The room acoustic rendering equation defines the propagation of radiance $L(x_i, x_j)$ between surface points.
It is an integral equation, and the radiance itself is a function of two surface points (the point $x_i$ radiance departs from, and the point $x_j$ radiance is directed towards).
The fundamental premise of acoustic radiance transfer is that, after discretizing the surface into a finite number of patches, point-to-point radiance will be approximately constant for all pairs of points on a pair of surface patches:

$$
    L(x_i, x_j) = \text{const}
    \quad \forall x_i \in A_i, \forall x_j \in A_j .
    %\approx
    %L(x_i, x_j)
$$

Under that assumption, the *discretized radiance* propagated by ART is given by said approximately constant value for each pair of surface patches &mdash; which may either be *averaged* over the pair of patches, or *integrated* over the pair of patches.
Both are valid choices, and different works in the literature disagree on this definition.
We take the time to discuss the choice here because it affects the way the ART scattering matrix is defined, as well as the weighting which needs to be applied to inputs and outputs of the ART system.

In RAVES, and by extension in this implementation of ART, we *integrate* radiance in each propagation path.
This means that the physical quantity being propagated between pairs of surface patches is acoustical *power*:

$$
    P_{i \to j}
    =
    \iint_{A_i}
    \iint_{A_j}
    L(x_i, x_j)
    \mathrm{d}G(\mathrm{d}A_i, \mathrm{d}A_j)
    ,
$$

where $\mathrm{d}G(\mathrm{d}A_i, \mathrm{d}A_j)$ is the differential *etendue* between a pair of differential area elements, defined as

$$
\begin{aligned}
    \mathrm{d}G(\mathrm{d}A_i, \mathrm{d}A_j)
    &{}=
    \frac{\cos\theta_{ij}\,\cos\theta_{ji}}{\lVert x_i - x_j\rVert^2}
    \mathrm{d}A_j
    \mathrm{d}A_i
    % \\ &{}=
    =
    \cos\theta_{ij}
    \mathrm{d}\Omega_j
    \mathrm{d}A_i
    .
\end{aligned}
$$

Taking the integral of $\mathrm{d}G(\mathrm{d}A_i, \mathrm{d}A_j)$ over both surface patches gives the full path etendue, $G_{i \to j}$.
Under the assumption that the radiance $L_{i \to j}$ is constant within the propagation path, it can be taken out of the integral, giving

$$
\begin{aligned}
    P_{i \to j}
    &{}=
    L_{i \to j}
    \iint_{A_i}
    \iint_{A_j}
    \mathrm{d}G(\mathrm{d}A_i, \mathrm{d}A_j)
    ,
    \\
    P_{i \to j} &{}= L_{i \to j} G_{i \to j}
    \iff
    L_{i \to j} = \frac{P_{i \to j}}{G_{i \to j}}
    .
\end{aligned}
$$

The etendue is what translates radiance to power and vice-versa: this property is essential for the input-output operations of ART, as discussed in the following.



## Considerations for using MoD-ART in RAVES

### ART recursion loop format

Let us consider the structure of ART when it is seen as a signal processing system involving feedback.
In particular, whether $A$ and $T_{a}(z)$ are in the feed-forward or feed-back path.
To start with, let's introduce a redundant (but hopefully clear) notation: $s_1(z)$ is the current input of each delay line (i.e., propagation path), and $s_2(z)$ is the current output of each delay line.
We envision the time delay induced by propagation along a path as a discrete-time delay line, which internally holds a number of state variables, and shifts them one at a time.
In this context, $s_1(z)$ is the *future first* element of each line's inner state, $z^{-1}s_1(z)$ is the *current first* element of each line's inner state, and $s_2(z)$ is the *current last* element of each line's inner state.
(In some of our papers, we used the symbol $s(z)$ in reference to $s_1(z)$, while in others it referenced $s_2(z)$. Sorry about that!)

Let's start by noting that the recursive loop is agnostic to the concept of ``feed-forward or feed-back path''.
The operation of the $A \to T_{a}(z)$ loop is univocally expressed in the state space as

$$
    \overline{s}(z) =
    z^{-1} \overline{A} \overline{s}(z)
    ,
$$

where $\overline{s}$ is the full state vector of the ART system, and $\overline{A}$ is the full state transition matrix of the ART system.
The eigenvalues $\Lambda$ and (state-space) eigenvectors $\overline{V}$, $\overline{W}$ are uniquely determined by the state transition matrix $\overline{A}$, and by extension, of the $A \to T_{a}(z)$ loop: they are not influences by the way inputs and outputs are positioned in the loop.

Using $s_1(z)$ and $s_2(z)$, the loop operation takes the form

$$
\begin{aligned}
    s_1(z) &{}= A s_2(z) ,
    \\
    s_2(z) &{}= T_{a}(z) s_1(z) .
\end{aligned}
$$

The full state vector $\overline{s}(z)$ contains both $z^{-1}s_1(z)$ and $s_2(z)$.
**N.B.: $\overline{s}(z)$ does *not* contain $s_1(z)$ explicitly &mdash; only implicitly, as $z^{-1}s_1(z)$ &mdash; because $s_1(z)$ is not part of the delay lines' state variables.**
We do not need to concern ourselves with where these elements reside exactly; the ordering of elements in $\overline{s}(z)$ is irrelevant as long as $\overline{A}$ is defined to match it.
The concept of "feed-forward or feed-back" path comes into play when inputs and outputs taps are placed in the loop, as we discuss later in this section.

The state-space eigenvalues $\Lambda$ always correspond to the MoD-ART $\Lambda$, and the state-space $\overline{V}$, $\overline{W}$ always *contain* the MoD-ART $V$, $W$ &mdash; similarly to how $\overline{s}(z)$ contains $z^{-1}s_1(z)$ and $s_2(z)$.
In the case of $\overline{s}(z)$ containing $z^{-1}s_1(z)$ and $s_2(z)$, the element locations are a fixed property of the loop.
In the case of $\overline{V}$, $\overline{W}$ containing $V$, $W$, the element locations depend on the relationship between system inputs and outputs with respect to $s_1(z)$ and $s_2(z)$.
We explain this relationship in the following.

### MoD-ART eigenvectors

The ART system structure we use here is informed by the use of MoD-ART eigenvectors in RAVES (at runtime).
The operation performed at runtime is as follows.
Energy* emanated from the sound source is delayed according to the distance between the source and the closest reflecting surface.
It is not reflected nor scaled by any surface reflection coefficient.
The signals injected into the system therefore act like signals exiting propagation paths, about to be scattered &mdash; in other words, $s_2(z)$.
The same ray-tracing operation is performed from the listener: the energy* detection accounts for the distance between the listener and the closest reflecting surface, without being reflected nor scaled by any surface reflection coefficient.
The detected signals therefore act like signals leaving scattering surfaces, about to be propagated &mdash; in other words, $s_1(z)$.

\* "Energy" is a bit of a misnomer here; the next section discusses what these quantities actually are in the physical sense.

Using the equivalence $s_1(z) = A s_2(z)$, the full system is then represented as

$$
\begin{aligned}
    s_2(z) &{}= T_\mathrm{a}(z) A s_2(z) + B(z) x(z) ,
    \\
    y(z) &{}= C(z) A s_2(z) + D(z) x(z) ;
    \\
    H(z) &{}= C(z) A \left[I - T_\mathrm{a}(z)A\right]^{-1} B(z) + D(z) .
\end{aligned}
$$

Reframing the same using only $s_1(z)$:

$$
\begin{aligned}
    s_1(z) &{}= A T_\mathrm{a}(z) s_1(z) + A B(z) x(z) ,
    \\
    y(z) &{}= C(z) s_1(z) + D(z) x(z) ;
    \\
    H(z) &{}= C(z) \left[I - AT_\mathrm{a}(z)\right]^{-1} A B(z) + D(z) .
\end{aligned}
$$

The objective of modal decomposition is to achieve the form

$$
\begin{aligned}
    H(z) &{}= C(z) V \left[zI - \Lambda\right]^{-1} W^H B(z) + D(z)
    \\ &{}=
    \sum_{i=1}^{M} \frac{C(z) w_i v_i^H b(z)}{z - \lambda_i} + D(z) .
\end{aligned}
$$

Let's remember that, in the input-output configuration we have selected, $B(z)$ are intended to feed directly into $s_2(z)$, while $C(z)$ are intended to feed directly from $s_1(z)$.
As such, $W$ are the elements of $\overline{W}$ related to $s_2(z)$, and $V$ are supposed to be the elements of $\overline{V}$ related to $s_1(z)$ &mdash; but as we've said, $s_1(z)$ does not appear explicitly in the state vector $\overline{s}(z)$, so we need to take the elements of $\overline{V}$ related to $s_2(z)$ and left-multiply them by $A$, because $s_1(z) = A s_2(z)$.

A final note: the algorithm we use for decomposition finds left and right eigenvectors separately, and then relates one to the other.
Algorithms which directly locate left/right pairs are only available for the decomposition of dense matrices, whilst accounting for the sparsity of $A$ and $\overline{A}$ is paramount in our case.
Besides having to perform the decomposition twice (once for each side), this means that the left/right pairs we locate are "mismatched" by an unknown scalar factor.
In order for the decomposition to hold, the left and right vectors must uphold

$$
    \overline{W}^H \overline{V}
    = \overline{V}^{-1} \overline{V}
    = I ,
$$

which is achieved by ensuring the dot product of each left/right pair is exactly 1.


### ART injection and detection operators

We earlier said that the input operators $B(z)$ describe how "acoustic energy" emanated from sound sources is injected into the discrete propagation paths between surface patches, and the output operators $C(z)$ describe how "acoustic energy" inside the discrete propagation paths is perceived by listeners.
Let's be a bit more formal about what these quantities actually are.
Calling them "energy" is a misnomer; energy is the integral of power over a duration of time, and here we are talking about quantities which are a function of time.
In the context of acoustics, input and output signals are the **acoustic intensity** at positions of sound sources and listeners.
In the context of radiometry, what we call acoustic intensity is instead called **radiosity** or **irradiance** depending on whether it is emitted or received (don't blame me, I'm just reporting the facts), and should not be confused with **radiant intensity** which is a different quantity (*don't blame me, I'm just reporting the facts*).

#### What does this mean in practice?

Since the quantity being propagated within the recursive ART loop is *power*, this means that the input operators $B(z)$ need to translate *radiosity to power* and then partition that power across different propagation paths.
The output operators $C(z)$ need to translate *power to irradiance* before combining the contributions from different propagation paths.

The first part is easy, at least for a point source.
The power emanated by a point source in the direction of a surface patch is equal to the source radiosity multiplied by the solid angle subtended by the surface patch.
For those who are unfamiliar with solid angles: think about taking a "fish-eye lens" view from the point source, laying it onto the surface of a unit-radius sphere, and measuring the area occupied by the patch on the sphere's surface.
This is the solid angle of the patch, as seen from the point source.Note that the sum of all solid angles seen by the source is equal to the total area of the unit sphere: $4\pi$.
In practice, this is evaluated by tracing a number $N_\omega$ of rays from the point source position (with directions uniformly covering the sphere), counting the number of rays hitting each surface patch, dividing the results by $N_\omega$ (which makes their sum equal to 1) and finally multiplying by $4\pi$.

The output operation is similar, but has an additional step.
Irradiance at a position is given by an integral of incoming radiance over solid angle &mdash; which, under the assumption of constant radiance in each propagation path, means a finite sum of discrete radiance values multiplied by discrete solid angles.
The solid angles themselves can be evaluated exactly like they are from the source positions (trace, bundle, normalize, $4\pi$), but this time we also need to translate the propagated *power* signals into *radiance*.
We saw how to do this in the first section: $L_{i \to j} = \frac{P_{i \to j}}{G_{i \to j}}$.
The output operators need to divide the propagated power by the path etendue.

In the RAVES implementation, the ray-tracing steps (find intersections, bundle per patch, divide by $N_\omega$) are performed at runtime.
To save some runtime multiplications, the eigenvectors $V$ and $W$ are multiplied by $4\pi$ during the pre-processing.
The left vectors $W$ are also divided by the relevant path etendues $G_{i \to j}$ to enact the power-to-radiance translation.
All of these scaling factors are baked into the eigenvectors saved in the output file `MoD-ART.csv`.

Lastly, when computing residues, we need to make sure we're evaluating $B(\lambda_i)$ and $C(\lambda_i)$ instead of just $B(1)$ and $C(1)$.
For example, one element of $B(\lambda_i)$ is defined as ${e_\mathrm{b} \lambda_i^{-\tau_\mathrm{b}}}$, where $e_\mathrm{b}$ is the contribution *scaling* and $\tau_\mathrm{b}$ is the *delay* in samples.
If we say that $t_\mathrm{b}$ is the same delay in seconds, then ${\tau_\mathrm{b} = t_\mathrm{b} f_e}$ where $f_e$ is the sample rate used to run the ART system (not the audio sample rate).
If we say that $\sigma_i$ is the energy decay per second (whereas $\lambda_i$ is the energy decay per sample), then ${\lambda_i = \sigma_i^{1/f_e}}$.
Then,

$$
    e_\mathrm{b} \lambda_i^{-\tau_\mathrm{b}}
    =
    e_\mathrm{b} \lambda_i^{-t_\mathrm{b} f_e}
    =
    e_\mathrm{b} \left(\sigma_i^{1/f_e}\right)^{-t_\mathrm{b} f_e}
    =
    e_\mathrm{b} \sigma_i^{-t_\mathrm{b}}
    % =
    % \frac{e_\mathrm{b}}{\sigma_i^{t_\mathrm{b}}}
    .
$$

This scaling is applied when contributions to (and from) propagation paths are computed in RAVES, at runtime.
Similarly, the compensation for the initial delay of the whole late reverberation section is $\sigma_i^{t_d}$ where $t_d$ is in seconds.



## Reflection kernel

The ART kernel is given by

$$
\begin{aligned}
    F_{h \to i \to j}
    &{}=
    \iint_{A_h}
    \iint_{A_i}
    \iint_{A_j}
    \rho(x_h, x_i, x_j)
    \left[(x_h - x_i) \in \Omega_h\right]
    \left[(x_j - x_i) \in \Omega_j\right]
    \frac{\cos\theta_{ij}\,\cos\theta_{ji}}{\lVert x_i - x_j\rVert^2}
    \mathrm{d}A_j
    \frac{\mathrm{d}A_i}{A_i}
    \frac{\mathrm{d}A_h}{A_h}
    \\ &{}=
    \iint_{A_h}
    \iint_{A_i}
    \iint_{\Omega_j}
    \rho(x_h, x_i, x_j)
    \left[(x_h - x_i) \in \Omega_h\right]
    \cos\theta_{ij}
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}A_i}{A_i}
    \frac{\mathrm{d}A_h}{A_h}
    ,
\end{aligned}
$$

where $\left[(x_h - x_i) \in \Omega_h\right]$ is a visibility term equal to $1$ if $x_h$ is visible from $x_i$ and $0$ otherwise, and $\cos \theta_{ij} = n_i \cdot (x_j - x_i)$.
Note that we incorporate visibility in the definition of differential solid angle.

Note that ${\iint_{A_h} \frac{\mathrm{d}A_h}{A_h}}$ indicates an averaging integration.
In the following, instead of averaging the integrand as ${\iint_{A_h} \frac{\mathrm{d}A_h}{A_h}}$, we use an averaging solid angle integral ${\iint_{\Omega_h} \frac{\mathrm{d}\Omega_h}{\Omega_h}}$.
This gives the kernel definition

$$
    F_{h \to i \to j}
    =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \rho(x_h, x_i, x_j)
    \cos \theta_{ij}
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    .
$$

This results in a different weighting, but both approximations converge to the room acoustic rendering equation.

Since we use ray-tracing for the numerical evaluation of solid angle integrals, visibility terms are implicitly enforced by the ray-tracing.
In the case of obstruction, $\Omega_j$ is the part of $A_j$ which is visible from $x_i$.

### Diffuse kernel component

#### Evaluation

In the diffuse case, the BRDF is constant: ${\rho(x_h, x_i, x_j) = \frac{1}{\pi}}$.
The kernel is

$$
\begin{aligned}
    F_{\text{diff } h \to i \to j}
    &{}=
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \frac{1}{\pi}
    \cos \theta_{ij}
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    \\ &{}=
    \iint_{A_i}
    \iint_{\Omega_j}
    \frac{\cos \theta_{ij}}{\pi}
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}A_i}{A_i}
    \quad \forall h
    .
\end{aligned}
$$

The outer integral with $\frac{\mathrm{d}A_i}{A_i}$ means that its integrand is averaged over all points in $A_i$.
In practice, for numerical integration, we can evaluate ${\iint_{\Omega_j} \frac{\cos \theta_{ij}}{\pi}\mathrm{d}\Omega_j}$ at a set of sample points on $A_i$ and average the results.
The inner integral is a solid angle integral.
We can uniformly sample $\Omega_j$ by taking uniform directions in the hemisphere around $n_i$ and selecting only the directions which fall inside $\Omega_j$ (with ray-tracing).
Then, if the full hemisphere is sampled with $N_\omega$ directions, ${\mathrm{d}\Omega_j = \frac{2\pi}{N_\omega}}$ and

$$
\begin{aligned}
    \iint_{\Omega_j}
    \frac{\cos \theta_{ij}}{\pi}
    \mathrm{d}\Omega_j(\omega_j)
    &{}\approx
    \frac{2}{N_\omega}
    \sum\nolimits_{\omega_j \in \Omega_j}
    \cos \theta_{ij}
    ,
    \\
    \iint_{A_i}
    \iint_{\Omega_j}
    \frac{\cos \theta_{ij}}{\pi}
    \mathrm{d}\Omega_j(\omega_j)
    \frac{\mathrm{d}A_i}{A_i}
    &{}\approx
    \frac{2}{N_\omega N_x}
    \sum\nolimits_{x_i \in A_i}
    \sum\nolimits_{\omega_j \in \Omega_j}
    \cos \theta_{ij}
    .
\end{aligned}
$$

As a side note,

$$
    F_{\text{diff } h \to i \to j}
    =
    \frac{G_{i \to j}}{\pi A_i}
    \quad \forall h
    .
$$

#### Validation

We can use two properties of form factors to assess the accuracy of the numerical integration.
Form factor unity summation (provided the surface is closed), and etendue symmetry:

$$
\begin{aligned}
    \sum_{j=1}^{n}
    F_{\text{diff } h \to i \to j}
    &{}= 1
    \quad \forall h
    ,
    \\
    % \pi A_i F_{\text{diff } h \to i \to j}
    % =
    G_{i \to j}
    &{}=
    G_{j \to i}
    % =
    % \pi A_j F_{\text{diff } h \to j \to i}
    % \quad \forall h
    .
\end{aligned}
$$

Both of these equalities are exact in theory, but approximate in practice, due to discretization of the integrals.
The incurred error acts as an assessment of the integration accuracy.

### Specular kernel component

#### Evaluation

The specular BRDF is

$$
\begin{aligned}
    \rho(x_h, x_i, x_j)
    &{}=
    \frac{\delta(\text{spec}(x_h - x_i) - (x_j - x_i))}{\cos \theta_{ih}}
    \\ &{}=
    \frac{\delta(\text{spec}(x_h - x_i) - (x_j - x_i))}{\cos \theta_{ij}}
    \\ &{}=
    \frac{\delta(\text{spec}(x_j - x_i) - (x_h - x_i))}{\cos \theta_{ih}}
    \\ &{}=
    \frac{\delta(\text{spec}(x_j - x_i) - (x_h - x_i))}{\cos \theta_{ij}}
    .
\end{aligned}
$$

which gives

$$
\begin{aligned}
    F_{\text{spec } h \to i \to j}
    &{}=
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \frac{\delta(\text{spec}(x_j - x_i) - (x_h - x_i))}{\cos \theta_{ij}}
    \cos \theta_{ij}
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    \\ &{}=
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \delta(\text{spec}(x_j - x_i) - (x_h - x_i))
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    \\ &{}=
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \delta((x_j - x_i) - \text{spec}(x_h - x_i))
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    .
\end{aligned}
$$

We can remove an integral thanks to the delta's sifting property ${\int_{-\infty}^{\infty} f(t) \delta(t-T) \,\mathrm{d}t = f(T)}$.
Before we do, let us make the visibility term w.r.t. $\Omega_j$ explicit again:

$$
\begin{aligned}
    F_{\text{spec } h \to i \to j}
    &{}=
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \left[(x_j - x_i) \in \Omega_j\right]
    \delta((x_j - x_i) - \text{spec}(x_h - x_i))
    \mathrm{d}\Omega_j
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    \\ &{}=
    \iint_{A_i}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h - x_i) \in \Omega_j\right]
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    .
\end{aligned}
$$

The innermost integrand $\left[\text{spec}(x_h - x_i) \in \Omega_j\right]$ is equal to 1 if the direction *specular to* $(x_h - x_i)$ falls within $\Omega_j$ and 0 otherwise.
In practice, for numerical integration, taking the average of a "boolean" integrand like this means counting the number of sample points (i.e., rays) for which the condition is true.

$$
\begin{aligned}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h - x_i) \in \Omega_j\right]
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    &{}\approx
    \frac{
    \sum\nolimits_{\omega_h \in \Omega_h}
    \left[\text{spec}(\omega_h) \in \Omega_j\right]
    }{
    \sum\nolimits_{\omega_h \in \Omega_h}
    1
    }
    ,
    \\
    \iint_{A_i}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h - x_i) \in \Omega_j\right]
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    &{}\approx
    \frac{1}{N_x}
    \sum\nolimits_{x_i \in A_i}
    \frac{
    \sum\nolimits_{\omega_h \in \Omega_h}
    \left[\text{spec}(\omega_h) \in \Omega_j\right]
    }{
    \sum\nolimits_{\omega_h \in \Omega_h}
    1
    }
    .
\end{aligned}
$$

With that said, in practice we carry out the averaging over $x_i \in A_i$ slightly differently, so that the results are closer to the kernel's unity summation property:

$$
    \iint_{A_i}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h - x_i) \in \Omega_j\right]
    \frac{\mathrm{d}\Omega_h}{\Omega_h}
    \frac{\mathrm{d}A_i}{A_i}
    \approx
    \frac{
    \sum\nolimits_{x_i \in A_i}
    \sum\nolimits_{\omega_h \in \Omega_h}
    \left[\text{spec}(\omega_h) \in \Omega_j\right]
    }{
    \sum\nolimits_{x_i \in A_i}
    \sum\nolimits_{\omega_h \in \Omega_h}
    1
    }
    .
$$
