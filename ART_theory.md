# Notes on ART and MoD-ART theory

## Model definition / structure

TODO: Specify the structure we use for decomposition (arrangement of input and output taps in the recursive loop).

TODO: Specify difference between propagating power or averaged radiance.
TODO: Show the integral definitions of power and averaged radiance.
TODO: Give explicit definition of etendue integral:
The element $\cos\theta_{ij}\, \mathrm{d}\Omega_j\, \mathrm{d}A_i$ is differential etendue between two points, and etendue is defined by
$$
    G_{i \to j}
    =
    \iint_{A_i}
    \iint_{\Omega_j}
    \cos\theta_{ij}
    \, \mathrm{d}\Omega_j
    \, \mathrm{d}A_i
    \, .
$$
TODO: Show that power and averaged radiance can be translated between each other via the etendue.



## Eigenvector scaling

TODO: Explain how and why eigenvectors are re-scaled for use at runtime.



## Reflection kernel

The ART kernel is given by
$$
\begin{aligned}
    F_{h \to i \to j}
    & =
    \iint_{A_h}
    \iint_{A_i}
    \iint_{A_j}
    \rho(x_h, x_i, x_j)
    \left[(x_h\!-\!x_i) \in \Omega_h\right]
    \left[(x_j\!-\!x_i) \in \Omega_j\right]
    \frac{\cos\theta_{ij}\,\cos\theta_{ji}}{\lVert x_i - x_j\rVert^2}
    \, \mathrm{d}A_j
    \, \frac{\mathrm{d}A_i}{A_i}
    \, \frac{\mathrm{d}A_h}{A_h}
    \\ & =
    \iint_{A_h}
    \iint_{A_i}
    \iint_{\Omega_j}
    \rho(x_h, x_i, x_j)
    \left[(x_h\!-\!x_i) \in \Omega_h\right]
    \cos\theta_{ij}
    \, \mathrm{d}\Omega_j
    \, \frac{\mathrm{d}A_i}{A_i}
    \, \frac{\mathrm{d}A_h}{A_h}
    \, ,
\end{aligned}
$$
where $\left[(x_h\!-\!x_i) \in \Omega_h\right]$ is a visibility term equal to $1$ if $x_h$ is visible from $x_i$ and $0$ otherwise, and $\cos \theta_{ij} = n_i \cdot (x_j\!-\!x_i)$.

Note that ${\iint_{A_h} \frac{\hbox{d}A_h}{A_h}}$ indicates an averaging integration. In the following, instead of averaging the integrand as ${\iint_{A_h} \frac{\hbox{d}A_h}{A_h}}$, we use an averaging solid angle integral ${\iint_{\Omega_h} \frac{\hbox{d}\Omega_h}{\Omega_h}}$. We also leave visibility terms implied unless otherwise specified. This simplifies the kernel definition to
$$
    F_{h \to i \to j}
    =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \rho(x_h, x_i, x_j)
    \cos \theta_{ij}
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \, .
$$
This results in a different weighting, but both approximations converge to the room acoustic rendering equation.

Since we use ray-tracing for the numerical evaluation of solid angle integrals, visibility terms are implicitly enforced by the ray-tracing. In the case of obstruction, $\Omega_j$ is the part of $A_j$ which is visible from $x_i$.

### Diffuse kernel component

#### Evaluation

In the diffuse case, the BRDF is constant: ${\rho(x_h, x_i, x_j) = \frac{1}{\pi}}$. The kernel is
$$
\begin{aligned}
    F_{\text{diff } h \to i \to j}
    & =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \frac{1}{\pi}
    \cos \theta_{ij}
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \\ & =
    \iint_{A_i}
    \iint_{\Omega_j}
    \frac{\cos \theta_{ij}}{\pi}
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}A_i}{A_i}
    \quad \forall h
    \, .
\end{aligned}
$$
The outer integral with $\frac{\hbox{d}A_i}{A_i}$ means that its integrand is averaged over all points in $A_i$. In practice, for numerical integration, we can evaluate ${\iint_{\Omega_j} \frac{\cos \theta_{ij}}{\pi}\, \hbox{d}\Omega_j}$ at a set of sample points on $A_i$ and average the results. The inner integral is a solid angle integral. We can uniformly sample $\Omega_j$ by taking uniform directions in the hemisphere around $n_i$ and selecting only the directions which fall inside $\Omega_j$ (with ray-tracing). Then, if the full hemisphere is sampled with $N_\omega$ directions, ${\hbox{d}\Omega_j = \frac{2\pi}{N_\omega}}$ and
$$
\begin{aligned}
    \iint_{\Omega_j}
    \frac{\cos \theta_{ij}}{\pi}
    \, \hbox{d}\Omega_j(\omega_j)
    & \approx
    \frac{2}{N_\omega}
    \sum\nolimits_{\omega_j \in \Omega_j}
    \cos \theta_{ij}
    \, ,
    \\
    \iint_{A_i}
    \iint_{\Omega_j}
    \frac{\cos \theta_{ij}}{\pi}
    \, \hbox{d}\Omega_j(\omega_j)
    \frac{\hbox{d}A_i}{A_i}
    & \approx
    \frac{2}{N_\omega N_x}
    \sum\nolimits_{x_i \in A_i}
    \sum\nolimits_{\omega_j \in \Omega_j}
    \cos \theta_{ij}
    \, .
\end{aligned}
$$

As a side note,
$$
    F_{\text{diff } h \to i \to j}
    =
    \frac{G_{i \to j}}{\pi A_i}
    \quad \forall h
    \, .
$$

#### Validation

We can use two properties of form factors to assess the accuracy of the numerical integration. Form factor unity summation (provided the surface is closed), and etendue symmetry:
$$
\begin{aligned}
    \sum_{j=1}^{n}
    F_{h \to i \to j}
    & = 1
    \quad \forall h
    \, ,
    \\
    % \pi A_i F_{\text{diff } h \to i \to j}
    % =
    G_{i \to j}
    & =
    G_{j \to i}
    % =
    % \pi A_j F_{\text{diff } h \to j \to i}
    % \quad \forall h
    \, .
\end{aligned}
$$
Both of these equalities are exact in theory, but approximate in practice, due to discretization of the integrals. The incurred error acts as an assessment of the integration accuracy.

### Specular kernel component

#### Evaluation

The specular BRDF is
$$
\begin{aligned}
    \rho(x_h, x_i, x_j)
    & =
    \frac{\delta(\text{spec}(x_h\!-\!x_i) - (x_j\!-\!x_i))}{\cos \theta_{ih}}
    \\ & =
    \frac{\delta(\text{spec}(x_h\!-\!x_i) - (x_j\!-\!x_i))}{\cos \theta_{ij}}
    \\ & =
    \frac{\delta(\text{spec}(x_j\!-\!x_i) - (x_h\!-\!x_i))}{\cos \theta_{ih}}
    \\ & =
    \frac{\delta(\text{spec}(x_j\!-\!x_i) - (x_h\!-\!x_i))}{\cos \theta_{ij}}
    \, .
\end{aligned}
$$
which gives
$$
\begin{aligned}
    F_{h \to i \to j}
    & =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \frac{\delta(\text{spec}(x_j\!-\!x_i) - (x_h\!-\!x_i))}{\cos \theta_{ij}}
    \cos \theta_{ij}
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \\ & =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \delta(\text{spec}(x_j\!-\!x_i) - (x_h\!-\!x_i))
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \\ & =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \delta((x_j\!-\!x_i) - \text{spec}(x_h\!-\!x_i))
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \, .
\end{aligned}
$$
We can remove an integral thanks to the delta's sifting property ${\int_{-\infty}^{\infty} f(t) \delta(t-T) \,\hbox{d}t = f(T)}$. Before we do, let us make the visibility term w.r.t. $\Omega_j$ explicit again:
$$
\begin{aligned}
    F_{h \to i \to j}
    & =
    \iint_{A_i}
    \iint_{\Omega_h}
    \iint_{\Omega_j}
    \left[(x_j\!-\!x_i) \in \Omega_j\right]
    \delta((x_j\!-\!x_i) - \text{spec}(x_h\!-\!x_i))
    \, \hbox{d}\Omega_j
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \\ & =
    \iint_{A_i}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h\!-\!x_i) \in \Omega_j\right]
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    \, .
\end{aligned}
$$
The innermost integrand $\left[\text{spec}(x_h\!-\!x_i) \in \Omega_j\right]$ is equal to 1 if the direction *specular to* $(x_h\!-\!x_i)$ falls within $\Omega_j$ and 0 otherwise. In practice, for numerical integration, taking the average of a "boolean" integrand like this means counting the number of sample points (i.e., rays) for which the condition is true.
$$
\begin{aligned}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h\!-\!x_i) \in \Omega_j\right]
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    & \approx
    \frac{
    \sum\nolimits_{\omega_h \in \Omega_h}
    \left[\text{spec}(\omega_h) \in \Omega_j\right]
    }{
    \sum\nolimits_{\omega_h \in \Omega_h}
    1
    }
    \, ,
    \\
    \iint_{A_i}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h\!-\!x_i) \in \Omega_j\right]
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
    & \approx
    \frac{1}{N_x}
    \sum\nolimits_{x_i \in A_i}
    \frac{
    \sum\nolimits_{\omega_h \in \Omega_h}
    \left[\text{spec}(\omega_h) \in \Omega_j\right]
    }{
    \sum\nolimits_{\omega_h \in \Omega_h}
    1
    }
    \, .
\end{aligned}
$$
With that said, in practice we carry out the averaging over $x_i \in A_i$ slightly differently, so that the results are closer to the kernel's unity summation property:
$$
    \iint_{A_i}
    \iint_{\Omega_h}
    \left[\text{spec}(x_h\!-\!x_i) \in \Omega_j\right]
    \frac{\hbox{d}\Omega_h}{\Omega_h}
    \frac{\hbox{d}A_i}{A_i}
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
    \, .
$$
