[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lshofsl/Advanced-topics-in-PoD-project-/blob/main/EBM_NCA.ipynb)


Neural Cellular Automa with a an energy-gradient approach.

Neural Cellular Automata (NCA) have emerged as a compelling framework for modelling morphogenetic processes, demonstrating the capacity to grow and maintain complex
spatial patterns through purely local cell interaction.
Despite this success, a fundamental limitation of current NCA formulations is the ack of interpretability: the rules governing how a pattern is learned
and stabilised remain largely opaque, and practitioners resort to empirical diagnostics, a black box neural network, rather than principled analysis.

This project addresses that gap by introducing two complementary mechanisms into
the NCA framework:

-A Hebbian convolutional layer that replaces standard radient-descent-only weight updates with 
biologically plausible, locally driven weight changes, connecting the NCA update rule to the associative
memory literature on Hopfield networks.

-An explicit energy function} $E_\theta$ whose gradient constrains the cell-state dynamics 
during both training and inference, imposing a well-defined energy landscape over the space of cell configurations.

The central hypothesis is that under these constraints, each learned pattern will correspond to an attractor in the energy landscape, a local minimum (valley), so
that perturbations to a stable configuration naturally relax back toward the nearest stored pattern. This yields three simultaneous gains: \emph{robustness}
through the attractor dynamics, \emph{biological plausibility} through Hebbian plasticity, and \emph{interpretability} through an energy landscape that can be
visualised and analysed directly.
