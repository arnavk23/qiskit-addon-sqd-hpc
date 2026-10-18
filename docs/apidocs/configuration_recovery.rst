======================
Configuration recovery
======================

Functions
=========

This library provides a function for performing configuration recovery.  By default, the bits to flip are chosen according to the average orbital occupancies:

.. doxygenfunction:: Qiskit::addon::sqd::recover_configurations(const BitstringVectorType &, const WeightVectorType &, const std::array<std::vector<double>, 2> &, std::array<std::uint64_t, 2>, RNGType &)

Alternatively, the weights for flipping each bit can be specified directly with a :cpp:class:`Qiskit::addon::sqd::FlipProbabilityTable`:

.. doxygenfunction:: Qiskit::addon::sqd::recover_configurations(const BitstringVectorType &, const WeightVectorType &, const FlipProbabilityTable &, std::array<std::uint64_t, 2>, RNGType &)

The default table can be obtained, for inspection or modification, with:

.. doxygenfunction:: Qiskit::addon::sqd::flip_probabilities_from_occupancies

A bit whose weight is zero is chosen only when too few bits with positive weight are eligible, in which case the remaining flips are chosen uniformly.  Whether, and how often, that will happen can be checked before calling ``recover_configurations``:

.. doxygenfunction:: Qiskit::addon::sqd::num_uniform_flips

Classes
=======

.. doxygenclass:: Qiskit::addon::sqd::FlipProbabilityTable
   :members:
