// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef QISKIT_ADDON_SQD_CONFIGURATION_RECOVERY_HPP_
#define QISKIT_ADDON_SQD_CONFIGURATION_RECOVERY_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "qiskit/addon/sqd/internal/concepts.hpp"
#include "qiskit/addon/sqd/internal/dense_map.hpp"
#include "qiskit/addon/sqd/internal/exception-macros.hpp"
#include "qiskit/addon/sqd/internal/finite-math.hpp"
#include "qiskit/addon/sqd/internal/parallel-rng.hpp"
#include "qiskit/addon/sqd/internal/sample-without-replacement.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Qiskit
{

namespace addon
{

namespace sqd
{

/// Relative weights for flipping each bit during configuration recovery.
///
/// When `recover_configurations()` must change the number of electrons in one
/// spin species of a bitstring, it chooses which bits to flip by sampling,
/// without replacement, among the bits that hold the value to be flipped (the
/// 1s if there are too many electrons, the 0s if there are too few).  Each
/// eligible bit is chosen in proportion to its weight in this table.
///
/// Weights are relative: only their ratios among the eligible bits of a
/// single spin species matter, so they need not be normalized.  They must be
/// finite and non-negative.  A bit with weight zero is chosen only if a spin
/// species needs more flips than it has eligible bits with positive weight; the
/// remaining flips are then chosen uniformly among the zero-weight bits (see
/// `num_uniform_flips()`).
///
/// A default table, derived from average orbital occupancies, is produced by
/// `flip_probabilities_from_occupancies()`; it may be modified before being
/// passed to `recover_configurations()`.  Alternatively, construct a table
/// directly and fill in every weight.
class FlipProbabilityTable
{
  public:
    /// Construct a table for `num_orbitals` spatial orbitals per spin species,
    /// with every weight initialized to zero.
    explicit FlipProbabilityTable(std::size_t num_orbitals)
      : num_orbitals_(num_orbitals), weights_(4 * num_orbitals, 0.0)
    {
    }

    /// Number of spatial orbitals per spin species.
    [[nodiscard]] std::size_t num_orbitals() const noexcept
    {
        return num_orbitals_;
    }

    /// Access the weight for flipping a bit.
    ///
    /// @param[in] spin Spin species: 0 for spin-up (the low, or right, half of
    ///     each bitstring) and 1 for spin-down (the high, or left, half).
    /// @param[in] bit The bit's current value: `false` for the weight of
    ///     flipping a 0 to a 1, `true` for flipping a 1 to a 0.
    /// @param[in] orbital Orbital index within the spin species, less than
    ///     `num_orbitals()`.
    double &operator()(std::size_t spin, bool bit, std::size_t orbital)
    {
        return weights_[index(spin, bit, orbital)];
    }

    /// @copydoc operator()(std::size_t, bool, std::size_t)
    double operator()(std::size_t spin, bool bit, std::size_t orbital) const
    {
        return weights_[index(spin, bit, orbital)];
    }

  private:
    [[nodiscard]] std::size_t
    index(std::size_t spin, bool bit, std::size_t orbital) const noexcept
    {
        assert(spin < 2);
        assert(orbital < num_orbitals_);
        return ((2 * spin) + static_cast<std::size_t>(bit)) * num_orbitals_ + orbital;
    }

    std::size_t num_orbitals_;
    std::vector<double> weights_;
};

namespace internal
{

template <typename WeightVectorType>
void _normalize(WeightVectorType &probs)
{
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (sum > 0.0) {
        for (double &prob : probs) {
            prob /= sum;
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline double _p_flip_0_to_1(double ratio_exp, double occ, double eps = 0.01)
{
    // Occupancy is less than the naive expectation.
    // Flip 0s to 1 with small (<eps) probability in this case.
    if (occ < ratio_exp) {
        return occ * eps / ratio_exp;
    }

    // Occupancy is greater than or equal to the naive expectation.
    // Increase the probability to flip the bit linearly as the deviation
    // increases.
    if (ratio_exp == 1.0) {
        return eps;
    }

    double slope = (1.0 - eps) / (1.0 - ratio_exp);
    double intercept = 1.0 - slope;
    return occ * slope + intercept;
}

inline double _p_flip_1_to_0(double ratio_exp, double occ, double eps = 0.01)
{
    return _p_flip_0_to_1(1.0 - ratio_exp, 1.0 - occ, eps);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

template <typename BitstringType>
void mask_lower_n_bits_inplace(BitstringType &bitstring, unsigned int n)
{
    assert(n <= bitstring.size());
    auto shift = bitstring.size() - n;
    bitstring <<= shift;
    bitstring >>= shift;
}

template <typename BitstringType>
constexpr BitstringType
mask_lower_n_bits(const BitstringType &bitstring, unsigned int n)
{
    BitstringType retval(bitstring);
    mask_lower_n_bits_inplace(retval, n);
    return retval;
}

template <typename BitstringType, QKA_SQD_CONCEPT_RNG_(RNGType)>
void _bipartite_bitstring_correcting(
    BitstringType &bitstring, const FlipProbabilityTable &flip_probs,
    std::array<std::uint64_t, 2> num_elec,
    std::pair<std::vector<std::size_t>, std::vector<double>> &scratch_vectors,
    RNGType &rng
)
{
    // Use the flip weights (flip_probs) and target Hamming weight to correct a
    // bitstring.

#if QKA_SQD_DEBUG_RECOVERY
    std::cerr << "Initial bitstring: " << bitstring << std::endl;
    std::cerr << "Desired Hamming weight: " << num_elec[0] << ' ' << num_elec[1]
              << std::endl;
#endif

    // The number of bits should be even - this was already checked in the calling
    // function
    const auto partition_size = flip_probs.num_orbitals();

    // Determine starting Hamming weights
    std::array<std::uint64_t, 2> initial_hamming_weight;
    const auto n_right = mask_lower_n_bits(bitstring, partition_size).count();
    initial_hamming_weight[0] = n_right;
    initial_hamming_weight[1] = bitstring.count() - n_right;

    // Handle RIGHT (alpha) then LEFT (beta) bits
    std::uint64_t offset = 0;
    for (int s = 0; s < 2; ++s) {
        if (initial_hamming_weight[s] != num_elec[s]) {
            const bool flip = bool(
                initial_hamming_weight[s] > num_elec[s]
            ); // 1 or 0 depending on which should be flipped
            const std::uint64_t num_flip = std::abs(
                static_cast<int>(initial_hamming_weight[s]) -
                static_cast<int>(num_elec[s])
            );
            // Gather the eligible bits: those with a positive flip weight first
            // (parallel to `weights`), followed by those with zero weight.
            auto &[indices, weights] = scratch_vectors;
            indices.clear();
            weights.clear();
            for (std::uint64_t j = 0; j < partition_size; ++j) {
                if (bitstring[j + offset] == flip && flip_probs(s, flip, j) > 0) {
                    indices.push_back(j + offset);
                    weights.push_back(flip_probs(s, flip, j));
                }
            }
            const auto num_weighted = weights.size();
            for (std::uint64_t j = 0; j < partition_size; ++j) {
                if (bitstring[j + offset] == flip && !(flip_probs(s, flip, j) > 0)) {
                    indices.push_back(j + offset);
                }
            }
            // `recover_configurations` has already checked the target Hamming
            // weight against the partition size, so there are always enough
            // eligible bits.
            assert(indices.size() >= num_flip);

            // Flip bits with a positive weight first, sampling by weight.  The
            // sampler only ever sees positive weights, so no draw can fail.
            std::uint64_t num_flipped = 0;
            if (num_weighted != 0) {
                internal::NoReplacementSampler sampler(weights);
                for (; num_flipped < num_flip && num_flipped < num_weighted;
                     ++num_flipped) {
                    bitstring.flip(indices[sampler(rng)]);
                }
            }

            // If more flips are still needed once every positive-weight bit has
            // been flipped, choose the rest uniformly among the zero-weight
            // bits, via a partial Fisher-Yates shuffle.  `num_uniform_flips()`
            // predicts how many flips take this path.
            for (auto k = num_weighted; num_flipped < num_flip; ++k, ++num_flipped) {
                std::uniform_int_distribution<std::size_t> pick(k, indices.size() - 1);
                std::swap(indices[k], indices[pick(rng)]);
                bitstring.flip(indices[k]);
            }
        }
        offset += partition_size;
    }

#if QKA_SQD_DEBUG_RECOVERY
    std::cerr << "Final bitstring: " << bitstring << '\n' << std::endl;
#endif
    assert(mask_lower_n_bits(bitstring, partition_size).count() == num_elec[0]);
    assert(bitstring.count() == num_elec[0] + num_elec[1]);
}

inline void _validate_flip_probabilities(const FlipProbabilityTable &flip_probs)
{
    for (std::size_t s = 0; s < 2; ++s) {
        for (int bit = 0; bit < 2; ++bit) {
            for (std::size_t i = 0; i < flip_probs.num_orbitals(); ++i) {
                const double weight = flip_probs(s, bit != 0, i);
#if !QKA_SQD_FINITE_MATH_ONLY
                if (!std::isfinite(weight)) {
                    QKA_SQD_THROW_INVALID_ARGUMENT_(
                        "Flip probabilities must be finite."
                    );
                }
#endif // !QKA_SQD_FINITE_MATH_ONLY
                if (weight < 0) {
                    QKA_SQD_THROW_INVALID_ARGUMENT_(
                        "Flip probabilities must be non-negative."
                    );
                }
            }
        }
    }
}

} // namespace internal

/// Compute the default flip-probability table from average orbital
/// occupancies.
///
/// This is the table that `recover_configurations()` uses when given average
/// occupancies.  It can be modified and then passed to the overload of
/// `recover_configurations()` that accepts a `FlipProbabilityTable`.
///
/// Each orbital's occupancy is clamped to [0, 1] and compared with the naive
/// expectation `num_elec[s] / num_orbitals`.  A 0 in an orbital whose
/// occupancy exceeds the expectation is likely to be flipped to a 1 (and
/// likewise for a 1 in an under-occupied orbital), with a weight that grows
/// linearly with the deviation; otherwise the weight is small (below 0.01).
///
/// @param[in] avg_occupancies Size-2 `std::array` of `std::vector<double>`s holding the
///     mean occupancy of the spin-up and spin-down orbitals, respectively.  Both
///     vectors must have the same size.
/// @param[in] num_elec Size-2 `std::array` containing the number of spin-up and
///     spin-down electrons in the system, respectively.
///
/// @return The flip-probability table.
[[nodiscard]] inline FlipProbabilityTable flip_probabilities_from_occupancies(
    const std::array<std::vector<double>, 2> &avg_occupancies,
    std::array<std::uint64_t, 2> num_elec
)
{
    const auto partition_size = avg_occupancies[0].size();
    if (avg_occupancies[1].size() != partition_size) {
        QKA_SQD_THROW_INVALID_ARGUMENT_(
            "Average occupancies vectors must have matching number of alpha and beta "
            "orbitals."
        );
    }
    if (num_elec[0] > partition_size || num_elec[1] > partition_size) {
        QKA_SQD_THROW_INVALID_ARGUMENT_(
            "Desired Hamming weight cannot be larger than the number of orbitals."
        );
    }

    FlipProbabilityTable flip_probs(partition_size);
    for (std::size_t s = 0; s < 2; ++s) {
        // NOLINTBEGIN(bugprone-narrowing-conversions)
        double density_s = static_cast<double>(num_elec[s]) / partition_size;
        // NOLINTEND(bugprone-narrowing-conversions)
        for (std::size_t i = 0; i < partition_size; ++i) {
            const auto occ = std::max(0.0, std::min(1.0, avg_occupancies[s][i]));
            flip_probs(s, false, i) = internal::_p_flip_0_to_1(density_s, occ);
            flip_probs(s, true, i) = internal::_p_flip_1_to_0(density_s, occ);
        }
    }
    return flip_probs;
}

/// Count the bit flips that configuration recovery would have to choose
/// uniformly, rather than by weight, when correcting a bitstring.
///
/// `recover_configurations()` corrects each spin species by flipping bits that
/// hold the value to be flipped, chosen by weight from `flip_probs`.  A bit
/// whose weight is zero can never be chosen that way.  If a species needs more
/// flips than it has eligible bits with a positive weight, every such bit is
/// flipped and the remaining flips are chosen uniformly at random among the
/// zero-weight eligible bits.  Those remaining flips are not informed by
/// `flip_probs`.
///
/// This function returns the number of such flips, summed over both spin
/// species, without sampling anything.  It is zero exactly when the correction
/// of `bitstring` is guided entirely by `flip_probs`.  Calling it on each input
/// bitstring before `recover_configurations()` tells a caller whether, and how
/// much, uniform filling will occur:
///
/// @code
/// auto table = flip_probabilities_from_occupancies(avg_occupancies, num_elec);
/// for (const auto &bitstring : bitstrings) {
///     if (num_uniform_flips(bitstring, table, num_elec) != 0) {
///         // e.g., warn, or floor the zero weights in `table`
///     }
/// }
/// @endcode
///
/// Zero weights arise from ordinary data, not only from hand-built tables:
/// `flip_probabilities_from_occupancies()` gives weight zero to flipping a 0 in
/// an orbital whose average occupancy is exactly 0, and to flipping a 1 in an
/// orbital whose average occupancy is exactly 1.
///
/// @param[in] bitstring The bitstring to be corrected.  Its size must be twice
///     `flip_probs.num_orbitals()`.
/// @param[in] flip_probs Relative weights for flipping each bit.
/// @param[in] num_elec Size-2 `std::array` containing the number of spin-up and
///     spin-down electrons in the system, respectively.
///
/// @tparam BitstringType Type of `bitstring`, compatible with
///     `boost::dynamic_bitset<>`.
///
/// @return The number of flips that would be chosen uniformly.
template <typename BitstringType>
[[nodiscard]] std::uint64_t num_uniform_flips(
    const BitstringType &bitstring, const FlipProbabilityTable &flip_probs,
    std::array<std::uint64_t, 2> num_elec
)
{
    const auto partition_size = flip_probs.num_orbitals();
    if (num_elec[0] > partition_size || num_elec[1] > partition_size) {
        QKA_SQD_THROW_INVALID_ARGUMENT_(
            "Desired Hamming weight cannot be larger than the number of orbitals."
        );
    }
    if (bitstring.size() != 2 * partition_size) {
        QKA_SQD_THROW_INVALID_ARGUMENT_(
            "Bitstring length must be twice the number of orbitals."
        );
    }

    // This mirrors `_bipartite_bitstring_correcting` exactly: the quantities
    // involved are fixed by (bitstring, flip_probs, num_elec), so the count is a
    // prediction rather than an estimate.
    const auto n_right_bits =
        internal::mask_lower_n_bits(bitstring, partition_size).count();
    const std::array<std::uint64_t, 2> hamming_weight{
        static_cast<std::uint64_t>(n_right_bits),
        static_cast<std::uint64_t>(bitstring.count() - n_right_bits)
    };
    std::uint64_t total = 0;
    std::uint64_t offset = 0;
    for (std::size_t s = 0; s < 2; ++s) {
        if (hamming_weight[s] != num_elec[s]) {
            const bool flip = bool(hamming_weight[s] > num_elec[s]);
            const std::uint64_t num_flip = flip ? hamming_weight[s] - num_elec[s]
                                                : num_elec[s] - hamming_weight[s];
            std::uint64_t num_weighted = 0;
            for (std::size_t j = 0; j < partition_size; ++j) {
                if (bitstring[j + offset] == flip && flip_probs(s, flip, j) > 0) {
                    ++num_weighted;
                }
            }
            if (num_flip > num_weighted) {
                total += num_flip - num_weighted;
            }
        }
        offset += partition_size;
    }
    return total;
}

/// Refine bitstrings based on a table of flip probabilities and a target
/// Hamming weight.
///
/// Each bitstring whose spin-up or spin-down Hamming weight differs from the
/// target is corrected by flipping bits, chosen according to `flip_probs` (see
/// `FlipProbabilityTable`).  Corrected bitstrings that coincide are merged, and
/// their probabilities summed.
///
/// Bits with weight zero are never chosen while a bit with positive weight
/// remains eligible.  If a spin species needs more flips than it has eligible
/// bits with positive weight, the remaining flips are chosen uniformly at random
/// among its zero-weight eligible bits, so every output has the target Hamming
/// weight.  Use `num_uniform_flips()` to find out in advance whether this will
/// happen.
///
///   - A counter-based engine (one whose `set_counter` accepts its counter
///     type, such as the C++26 `std::philox_engine`) is keyed by bitstring
///     index, so the result -- including the order of the returned vectors --
///     is identical for any number of threads.
///   - Any other engine must be seedable (have `seed()`); each thread then uses
///     an independently seeded copy.  Results are valid but depend on the
///     thread count.  A generator that is neither counter-based nor seedable is
///     rejected at compile time under OpenMP (it remains usable in a serial
///     build).
///
/// `rng` is left in a substantially different state than a purely sequential
/// implementation would leave it, in *both* builds: this function draws from it
/// to derive a base seed and then re-seeds it (and, for a counter-based engine,
/// re-keys it per work item), so it is not merely advanced by the number of
/// values consumed.  Callers who continue to draw from the same generator
/// afterwards should not assume any particular number of values was consumed.
///
/// @param[in] bitstrings A container (e.g., `std::vector`) of bitstrings.
/// @param[in] probabilities A 1D array specifying a probability distribution over
///     the bitstrings.  Must contain the same number of elements as `bitstrings`.
/// @param[in] flip_probs Relative weights for flipping each bit.  Each bitstring's
///     size must be twice `flip_probs.num_orbitals()`.
/// @param[in] num_elec Size-2 `std::array` containing the number of spin-up and
///     spin-down electrons in the system, respectively.
/// @param[in,out] rng Random number generator.
///
/// @tparam BitstringVectorType Type of `bitstrings`, compatible with
///     `std::vector<boost::dynamic_bitset<>>`.
/// @tparam WeightVectorType Type of `weights`, compatible with `std::vector<double>`.
/// @tparam RNGType Type of random number generator.
///
/// @return A refined `std::vector` of unique bitstrings and a parallel, updated
///     probability array.
template <
    typename BitstringVectorType, typename WeightVectorType,
    QKA_SQD_CONCEPT_RNG_(RNGType)
>
[[nodiscard]] std::pair<BitstringVectorType, WeightVectorType> recover_configurations(
    const BitstringVectorType &bitstrings, const WeightVectorType &probabilities,
    const FlipProbabilityTable &flip_probs, std::array<std::uint64_t, 2> num_elec,
    RNGType &rng
)
{
    if (bitstrings.size() != probabilities.size()) {
        QKA_SQD_THROW_INVALID_ARGUMENT_(
            "Probabilities vector must have length that matches the bitstrings vector."
        );
    }

    const auto partition_size = flip_probs.num_orbitals();
    if (num_elec[0] > partition_size || num_elec[1] > partition_size) {
        QKA_SQD_THROW_INVALID_ARGUMENT_(
            "Desired Hamming weight cannot be larger than the number of orbitals."
        );
    }
    internal::_validate_flip_probabilities(flip_probs);

    using BitstringType = typename BitstringVectorType::value_type;

    // Validate every input up front, so the correction loop below (which may run
    // in parallel) needs no exception-throwing control flow.  This is a
    // correctness requirement and not merely tidiness: throwing out of an OpenMP
    // structured block is undefined, and libgomp terminates the process -- so a
    // condition detected inside the loop would abort rather than raise.
    //
    // Once the target Hamming weight is known to fit in a partition, correction
    // cannot fail: flips that cannot be chosen by weight are chosen uniformly
    // (see `num_uniform_flips()`).  So the only per-bitstring check is length.
    for (const auto &bitstring : bitstrings) {
        if (bitstring.size() != 2 * partition_size) {
            QKA_SQD_THROW_INVALID_ARGUMENT_(
                "Bitstring length must be twice the number of orbitals."
            );
        }
    }

    // A counter-based engine keys each work item by its index, and that index must
    // fit in one of the engine's counter words -- `set_counter` reduces each word
    // mod 2^word_size, so beyond that two items would name the same substream and
    // draw identical randomness.  Reject such a workload here rather than inside
    // the loop: an assert would vanish under NDEBUG, and the loop may be parallel,
    // where throwing is not available.  `max_substreams` returns 0 when no index
    // can overflow the engine's word, which is the usual case.
    if constexpr (internal::is_counter_based_rng_v<RNGType>) {
        constexpr std::size_t substream_capacity = internal::max_substreams<RNGType>();
        if (substream_capacity != 0 && bitstrings.size() > substream_capacity) {
            QKA_SQD_THROW_INVALID_ARGUMENT_(
                "Too many bitstrings to key distinct random substreams for this "
                "generator; its counter word is too narrow."
            );
        }
    }

    // Correct every bitstring into a position-indexed array.  Duplicate removal
    // is done afterwards, on a single thread, in input order -- so the result is
    // independent of how the correction loop was scheduled.
    std::vector<BitstringType> corrected(bitstrings.size());

#ifdef _OPENMP
    // Parallel correction.  Each iteration touches only its own slot and its own
    // thread-local scratch and generator, so there is no contention.  With a
    // counter-based RNG the per-item stream is keyed by index, making the result
    // bit-for-bit independent of the thread count; with an ordinary RNG each
    // thread gets an independently-seeded copy (valid, but thread-count
    // dependent).  See internal/parallel-rng.hpp.
    static_assert(
        internal::is_counter_based_rng_v<RNGType> ||
            internal::is_seedable_rng_v<RNGType>,
        "Under OpenMP, recover_configurations requires an RNG that is either "
        "counter-based (set_counter accepts its counter type, e.g. "
        "std::philox_engine) or seedable "
        "(has seed()); a concept-only uniform_random_bit_generator is supported "
        "only in the serial (non-OpenMP) build."
    );
    // Derive a 64-bit base seed from the caller's generator.  Two draws XORed
    // (the second shifted into the high half) so the seed has full 64-bit
    // entropy even when the engine's result_type is only 32 bits wide.
    const std::uint64_t base_seed =
        static_cast<std::uint64_t>(rng()) ^ (static_cast<std::uint64_t>(rng()) << 32);
#pragma omp parallel
    {
        std::pair<std::vector<std::size_t>, std::vector<double>> scratch;
        // Per-thread generator.  The counter-based branch re-seeds and re-keys
        // this below (so its initial copied state is discarded); the ordinary
        // branch keeps the copy and just re-seeds it once per thread here.
        RNGType thread_rng = rng;
        if constexpr (!internal::is_counter_based_rng_v<RNGType>) {
            // Give each thread a well-separated seed.  See `thread_seed`, which
            // owns the expression so a test can assert on the same one.
            thread_rng.seed(
                static_cast<typename RNGType::result_type>(internal::thread_seed(
                    base_seed, static_cast<std::uint64_t>(omp_get_thread_num())
                ))
            );
        }
#pragma omp for
        for (std::size_t i = 0; i < bitstrings.size(); ++i) {
            if constexpr (internal::is_counter_based_rng_v<RNGType>) {
                internal::key_counter_based_rng(thread_rng, base_seed, i);
            }
            BitstringType corrected_bitstring = bitstrings[i];
            internal::_bipartite_bitstring_correcting(
                corrected_bitstring, flip_probs, num_elec, scratch, thread_rng
            );
            corrected[i] = std::move(corrected_bitstring);
        }
    }
#else
    std::pair<std::vector<std::size_t>, std::vector<double>> scratch;
    // This mirrors the OpenMP path above step for step, so that a serial build
    // agrees with an OpenMP one: the base seed is drawn the same way, and the
    // generator is then keyed or re-seeded exactly as a single thread would be.
    // For a counter-based engine that gives agreement at any thread count, since
    // a work item's stream depends only on its index.  For an ordinary engine it
    // gives agreement at one thread -- the case worth having, because it is the
    // reference point for debugging a parallel run -- while two or more threads
    // still differ by construction, each drawing from its own seeded copy.
    //
    // No test covers this: the two paths are selected by `_OPENMP` at compile
    // time, so no single binary can compare them.  Two builds of the *same*
    // standard library are diffed in CI instead; see the `openmp-tests` job.
    // (The sampler now consumes exactly one variate per draw, rather than
    // std::discrete_distribution's implementation-defined number, so a golden
    // value is no longer out of reach -- but it would still be pinned to
    // uniform_real_distribution's unspecified algorithm, so the diff stands.)
    const std::uint64_t base_seed =
        static_cast<std::uint64_t>(rng()) ^ (static_cast<std::uint64_t>(rng()) << 32);
    if constexpr (
        !internal::is_counter_based_rng_v<RNGType> &&
        internal::is_seedable_rng_v<RNGType>
    ) {
        // The thread id an OpenMP build would see here is 0, so this is the
        // `omp_get_thread_num() + 1` seed of that path with the id fixed at 0.
        // Seedability is required because a generator that has no `seed()` is
        // accepted only in a serial build, so there is no OpenMP output for it
        // to agree with in the first place.
        rng.seed(
            static_cast<typename RNGType::result_type>(
                base_seed + 0x9e3779b97f4a7c15ULL
            )
        );
    }
    for (std::size_t i = 0; i < bitstrings.size(); ++i) {
        if constexpr (internal::is_counter_based_rng_v<RNGType>) {
            // Keying is effectively free even though nothing here is parallel:
            // `set_counter` invalidates the engine's output buffer, and a loop
            // that draws continuously regenerates a block on its next draw
            // regardless, so the cost is moved rather than added (measured at
            // 0.05% on 100,000 bitstrings).
            internal::key_counter_based_rng(rng, base_seed, i);
        }
        BitstringType corrected_bitstring = bitstrings[i];
        internal::_bipartite_bitstring_correcting(
            corrected_bitstring, flip_probs, num_elec, scratch, rng
        );
        corrected[i] = std::move(corrected_bitstring);
    }
#endif

    // Remove duplicates.  Accumulate frequencies keyed by corrected bitstring,
    // then emit each distinct bitstring once in first-seen (input) order.  A
    // flat hash map (see internal/dense_map.hpp) is considerably faster than
    // std::unordered_map here, especially when duplicates are rare -- the common
    // case, since correction seldom maps distinct inputs to the same output.
    // Reserve up front: the number of distinct bitstrings is at most the number
    // of inputs, and close to it when collisions are rare.
    internal::dense_map<BitstringType, double> corrected_dict;
    corrected_dict.reserve(corrected.size());
    for (std::size_t i = 0; i < corrected.size(); ++i) {
        corrected_dict[corrected[i]] += probabilities[i];
    }

    BitstringVectorType bitstrings_out;
    WeightVectorType freqs_out;
    bitstrings_out.reserve(corrected_dict.size());
    freqs_out.reserve(corrected_dict.size());
    for (std::size_t i = 0; i < corrected.size(); ++i) {
        auto it = corrected_dict.find(corrected[i]);
        if (it != corrected_dict.end()) {
            bitstrings_out.emplace_back(it->first);
            freqs_out.push_back(it->second);
            corrected_dict.erase(it); // ensures each distinct bitstring emitted once
        }
    }

    // Normalize the frequencies
    internal::_normalize(freqs_out);

    return {bitstrings_out, freqs_out};
}

/// Refine bitstrings based on average orbital occupancy and a target
/// Hamming weight.
///
/// Equivalent to calling the overload that accepts a `FlipProbabilityTable`
/// with `flip_probabilities_from_occupancies(avg_occupancies, num_elec)`.
///
/// When compiled with OpenMP enabled, the per-bitstring correction runs in
/// parallel.  In that case `rng` is used to seed independent per-work-item
/// random streams rather than being drawn from sequentially, so the numerical
/// output differs from a purely sequential run.  The degree of reproducibility
/// depends on the generator:
///
///   - A counter-based engine (one with `set_counter`, such as the C++26
///     `std::philox_engine`) is keyed by bitstring index, so the result --
///     including the order of the returned vectors -- is identical for any
///     number of threads.
///   - Any other engine must be seedable (have `seed()`); each thread then uses
///     an independently seeded copy.  Results are valid but depend on the
///     thread count.  A generator that is neither counter-based nor seedable is
///     rejected at compile time under OpenMP (it remains usable in a serial
///     build).
///
/// @param[in] bitstrings A container (e.g., `std::vector`) of bitstrings.
/// @param[in] probabilities A 1D array specifying a probability distribution over
///     the bitstrings.  Must contain the same number of elements as `bitstrings`.
/// @param[in] avg_occupancies Size-2 `std::array` of `std::vector<double>`s holding the
///     mean occupancy of the spin-up and spin-down orbitals, respectively.  Each
///     vector's size must be half the size of a single bitstring.
/// @param[in] num_elec Size-2 `std::array` containing the number of spin-up and
///     spin-down electrons in the system, respectively.
/// @param[in,out] rng Random number generator.
///
/// @tparam BitstringVectorType Type of `bitstrings`, compatible with
///     `std::vector<boost::dynamic_bitset<>>`.
/// @tparam WeightVectorType Type of `weights`, compatible with `std::vector<double>`.
/// @tparam RNGType Type of random number generator.
///
/// @return A refined `std::vector` of unique bitstrings and a parallel, updated
///     probability array.
template <
    typename BitstringVectorType, typename WeightVectorType,
    QKA_SQD_CONCEPT_RNG_(RNGType)
>
[[nodiscard]] std::pair<BitstringVectorType, WeightVectorType> recover_configurations(
    const BitstringVectorType &bitstrings, const WeightVectorType &probabilities,
    const std::array<std::vector<double>, 2> &avg_occupancies,
    std::array<std::uint64_t, 2> num_elec, RNGType &rng
)
{
    return recover_configurations(
        bitstrings, probabilities,
        flip_probabilities_from_occupancies(avg_occupancies, num_elec), num_elec, rng
    );
}

} // namespace sqd

} // namespace addon

} // namespace Qiskit

#endif // QISKIT_ADDON_SQD_CONFIGURATION_RECOVERY_HPP_
