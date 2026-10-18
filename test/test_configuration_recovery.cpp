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

#include "doctest.h"
#include "qiskit/addon/sqd/configuration_recovery.hpp"

#include <array>
#include <bitset>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "bitset_compat.hpp"

using Qiskit::addon::sqd::flip_probabilities_from_occupancies;
using Qiskit::addon::sqd::FlipProbabilityTable;
using Qiskit::addon::sqd::num_uniform_flips;
using Qiskit::addon::sqd::recover_configurations;

#if !QKA_SQD_DISABLE_EXCEPTIONS && !(_MSVC_LANG == 202002L)
#define BITSET2_IF_AVAILABLE , Bitset2::bitset2<4>
#else
#define BITSET2_IF_AVAILABLE
#endif

TEST_CASE_TEMPLATE(
    "Configuration recovery", BitstringType, std::bitset<4>,
    boost::dynamic_bitset<> BITSET2_IF_AVAILABLE
)
{
    constexpr auto N = 4;
    std::mt19937_64 rng;
    std::vector<BitstringType> bitstrings;
    for (unsigned int i = 0; i < 5; ++i) {
        BitstringType bs;
        set_bitset(N, bs, i);
        bitstrings.push_back(bs);
    }
    std::vector<double> probabilities{1, 2, 3, 4, 5};
    std::vector<double> avg_occupancies{0.1, 0.2};
    unsigned int num_elec_a = 1, num_elec_b = 1;

    std::ignore = recover_configurations(
        bitstrings, probabilities, {avg_occupancies, avg_occupancies},
        {num_elec_a, num_elec_b}, rng
    );
    CHECK(true); // FIXME
}

TEST_CASE("Configuration recovery tests from python addon")
{
    // https://github.com/Qiskit/qiskit-addon-sqd/blob/main/test/test_configuration_recovery.py
    std::mt19937_64 rng;
    SUBCASE("Empty test")
    {
        constexpr auto num_orbs = 6;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 0;
        constexpr auto ham_l = 1;
        const std::vector<std::bitset<num_orbs>> empty_bitstring_vec;
        const std::vector<double> empty_probs;
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs), std::vector<double>(half_orbs)
        };
        auto [mat_rec, probs_rec] = recover_configurations(
            empty_bitstring_vec, empty_probs, occs, {ham_r, ham_l}, rng
        );
        CHECK(mat_rec.size() == 0);
        CHECK(probs_rec.size() == 0);
    }
    SUBCASE("Basic test. Zeros to ones.")
    {
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 2;
        constexpr auto ham_l = 2;
        const std::vector<std::bitset<num_orbs>> bitstrings(1);
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, 1.000001),
            std::vector<double>(half_orbs, 1.0)
        };
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {ham_r, ham_l}, rng);
        CHECK(mat_rec.size() == 1);
        CHECK(mat_rec[0] == 0b1111);
        CHECK(probs_rec.size() == 1);
        CHECK(probs_rec[0] == 1.0);
    }
    SUBCASE("Basic test. Ones to zeros.")
    {
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 0;
        constexpr auto ham_l = 0;
        const std::vector<std::bitset<num_orbs>> bitstrings(1, 0b1111);
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, -1e6), std::vector<double>(half_orbs)
        };
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {ham_r, ham_l}, rng);
        CHECK(mat_rec.size() == 1);
        CHECK(mat_rec[0] == 0);
        CHECK(probs_rec.size() == 1);
        CHECK(probs_rec[0] == 1.0);
    }
    SUBCASE("Basic test. Mismatching orbitals.")
    {
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 0;
        constexpr auto ham_l = 1;
        const std::vector<std::bitset<num_orbs>> bitstrings(1, 0b1111);
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs), std::vector<double>(half_orbs)
        };
        occs[1][0] = 1.0;
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {ham_r, ham_l}, rng);
        CHECK(mat_rec.size() == 1);
        CHECK(mat_rec[0] == 0b0100);
        CHECK(probs_rec.size() == 1);
        CHECK(probs_rec[0] == 1.0);
    }
    SUBCASE("All input probabilities are zero.")
    {
        // See https://github.com/Qiskit/qiskit-addon-sqd-hpc/issues/33.  When
        // every input probability is zero, the recovered probabilities sum to
        // zero, and internal::_normalize must skip the division rather than
        // produce NaNs.  (The flip weights here are nonzero, so this exercises
        // only output normalization.  The division by zero in
        // https://github.com/Qiskit/qiskit-addon-sqd/issues/274 cannot arise
        // here, since this port never normalizes flip weights; it passes the
        // raw weights to NoReplacementSampler.)
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 2;
        constexpr auto ham_l = 2;
        const std::vector<std::bitset<num_orbs>> bitstrings{0b0000, 0b1111};
        const std::vector<double> probs(2, 0.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, 1.0), std::vector<double>(half_orbs, 1.0)
        };
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {ham_r, ham_l}, rng);
        REQUIRE(mat_rec.size() == probs_rec.size());
        // Both inputs correct to 0b1111 and are merged into a single entry.
        REQUIRE(mat_rec.size() == 1);
        CHECK(mat_rec[0] == 0b1111);
        for (const auto &prob : probs_rec) {
            CHECK_FALSE(std::isnan(prob));
            CHECK(prob == 0.0);
        }
    }
    SUBCASE("Bad Hamming right")
    {
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 3;
        constexpr auto ham_l = 1;
        const std::vector<std::bitset<num_orbs>> bitstrings(1);
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, 1.0), std::vector<double>(half_orbs, 1.0)
        };
        CHECK_THROWS_AS(
            std::ignore =
                recover_configurations(bitstrings, probs, occs, {ham_r, ham_l}, rng),
            std::invalid_argument
        );
    }
    SUBCASE("No orbital is eligible to flip by weight")
    {
        // See https://github.com/Qiskit/qiskit-addon-sqd-hpc/issues/64 and
        // https://github.com/Qiskit/qiskit-addon-sqd-hpc/issues/71.  Fully
        // saturated occupancies drive every 1-to-0 flip weight to zero
        // (density_s == 0, so _p_flip_1_to_0(0.0, 1.0) == 0.0), while the target
        // Hamming weight of zero asks for two flips per sector.  The flips are
        // chosen uniformly instead, and here there is only one way to do it.
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        const std::vector<std::bitset<num_orbs>> bitstrings{0b1111};
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, 1.0), std::vector<double>(half_orbs, 1.0)
        };
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {0, 0}, rng);
        REQUIRE(mat_rec.size() == 1);
        CHECK(mat_rec[0] == 0b0000);
        CHECK(probs_rec[0] == 1.0);
    }
    SUBCASE("Too few orbitals are eligible to flip by weight")
    {
        // Two 0->1 flips are needed in the right partition, but only one bit
        // has a positive weight; the other must still be flipped.
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        const std::vector<std::bitset<num_orbs>> bitstrings(1);
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>{0.0, 0.5}, std::vector<double>(half_orbs)
        };
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {2, 0}, rng);
        REQUIRE(mat_rec.size() == 1);
        CHECK(mat_rec[0] == 0b0011);
    }
    SUBCASE("Zero-weight orbitals are chosen uniformly")
    {
        // From 0b1111 with target {1, 1} and saturated occupancies, each of
        // the four outcomes (one bit kept per partition) is equally likely.
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr int num_samples = 4000;
        const std::vector<std::bitset<num_orbs>> bitstrings(num_samples, 0b1111);
        const std::vector<double> probs(num_samples, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, 1.0), std::vector<double>(half_orbs, 1.0)
        };
        auto [mat_rec, probs_rec] =
            recover_configurations(bitstrings, probs, occs, {1, 1}, rng);
        REQUIRE(mat_rec.size() == 4);
        for (std::size_t i = 0; i < mat_rec.size(); ++i) {
            const auto bs = mat_rec[i].to_ulong();
            CHECK((bs == 0b0101 || bs == 0b0110 || bs == 0b1001 || bs == 0b1010));
            // Expected 0.25 each; 0.05 is over 7 standard deviations.
            CHECK(probs_rec[i] == doctest::Approx(0.25).epsilon(0.2));
        }
    }
    SUBCASE("Bad Hamming left")
    {
        constexpr auto num_orbs = 4;
        constexpr auto half_orbs = num_orbs / 2;
        constexpr auto ham_r = 2;
        constexpr auto ham_l = 3;
        const std::vector<std::bitset<num_orbs>> bitstrings(1);
        const std::vector<double> probs(1, 1.0);
        std::array<std::vector<double>, 2> occs{
            std::vector<double>(half_orbs, 1.0), std::vector<double>(half_orbs, 1.0)
        };
        CHECK_THROWS_AS(
            std::ignore =
                recover_configurations(bitstrings, probs, occs, {ham_r, ham_l}, rng),
            std::invalid_argument
        );
    }
}

TEST_CASE("Flip probability table")
{
    SUBCASE("Zero-initialized")
    {
        const FlipProbabilityTable table(3);
        CHECK(table.num_orbitals() == 3);
        for (std::size_t s = 0; s < 2; ++s) {
            for (std::size_t i = 0; i < 3; ++i) {
                CHECK(table(s, false, i) == 0.0);
                CHECK(table(s, true, i) == 0.0);
            }
        }
    }
    SUBCASE("Elements are independent")
    {
        FlipProbabilityTable table(2);
        double value = 1.0;
        for (std::size_t s = 0; s < 2; ++s) {
            for (const bool bit : {false, true}) {
                for (std::size_t i = 0; i < 2; ++i) {
                    table(s, bit, i) = value++;
                }
            }
        }
        value = 1.0;
        for (std::size_t s = 0; s < 2; ++s) {
            for (const bool bit : {false, true}) {
                for (std::size_t i = 0; i < 2; ++i) {
                    CHECK(table(s, bit, i) == value++);
                }
            }
        }
    }
    SUBCASE("Default table from occupancies")
    {
        // density 0.5; an orbital at occupancy 1 strongly favors 0->1 and
        // disfavors 1->0, and vice versa at occupancy 0.
        const std::array<std::vector<double>, 2> occs{
            std::vector<double>{1.0, 0.0}, std::vector<double>{0.5, 0.5}
        };
        const auto table = flip_probabilities_from_occupancies(occs, {1, 1});
        REQUIRE(table.num_orbitals() == 2);
        CHECK(table(0, false, 0) == doctest::Approx(1.0));
        CHECK(table(0, true, 0) == doctest::Approx(0.0));
        CHECK(table(0, false, 1) == doctest::Approx(0.0));
        CHECK(table(0, true, 1) == doctest::Approx(1.0));
        CHECK(table(1, false, 0) == doctest::Approx(0.01));
        CHECK(table(1, true, 0) == doctest::Approx(0.01));
    }
    SUBCASE("Default table validates its inputs")
    {
        const std::array<std::vector<double>, 2> mismatched{
            std::vector<double>(2), std::vector<double>(3)
        };
        CHECK_THROWS_AS(
            std::ignore = flip_probabilities_from_occupancies(mismatched, {1, 1}),
            std::invalid_argument
        );
        const std::array<std::vector<double>, 2> occs{
            std::vector<double>(2), std::vector<double>(2)
        };
        CHECK_THROWS_AS(
            std::ignore = flip_probabilities_from_occupancies(occs, {3, 1}),
            std::invalid_argument
        );
    }
}

TEST_CASE("Configuration recovery with a flip probability table")
{
    constexpr auto num_orbs = 4;
    constexpr auto half_orbs = num_orbs / 2;
    SUBCASE("Matches the occupancy overload")
    {
        std::vector<std::bitset<num_orbs>> bitstrings;
        bitstrings.reserve(16);
        for (unsigned long i = 0; i < 16; ++i) {
            bitstrings.emplace_back(i);
        }
        const std::vector<double> probs(bitstrings.size(), 1.0);
        const std::array<std::vector<double>, 2> occs{
            std::vector<double>{0.3, 0.8}, std::vector<double>{0.6, 0.1}
        };
        std::mt19937_64 rng_a(42), rng_b(42);
        const auto expected =
            recover_configurations(bitstrings, probs, occs, {1, 1}, rng_a);
        const auto actual = recover_configurations(
            bitstrings, probs, flip_probabilities_from_occupancies(occs, {1, 1}),
            {1, 1}, rng_b
        );
        CHECK(actual.first == expected.first);
        CHECK(actual.second == expected.second);
    }
    SUBCASE("Custom table steers which bits flip")
    {
        // Only orbital 1 of each spin species may be flipped from 0 to 1, and
        // only orbital 0 from 1 to 0.
        FlipProbabilityTable table(half_orbs);
        for (std::size_t s = 0; s < 2; ++s) {
            table(s, false, 1) = 1.0;
            table(s, true, 0) = 1.0;
        }
        // Spin-up has too few electrons (0b00), spin-down too many (0b11).
        const std::vector<std::bitset<num_orbs>> bitstrings{0b1100};
        const std::vector<double> probs{1.0};
        std::mt19937_64 rng;
        for (int trial = 0; trial < 20; ++trial) {
            auto [mat_rec, probs_rec] =
                recover_configurations(bitstrings, probs, table, {1, 1}, rng);
            REQUIRE(mat_rec.size() == 1);
            CHECK(mat_rec[0] == 0b1010);
        }
    }
    SUBCASE("Positive weights are flipped before zero weights")
    {
        // Spin-up needs both 0s flipped but only orbital 1 has a positive
        // weight; spin-down needs one 1 flipped and has no positive weight.
        FlipProbabilityTable table(half_orbs);
        table(0, false, 1) = 1.0;
        const std::vector<std::bitset<num_orbs>> bitstrings{0b1100};
        const std::vector<double> probs{1.0};
        CHECK(num_uniform_flips(bitstrings[0], table, {2, 1}) == 2);
        std::mt19937_64 rng;
        for (int trial = 0; trial < 20; ++trial) {
            auto [mat_rec, probs_rec] =
                recover_configurations(bitstrings, probs, table, {2, 1}, rng);
            REQUIRE(mat_rec.size() == 1);
            CHECK((mat_rec[0] & std::bitset<num_orbs>(0b0011)) == 0b0011);
            CHECK((mat_rec[0] & std::bitset<num_orbs>(0b1100)).count() == 1);
        }
    }
    SUBCASE("Invalid tables are rejected")
    {
        const std::vector<std::bitset<num_orbs>> bitstrings{0b0000};
        const std::vector<double> probs{1.0};
        std::mt19937_64 rng;
        FlipProbabilityTable table(half_orbs);
        table(0, false, 0) = 1.0;
        table(0, false, 1) = 1.0;
        SUBCASE("Negative weight")
        {
            table(1, true, 1) = -0.5;
        }
#if !QKA_SQD_FINITE_MATH_ONLY
        SUBCASE("NaN weight")
        {
            table(1, true, 1) = std::nan("");
        }
        SUBCASE("Infinite weight")
        {
            table(1, false, 0) = INFINITY;
        }
#endif
        CHECK_THROWS_AS(
            std::ignore = recover_configurations(bitstrings, probs, table, {1, 0}, rng),
            std::invalid_argument
        );
    }
    SUBCASE("Table size must match the bitstrings")
    {
        const std::vector<std::bitset<num_orbs>> bitstrings{0b0000};
        const std::vector<double> probs{1.0};
        std::mt19937_64 rng;
        const FlipProbabilityTable table(3);
        CHECK_THROWS_AS(
            std::ignore = recover_configurations(bitstrings, probs, table, {1, 1}, rng),
            std::invalid_argument
        );
    }
    SUBCASE("Hamming weight cannot exceed the table size")
    {
        const std::vector<std::bitset<num_orbs>> bitstrings{0b0000};
        const std::vector<double> probs{1.0};
        std::mt19937_64 rng;
        const FlipProbabilityTable table(half_orbs);
        CHECK_THROWS_AS(
            std::ignore = recover_configurations(bitstrings, probs, table, {3, 0}, rng),
            std::invalid_argument
        );
    }
}

TEST_CASE_TEMPLATE(
    "Counting uniform flips", BitstringType, std::bitset<4>, boost::dynamic_bitset<>
)
{
    constexpr auto num_orbs = 4;
    constexpr auto half_orbs = num_orbs / 2;
    // Only orbital 0 of each spin species may be flipped by weight, in either
    // direction.
    FlipProbabilityTable table(half_orbs);
    for (std::size_t s = 0; s < 2; ++s) {
        table(s, false, 0) = 1.0;
        table(s, true, 0) = 1.0;
    }
    BitstringType bs;
    SUBCASE("No correction needed")
    {
        set_bitset(num_orbs, bs, 0b0101);
        CHECK(num_uniform_flips(bs, table, {1, 1}) == 0);
    }
    SUBCASE("Correction guided entirely by weight")
    {
        set_bitset(num_orbs, bs, 0b0000);
        CHECK(num_uniform_flips(bs, table, {1, 1}) == 0);
    }
    SUBCASE("Sectors are counted separately and summed")
    {
        // Spin-up: two 0->1 flips, one by weight.  Spin-down: two 1->0 flips,
        // one by weight.
        set_bitset(num_orbs, bs, 0b1100);
        CHECK(num_uniform_flips(bs, table, {2, 0}) == 2);
    }
    SUBCASE("Ineligible positive weights do not count")
    {
        // Orbital 0 already holds the target value in both sectors.
        set_bitset(num_orbs, bs, 0b0001);
        CHECK(num_uniform_flips(bs, table, {2, 0}) == 1);
    }
    SUBCASE("Agrees with the default table")
    {
        const std::array<std::vector<double>, 2> occs{
            std::vector<double>{1.0, 1.0}, std::vector<double>{0.5, 0.5}
        };
        set_bitset(num_orbs, bs, 0b1111);
        CHECK(
            num_uniform_flips(
                bs, flip_probabilities_from_occupancies(occs, {0, 1}), {0, 1}
            ) == 2
        );
    }
    SUBCASE("Invalid inputs are rejected")
    {
        set_bitset(num_orbs, bs, 0b0000);
        CHECK_THROWS_AS(
            std::ignore = num_uniform_flips(bs, table, {3, 0}), std::invalid_argument
        );
        CHECK_THROWS_AS(
            std::ignore = num_uniform_flips(bs, FlipProbabilityTable(3), {1, 1}),
            std::invalid_argument
        );
    }
}

TEST_CASE_TEMPLATE(
    "Bit manipulation", BitstringType, std::bitset<7>, boost::dynamic_bitset<>
)
{
    constexpr std::size_t N = 7;
    BitstringType bs, bs_expected;
    set_bitset(N, bs, 93);
    set_bitset(N, bs_expected, 5);
    Qiskit::addon::sqd::internal::mask_lower_n_bits_inplace(bs, 3);
    CHECK(bs == bs_expected);
}

// Inputs whose correction needs more flips than there are positive-weight
// eligible bits.  These used to be rejected up front; they are now corrected,
// with the remaining flips chosen uniformly (see
// https://github.com/Qiskit/qiskit-addon-sqd-hpc/issues/71).
//
// Both spin sectors have orbitals with average occupancy 0.0, and _p_flip_0_to_1
// returns exactly 0.0 for those -- so those bits are eligible to flip but can
// never be chosen by weight.
TEST_CASE("recover_configurations fills flips that weights cannot choose")
{
    constexpr unsigned int num_orbs = 6;
    std::array<std::vector<double>, 2> occs{
        std::vector<double>{0.0, 0.0, 0.5}, std::vector<double>{0.0, 0.3, 0.5}
    };
    const auto table = flip_probabilities_from_occupancies(occs, {2, 2});
    std::vector<double> probs{1.0};

    // Two inputs that need uniform filling for different reasons:
    //   0: needs 2 alpha flips, only 1 alpha bit carries a positive weight.
    //   4: the eligible alpha bits all carry weight 0, so no alpha flip can be
    //      chosen by weight.
    for (unsigned int value : {0u, 4u}) {
        std::vector<std::bitset<num_orbs>> bitstrings{std::bitset<num_orbs>(value)};
        CHECK(num_uniform_flips(bitstrings[0], table, {2, 2}) == 1);
        std::mt19937 rng(1234u);
        const auto result =
            recover_configurations(bitstrings, probs, occs, {2, 2}, rng);
        REQUIRE(result.first.size() == 1);
        const auto &bs = result.first[0];
        CHECK((bs & std::bitset<num_orbs>(0b000111)).count() == 2);
        CHECK((bs & std::bitset<num_orbs>(0b111000)).count() == 2);
    }
}

// The complement of the case above: zero weights are only a problem when they
// leave too few eligible bits.  An orbital with occupancy 0.0 is ordinary input
// and must still be corrected, not rejected.
TEST_CASE("recover_configurations still corrects when zero weights leave enough bits")
{
    constexpr unsigned int num_orbs = 6;
    std::array<std::vector<double>, 2> occs{
        std::vector<double>{0.0, 0.5, 0.5}, std::vector<double>{0.0, 0.5, 0.5}
    };
    std::vector<std::bitset<num_orbs>> bitstrings{std::bitset<num_orbs>(0)};
    std::vector<double> probs{1.0};
    std::mt19937 rng(1234u);

    // One flip needed per sector, two nonzero-weight eligible bits in each.
    const auto result = recover_configurations(bitstrings, probs, occs, {1, 1}, rng);
    REQUIRE(result.first.size() == 1);
    CHECK(result.first[0].count() == 2); // reached the target Hamming weight
}

#if defined(_OPENMP)

#include <omp.h>

#include "mock_cbrng.hpp"

// With a counter-based RNG, the parallel correction keys each bitstring's random
// stream by its index, so the result -- including the order of the returned
// vectors -- must be identical no matter how many threads run the loop.
//
// Templatized over the counter-based engines rather than written against
// std::philox_engine alone.  philox exists only on C++26, so a philox-only case
// left this -- the headline guarantee of the parallel work -- asserted on exactly
// one row of the CI matrix; the mocks carry it to every row.
#if defined(__cpp_lib_philox_engine)
#define CBRNG_PHILOX_IF_AVAILABLE , std::philox4x64, std::philox4x32
#else
#define CBRNG_PHILOX_IF_AVAILABLE
#endif

TEST_CASE_TEMPLATE_DEFINE(
    "recover_configurations is thread-count independent with a counter-based RNG",
    RNGType, thread_count_independence
)
{
    constexpr unsigned int num_orbs = 8;
    constexpr unsigned int half = num_orbs / 2;

    // A non-trivial workload: many bitstrings needing correction.
    std::vector<std::bitset<num_orbs>> bitstrings;
    std::vector<double> probs;
    for (unsigned int i = 0; i < 500; ++i) {
        bitstrings.emplace_back((i * 37u + 5u) & 0xffu);
        probs.push_back(1.0 + (i % 7));
    }
    std::array<std::vector<double>, 2> occs{
        std::vector<double>(half, 0.3), std::vector<double>(half, 0.7)
    };

    // Single-threaded reference.
    omp_set_num_threads(1);
    RNGType ref_rng(2024u);
    const auto reference =
        recover_configurations(bitstrings, probs, occs, {2, 2}, ref_rng);

    // Sanity: the workload must actually produce something to compare, or the
    // comparisons below would hold vacuously.
    REQUIRE(reference.first.size() > 1);

    // Scope note: this case pins thread-count independence, which is a weaker
    // property than "each item drew its own substream" -- keying every item to
    // the same counter would satisfy it trivially.  Nor is a stronger assertion
    // available here: the recovered set is fixed by the Hamming-weight
    // constraint, so it is identical (36 bitstrings, measured) whether the index
    // is used or replaced by a constant.  That the index is load-bearing is
    // pinned instead by the unit-level `keyed substreams are pairwise distinct`
    // and `... do not overlap under deep draws` cases in
    // test_rng_substitutability.cpp.

    // The result must match at every thread count, element for element.
    for (int nthreads : {2, 4, 8}) {
        omp_set_num_threads(nthreads);
        RNGType rng(2024u);
        const auto result =
            recover_configurations(bitstrings, probs, occs, {2, 2}, rng);
        CHECK(result.first == reference.first);   // same bitstrings, same order
        CHECK(result.second == reference.second); // same frequencies, same order
    }
}
TEST_CASE_TEMPLATE_INVOKE(
    thread_count_independence, MockCBRNGTemplate, MockCBRNGOverloaded,
    MockCBRNGNarrowWord CBRNG_PHILOX_IF_AVAILABLE
);

#endif // _OPENMP
