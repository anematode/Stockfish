/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// SPSA perturbation of NNUE fc_2 weights to minimize |rawEval - depth5Search|.
// Replaces the normal UCI main loop.
//
// Both rawEval (from Eval::evaluate) and depth5Search (from InfoFull.rawScore)
// are in SF internal Value units, from the side-to-move perspective.  No
// centipawn conversion is applied.
//
// Multiple Engine instances are created (one per hardware thread) and kept
// alive for the entire run to parallelise the expensive depth-5 searches.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "bitboard.h"
#include "engine.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/nnue_architecture.h"
#include "position.h"
#include "search.h"
#include "types.h"
#include "uci.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

// ---------------------------------------------------------------------------
// Collect all fc_2 parameters (weights + biases) across all LayerStacks into a
// flat double vector.  fc_2 has OutputDimensions=1, InputDimensions=FC_1_OUTPUTS.
// We perturb: 1 bias + FC_1_OUTPUTS weights per LayerStack.
// ---------------------------------------------------------------------------

static constexpr int FC2_WEIGHTS   = L3Big;                        // 32
static constexpr int FC2_BIASES    = 1;                            // 1
static constexpr int FC2_PER_STACK = FC2_WEIGHTS + FC2_BIASES;     // 33
static constexpr int TOTAL_PARAMS  = FC2_PER_STACK * LayerStacks;  // 33*8 = 264
static constexpr int SEARCH_DEPTH  = 5;

// Gather current fc_2 parameters into a flat vector of doubles.
static std::vector<double> gather_params(const NetworkBig& net) {
    std::vector<double> theta(TOTAL_PARAMS);
    for (std::size_t s = 0; s < LayerStacks; ++s) {
        const auto& fc2 = net.get_network(s).get_fc_2();
        theta[s * FC2_PER_STACK] = static_cast<double>(fc2.biases[0]);
        for (int w = 0; w < FC2_WEIGHTS; ++w)
            theta[s * FC2_PER_STACK + 1 + w] = static_cast<double>(fc2.weights[w]);
    }
    return theta;
}

// Scatter a flat vector of doubles back into fc_2 parameters, clamping to the
// representable range of the underlying integer types.
static void scatter_params(NetworkBig& net, const std::vector<double>& theta) {
    for (std::size_t s = 0; s < LayerStacks; ++s) {
        auto& fc2 = net.get_network(s).get_fc_2();
        double b  = std::clamp(theta[s * FC2_PER_STACK],
                               static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                               static_cast<double>(std::numeric_limits<std::int32_t>::max()));
        fc2.biases[0] = static_cast<std::int32_t>(std::round(b));
        for (int w = 0; w < FC2_WEIGHTS; ++w) {
            double v = std::clamp(theta[s * FC2_PER_STACK + 1 + w], -128.0, 127.0);
            fc2.weights[w] = static_cast<std::int8_t>(std::round(v));
        }
    }
}

// Scatter parameters into every engine's network.
static void scatter_to_all(std::vector<std::unique_ptr<Engine>>& engines,
                           const std::vector<double>&            theta) {
    for (auto& eng : engines)
        eng->get_networks().modify_and_replicate(
          [&](Eval::NNUE::Networks& nets) { scatter_params(nets.big, theta); });
}

// ---------------------------------------------------------------------------
// Evaluate the objective: average |rawEval(pos) - depth5Search(pos)| over all
// positions.  Both values are in SF internal Value units (side-to-move
// perspective).  Work is split across the engine pool with one thread per
// engine so that the depth-5 searches run in parallel.
// ---------------------------------------------------------------------------
static double evaluate_objective(std::vector<std::unique_ptr<Engine>>& engines,
                                 const std::vector<std::string>&       fens) {
    const int numPositions = static_cast<int>(fens.size());
    if (numPositions == 0)
        return 0.0;

    const int numEngines = static_cast<int>(engines.size());

    struct PosResult {
        Value rawEval    = VALUE_ZERO;
        Value searchEval = VALUE_ZERO;
        bool  valid      = false;
    };
    std::vector<PosResult> results(numPositions);

    // One thread per engine; each engine processes its stripe of positions.
    std::vector<std::thread> workers;
    for (int t = 0; t < numEngines; ++t) {
        workers.emplace_back([&, t]() {
            Engine& eng = *engines[t];
            for (int i = t; i < numPositions; i += numEngines) {
                // --- raw NN eval (internal Value, side-to-move) ---
                Position     pos;
                StateListPtr states(new std::deque<StateInfo>(1));
                pos.set(fens[i], false, &states->back());

                if (pos.checkers())
                    continue;

                auto accumulators = std::make_unique<AccumulatorStack>();
                auto caches       = std::make_unique<AccumulatorCaches>(*eng.get_networks());
                Value rv = Eval::evaluate(*eng.get_networks(), pos, *accumulators, *caches, 0);

                // --- depth-5 search ---
                // InfoFull.rawScore carries the raw internal Value (side-to-
                // move perspective), bypassing the centipawn conversion that
                // Score::InternalUnits applies.
                eng.set_position(fens[i], {});

                Search::LimitsType limits;
                limits.depth     = SEARCH_DEPTH;
                limits.startTime = 0;

                Value searchVal = VALUE_ZERO;
                eng.set_on_update_full([&](const Engine::InfoFull& info) {
                    searchVal = info.rawScore;
                });
                eng.set_on_bestmove([](std::string_view, std::string_view) {});

                eng.go(limits);
                eng.wait_for_search_finished();

                results[i].rawEval    = rv;
                results[i].searchEval = searchVal;
                results[i].valid      = true;
            }
        });
    }
    for (auto& w : workers)
        w.join();

    // --- Compute average error ---
    double totalError = 0.0;
    int    count      = 0;
    for (int i = 0; i < numPositions; ++i) {
        if (!results[i].valid)
            continue;
        totalError +=
          std::abs(static_cast<double>(results[i].rawEval) - static_cast<double>(results[i].searchEval));
        ++count;
    }
    return count > 0 ? totalError / count : 0.0;
}

// ---------------------------------------------------------------------------
// main - SPSA loop
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::cout << engine_info() << std::endl;

    Bitboards::init();
    Position::init();

    // Determine number of parallel engines (= hardware threads).
    const int numEngines =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

    std::cout << "Creating " << numEngines << " engine(s) for parallel evaluation..."
              << std::endl;

    // Create and configure all engines once; they persist for the entire run.
    std::vector<std::unique_ptr<Engine>> engines;
    for (int i = 0; i < numEngines; ++i) {
        auto eng = std::make_unique<Engine>(
          argc > 0 ? std::optional<std::string>(argv[0]) : std::nullopt);
        eng->set_on_verify_networks(
          [](std::string_view msg) { std::cout << msg << std::endl; });
        eng->load_networks();
        engines.push_back(std::move(eng));
    }

    // Load positions from file (one FEN per line)
    std::vector<std::string> fens;
    {
        std::ifstream fin("positions.pgn");
        if (!fin.is_open()) {
            std::cerr << "Error: could not open positions.pgn" << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(fin, line)) {
            if (!line.empty())
                fens.push_back(line);
        }
        std::cout << "Loaded " << fens.size() << " positions." << std::endl;
    }

    if (fens.empty()) {
        std::cerr << "No positions loaded - nothing to do." << std::endl;
        return 1;
    }

    // ---- SPSA hyper-parameters ----
    const int    maxIter = 200;
    const double a0      = 0.5;
    const double c0      = 1.0;
    const double alpha   = 0.602;
    const double gamma   = 0.101;
    const double A       = 10.0;

    // Current parameters (doubles, will be rounded when scattered)
    std::vector<double> theta = gather_params((*engines[0]->get_networks()).big);

    std::mt19937                rng(42);
    std::bernoulli_distribution coin(0.5);

    std::cout << "Starting SPSA optimisation (" << TOTAL_PARAMS << " parameters, "
              << fens.size() << " positions, " << maxIter << " iterations, "
              << numEngines << " engine(s))" << std::endl;

    for (int k = 0; k < maxIter; ++k) {
        double ak = a0 / std::pow(k + 1 + A, alpha);
        double ck = c0 / std::pow(k + 1, gamma);

        // Generate random perturbation vector delta_k in {-1, +1}^p
        std::vector<double> delta(TOTAL_PARAMS);
        for (auto& d : delta)
            d = coin(rng) ? 1.0 : -1.0;

        // theta+ and theta-
        std::vector<double> thetaPlus(TOTAL_PARAMS), thetaMinus(TOTAL_PARAMS);
        for (int i = 0; i < TOTAL_PARAMS; ++i) {
            thetaPlus[i]  = theta[i] + ck * delta[i];
            thetaMinus[i] = theta[i] - ck * delta[i];
        }

        // Evaluate f(theta+)
        scatter_to_all(engines, thetaPlus);
        double fPlus = evaluate_objective(engines, fens);

        // Evaluate f(theta-)
        scatter_to_all(engines, thetaMinus);
        double fMinus = evaluate_objective(engines, fens);

        // Gradient estimate and update
        for (int i = 0; i < TOTAL_PARAMS; ++i) {
            double ghat = (fPlus - fMinus) / (2.0 * ck * delta[i]);
            theta[i] -= ak * ghat;
        }

        // Set current theta and print progress
        scatter_to_all(engines, theta);
        double curError = evaluate_objective(engines, fens);
        std::cout << "SPSA iter " << (k + 1) << "/" << maxIter
                  << "  avg_error=" << curError
                  << "  f+=" << fPlus << "  f-=" << fMinus
                  << "  ak=" << ak << "  ck=" << ck << std::endl;
    }

    // Final scatter & save
    scatter_to_all(engines, theta);

    std::pair<std::optional<std::string>, std::string> files[2] = {
        {std::optional<std::string>("perturbed.nnue"), ""},
        {std::nullopt, ""}
    };
    engines[0]->save_network(files);

    std::cout << "Done. Perturbed network saved to perturbed.nnue" << std::endl;
    return 0;
}
