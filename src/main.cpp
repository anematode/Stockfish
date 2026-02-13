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
// One primary Engine owns the single network copy.  N worker Engines share
// that network via the shared-network constructor, so depth-5 searches run
// in parallel across positions without duplicating the ~100MB network.
// Each worker Engine uses 1 search thread internally.
//
// Both rawEval (from Eval::evaluate) and depth5Search (from InfoFull.rawScore)
// are in SF internal Value units, from the side-to-move perspective.  No
// centipawn conversion is applied.

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

// ---------------------------------------------------------------------------
// Evaluate the objective: average |rawEval(pos) - depth5Search(pos)| over all
// positions.  Both values are in SF internal Value units (side-to-move
// perspective).
//
// Raw NN evals are parallelised across std::threads sharing the primary
// engine's network (read-only).  Depth-5 searches are parallelised across
// the worker engine pool, each running a single-threaded search.
// ---------------------------------------------------------------------------
static double evaluate_objective(Engine&                                engine,
                                 std::vector<std::unique_ptr<Engine>>&  workers,
                                 const std::vector<std::string>&        fens) {
    const int numPositions = static_cast<int>(fens.size());
    if (numPositions == 0)
        return 0.0;

    const int numWorkers = static_cast<int>(workers.size());

    // --- Phase 1: parallel raw NN evals ---
    std::vector<Value> rawEvals(numPositions, VALUE_ZERO);
    std::vector<bool>  valid(numPositions, false);

    {
        std::vector<std::thread> evalThreads;
        for (int t = 0; t < numWorkers; ++t) {
            evalThreads.emplace_back([&, t]() {
                // Heap-allocate once per thread, reuse across positions.
                auto accumulators = std::make_unique<AccumulatorStack>();
                auto caches = std::make_unique<AccumulatorCaches>(*engine.get_networks());

                for (int i = t; i < numPositions; i += numWorkers) {
                    Position     pos;
                    StateListPtr states(new std::deque<StateInfo>(1));
                    pos.set(fens[i], false, &states->back());

                    if (pos.checkers())
                        continue;

                    caches->clear(*engine.get_networks());
                    rawEvals[i] =
                      Eval::evaluate(*engine.get_networks(), pos, *accumulators, *caches, 0);
                    valid[i] = true;
                }
            });
        }
        for (auto& w : evalThreads)
            w.join();
    }

    // --- Phase 2: parallel depth-5 searches across worker engines ---
    std::vector<Value> searchEvals(numPositions, VALUE_ZERO);

    {
        std::vector<std::thread> searchThreads;
        for (int t = 0; t < numWorkers; ++t) {
            searchThreads.emplace_back([&, t]() {
                Engine& eng = *workers[t];
                for (int i = t; i < numPositions; i += numWorkers) {
                    if (!valid[i])
                        continue;

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

                    searchEvals[i] = searchVal;
                }
            });
        }
        for (auto& w : searchThreads)
            w.join();
    }

    // --- Compute average error ---
    double totalError = 0.0;
    int    count      = 0;
    for (int i = 0; i < numPositions; ++i) {
        if (!valid[i])
            continue;
        totalError +=
          std::abs(static_cast<double>(rawEvals[i]) - static_cast<double>(searchEvals[i]));
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

    const int numWorkers =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

    // Primary engine owns the single network copy.
    Engine engine(argc > 0 ? std::optional<std::string>(argv[0]) : std::nullopt);
    engine.set_on_verify_networks(
      [](std::string_view msg) { std::cout << msg << std::endl; });
    engine.load_networks();

    // Worker engines share the primary engine's network (no copy).
    std::cout << "Creating " << numWorkers
              << " worker engine(s) sharing one network copy..." << std::endl;

    std::vector<std::unique_ptr<Engine>> workers;
    for (int i = 0; i < numWorkers; ++i) {
        auto w = std::make_unique<Engine>(
          argc > 0 ? std::optional<std::string>(argv[0]) : std::nullopt,
          engine.get_networks());
        w->set_on_verify_networks(
          [](std::string_view msg) { std::cout << msg << std::endl; });
        workers.push_back(std::move(w));
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
    std::vector<double> theta = gather_params((*engine.get_networks()).big);

    std::mt19937                rng(42);
    std::bernoulli_distribution coin(0.5);

    std::cout << "Starting SPSA optimisation (" << TOTAL_PARAMS << " parameters, "
              << fens.size() << " positions, " << maxIter << " iterations, "
              << numWorkers << " worker(s))" << std::endl;

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
        engine.get_networks().modify_and_replicate(
          [&](Eval::NNUE::Networks& nets) { scatter_params(nets.big, thetaPlus); });
        double fPlus = evaluate_objective(engine, workers, fens);

        // Evaluate f(theta-)
        engine.get_networks().modify_and_replicate(
          [&](Eval::NNUE::Networks& nets) { scatter_params(nets.big, thetaMinus); });
        double fMinus = evaluate_objective(engine, workers, fens);

        // Gradient estimate and update
        for (int i = 0; i < TOTAL_PARAMS; ++i) {
            double ghat = (fPlus - fMinus) / (2.0 * ck * delta[i]);
            theta[i] -= ak * ghat;
        }

        // Set current theta and print progress
        engine.get_networks().modify_and_replicate(
          [&](Eval::NNUE::Networks& nets) { scatter_params(nets.big, theta); });
        double curError = evaluate_objective(engine, workers, fens);
        std::cout << "SPSA iter " << (k + 1) << "/" << maxIter
                  << "  avg_error=" << curError
                  << "  f+=" << fPlus << "  f-=" << fMinus
                  << "  ak=" << ak << "  ck=" << ck << std::endl;
    }

    // Final scatter & save
    engine.get_networks().modify_and_replicate(
      [&](Eval::NNUE::Networks& nets) { scatter_params(nets.big, theta); });

    std::pair<std::optional<std::string>, std::string> files[2] = {
        {std::optional<std::string>("perturbed.nnue"), ""},
        {std::nullopt, ""}
    };
    engine.save_network(files);

    std::cout << "Done. Perturbed network saved to perturbed.nnue" << std::endl;
    return 0;
}
