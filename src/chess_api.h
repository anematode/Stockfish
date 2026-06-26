//
// Created by Cowpox on 6/26/26.
//

#ifndef SRC_CHESS_API_H
#define SRC_CHESS_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum chessapi_err_t {
    CHESSAPI_OK,

    // The engine is currently in a search.
    CHESSAPI_ENGINE_BUSY,

    // The passed FEN is invalid.
    CHESSAPI_INVALID_FEN,
} chessapi_err_t;

typedef struct chessapi_engine_extra_arg_t {
    const char* key;
    const char* value;
} chessapi_engine_extra_arg_t;

typedef struct chessapi_engine_config_t {
    // Hash size in MiB
    uint64_t hashMegabytes;
    // Thread count
    uint32_t threadCount;

    chessapi_engine_extra_arg_t *extraArgs;
    size_t extraArgc;
} chessapi_engine_config_t;

typedef enum chessapi_search_kind_t {
    CHESSAPI_SEARCH_KIND_DEPTH,
    CHESSAPI_SEARCH_KIND_PERFT,
    CHESSAPI_SEARCH_KIND_FIXED_NODES,
} chessapi_search_kind_t;

typedef struct chessapi_search_t {
    chessapi_search_kind_t searchKind;

    uint32_t depth;
} chessapi_search_t;

#define CHESSAPI_MAX_PVS 256

typedef struct chessapi_pv_t {
    // blah blah
} chessapi_pv_t;

typedef struct chessapi_search_result_t {
    uint64_t nodes;
    uint64_t elapsedMs;

    chessapi_pv_t pvs[CHESSAPI_MAX_PVS];
    size_t pvsCount;
} chessapi_search_result_t;

typedef void (*search_callback_t)(const chessapi_search_result_t* result, void* ctx);

typedef struct chessapi_function_list {
    // Create an engine.
    chessapi_err_t (*createEngine)(void** engine);

    // Destroy an engine.
    void (*destroyEngine)(void* engine);

    // Launch a search synchronously.
    chessapi_err_t (*startSearchSync)(void* engine, chessapi_search_t search, chessapi_search_result_t* out);

    // Launch a search asynchronously.
    chessapi_err_t (*startSearchAsync)(void* engine, chessapi_search_t search, void* ctx, search_callback_t* cb);

    // Stop any current search.
    chessapi_err_t (*stopSearch)(void* engine);

    // Configure the engine parameters.
    chessapi_err_t (*configureEngine)(void* engine, chessapi_engine_config_t config);

    // Set up a position.
    chessapi_err_t (*setEngineFEN)(void* engine, const char* fen, size_t fen_length);
} chessapi_function_list_t;

/** Example use:
 *
 * #include "chess_api.h"
 * #include "stockfish.h"
 *
 * int main() {
 *     void* engine;
 *     chessapi_function_list fns;
 *
 *     chessapi_err_t err = stockfishCreateInstance(&fns, "native");
 *     if (err) abort();
 *
 *     err = fns.createEngine(&engine);
 *     if (err) abort();
 *
 *     err = fns.configureEngine(engine, {
 *          .threads = 8,
 *          .hash = 16,
 *          .extraArgs = nullptr,  // forward non-standard UCI options in here?
 *          .extraArgc = 0,
 *     });
 *     if (err) abort();  // etc.
 *
 *     const char fen[] = "r....";
 *     err = fns.setFEN(engine, fen, sizeof(fen));
 *
 *     chessapi_search_t searchConfig{};
 *     searchConfig.kind = CHESSAPI_SEARCH_KIND_DEPTH;
 *     searchConfig.depth = 5;
 *     searchConfig.multiPV = 1;
 *
 *     chessapi_search_result_t searchResult{};
 *     err = fns.searchSync(engine, &searchConfig, &searchResult);
 *
 *     printf("%zu nodes, %d eval\n", searchResult.nodesSearched, searchResult.eval);
 *
 *     searchConfig = 10;
 *     err = fns.searchAsync(engine, &searchConfig, [] (const chessapi_search_result_t* result, void* ctx) {
 *          // e.g. print progress
 *     });
 *
 *     err = fns.setFEN(engine, ...);   // CHESSAPI_ERR_BUSY
 *
 *     sleep(2);
 *     err = fns.searchStop(engine);
 * }
 */

#ifdef __cplusplus
}
#endif

#endif //SRC_CHESS_API_H
