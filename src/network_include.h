//
// Created by toystory on 10/5/25.
//

#ifndef STOCKFISH_NETWORK_INCLUDE_H
#define STOCKFISH_NETWORK_INCLUDE_H

#include "evaluate.h"

#define INCBIN_SILENCE_BITCODE_WARNING
#include "incbin/incbin.h"

INCBIN_EXTERN(EmbeddedNNUEBig);
INCBIN_EXTERN(EmbeddedNNUESmall);

#endif //STOCKFISH_NETWORK_INCLUDE_H