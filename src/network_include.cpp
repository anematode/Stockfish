//
// Created by toystory on 10/5/25.
//

#include "network_include.h"
#include "evaluate.h"

#define INCBIN_SILENCE_BITCODE_WARNING
#include "incbin/incbin.h"

INCBIN(EmbeddedNNUEBig, EvalFileDefaultNameBig);
INCBIN(EmbeddedNNUESmall, EvalFileDefaultNameSmall);