#pragma once

#include "../common/voxtral_types.h"
#include <string>
#include <vector>

const std::string & token_bytes_for_id(const voxtral_model & model, int32_t token_id);
std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens);
bool precompute_decoder_ada_scale(voxtral_context * ctx);
