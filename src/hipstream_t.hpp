#pragma once

#include <hsa/hsa.h>

class ihipStream_t {
public:
    // ACCESSORS
    hsa_agent_t hsa_agent() const noexcept { return {}; }
};