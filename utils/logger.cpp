#include "utils/logger.h"

#include "absl/log/initialize.h"

void InitLogger() {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        absl::InitializeLog();
    }
}
