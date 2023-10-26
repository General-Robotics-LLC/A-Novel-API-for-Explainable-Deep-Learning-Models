#pragma once

#include <c10/util/Optional.h>
#include <string>

namespace c10d {

enum class EventKind { CollectiveStart, CollectiveEnd };

TORCH_API void enable_event_collection(int sync_pipe);

namespace details {

struct TORCH_API EventInfo {
  EventKind event_kind;
  std::string pg_name;
  std::string backend;
  int64_t sequence_number;
  std::string operation;
  int64_t timestamp;
  c10::optional<float> duration_ms;
  int64_t drop_count;
};

// TODO do we want to expose something else here?
TORCH_API bool dequeue_c10d_event(EventInfo& evt);
TORCH_API void enqueue_c10d_event(EventInfo&& evt);

} // namespace details
} // namespace c10d
