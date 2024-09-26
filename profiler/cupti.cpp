#include <torch/extension.h>

#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cupti.h>

#include <memory>
#include <stdexcept>

// format string
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      if(_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)          \
          exit(0);                                                      \
      else                                                              \
          exit(-1);                                                     \
    }                                                                   \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time.
// Used to normalized other timestamps
static uint64_t startTimestamp;

size_t buffer_size_;
static std::string trace_string = "";

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

static const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
    return "CUDA";
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
    return "CUDA_MPS";
  default:
    break;
  }

  return "<unknown>";
}

static void
collectActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
      trace_string += string_format("%llu,%llu,MEMCPY,%s,%u,%u,%u,%u\n",
            (unsigned long long) (memcpy->start - startTimestamp),
            (unsigned long long) (memcpy->end - memcpy->start),
            getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
            memcpy->deviceId, memcpy->contextId, memcpy->streamId,
            memcpy->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *) record;
      trace_string += string_format("%llu,%llu,MEMSET,%u,%u,%u,%u,%u\n",
            (unsigned long long) (memset->start - startTimestamp),
            (unsigned long long) (memset->end - memset->start),
            memset->value,
            memset->deviceId, memset->contextId, memset->streamId,
            memset->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *) record;
      trace_string += string_format("%llu,%llu,%s,\"%s\",%u,%u,%u,%d,%d,%d,%d,%d,%d,%u\n",
            (unsigned long long) (kernel->start - startTimestamp),
            (unsigned long long) (kernel->end - kernel->start),
            kindString,
            kernel->name,
            kernel->deviceId, kernel->contextId, kernel->streamId,
            kernel->gridX, kernel->gridY, kernel->gridZ,
            kernel->blockX, kernel->blockY, kernel->blockZ,
            kernel->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      trace_string += string_format("%llu,%llu,DRIVER,%u,%u,%u,%u\n",
            (unsigned long long) (api->start - startTimestamp),
            (unsigned long long) (api->end - api->start),
            api->cbid, api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      trace_string += string_format("%llu,%llu,RUNTIME,%u,%u,%u,%u\n",
            (unsigned long long) (api->start - startTimestamp),
            (unsigned long long) (api->end - api->start),
            api->cbid, api->processId, api->threadId, api->correlationId);
      break;
    }
  default:
    break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint64_t start, end;
  CUPTI_CALL(cuptiGetTimestamp(&start));
  
  uint8_t *bfr = (uint8_t *) malloc(buffer_size_ + ALIGN_SIZE);

  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = buffer_size_;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;

  CUPTI_CALL(cuptiGetTimestamp(&end));
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  uint64_t start, end;

  CUptiResult status;
  CUpti_Activity *record = NULL;

  CUPTI_CALL(cuptiGetTimestamp(&start));

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        collectActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);

  CUPTI_CALL(cuptiGetTimestamp(&end));
}



void
initTrace()
{
  size_t attrValue = 0, attrValueSize = sizeof(size_t);

  trace_string = "";

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the activity API.
  // Some attributes require to be set before any CUDA context is created to be effective,
  // e.g. to be applied to all device buffer allocations (see documentation).
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
  attrValue *= 2;
  buffer_size_ = attrValue;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}


void
flushTrace()
{
  CUPTI_CALL(cuptiActivityFlushAll(0));
}

const char*
finishTrace()
{
  CUPTI_CALL(cuptiActivityFlushAll(0));

  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));

  return trace_string.data();
}

void
timestamp(char * msg)
{
  uint64_t ts;
  CUPTI_CALL(cuptiGetTimestamp(&ts));
  CUPTI_CALL(cuptiActivityFlushAll(0));
  trace_string += string_format("%llu,0,TIMESTAMP,\"%s\"\n", (long long unsigned)(ts - startTimestamp), msg);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_trace", &initTrace, "Initialize CUPTI trace");
  m.def("finish_trace", &finishTrace, "Finish CUPTI trace");
  m.def("timestamp", &timestamp, "Make a timestamp");
  m.def("flush_trace", &flushTrace, "");
}
