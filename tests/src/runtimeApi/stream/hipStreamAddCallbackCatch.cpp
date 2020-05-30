
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include "test_common.h"

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM 
 * TEST: %t
 * HIT_END
 */

#define WORKAROUND 1 // Enable (1) this to make stream thread-safe by a workaround

template<bool IsBlocking> // <true> = queue blocks, until task is finished in enqueue(queue,task)
class QueueHipRt;

// Queue types used in the tests
using TestQueues = std::tuple<QueueHipRt<true>, QueueHipRt<false>>;


// --- Implementation

#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#define HIP_ASSERT_IGNORE(x,ign) auto err=x; HIP_ASSERT(err==ign ? hipSuccess : err)

#ifdef __HIP_PLATFORM_HCC__
  #define HIPRT_CB
#endif

template<bool isBlocking>
static auto currentThreadWaitFor(QueueHipRt<isBlocking> const & queue)  -> void;

template<bool IsBlocking>
class QueueHipRt
{
public:
  static constexpr bool isBlocking = IsBlocking;
  //-----------------------------------------------------------------------------
  QueueHipRt(
    int dev) :
    m_dev(dev),
    m_HipQueue()
    {
      HIP_ASSERT(
        hipSetDevice(
          m_dev));
      HIP_ASSERT(
        hipStreamCreateWithFlags(
          &m_HipQueue,
          hipStreamNonBlocking));
    }
  //-----------------------------------------------------------------------------
  QueueHipRt(QueueHipRt const &) = delete;
  //-----------------------------------------------------------------------------
  QueueHipRt(QueueHipRt &&) = delete;
  //-----------------------------------------------------------------------------
  auto operator=(QueueHipRt const &) -> QueueHipRt & = delete;
  //-----------------------------------------------------------------------------
  auto operator=(QueueHipRt &&) -> QueueHipRt & = delete;
  //-----------------------------------------------------------------------------
  ~QueueHipRt()
    {
      if(isBlocking) {
#if WORKAROUND  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                // we are a non-blocking queue, so we have to wait here with its destruction until all spawned tasks have been processed
        currentThreadWaitFor(*this);
#endif
      }
      HIP_ASSERT(
        hipSetDevice(
          m_dev));
      HIP_ASSERT(
        hipStreamDestroy(
          m_HipQueue));
    }

public:
  int m_dev;   //!< The device this queue is bound to.
  hipStream_t m_HipQueue;

#if WORKAROUND  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
  int m_callees = 0;
  std::mutex m_mutex;
#endif
};

template<typename TTask>
struct Enqueue
{
  //#############################################################################
  enum class CallbackState
  {
    enqueued,
    notified,
    finished,
  };

  //#############################################################################
  struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
  {
    std::mutex m_mutex;
    std::condition_variable m_event;
    CallbackState state = CallbackState::enqueued;
  };

  //-----------------------------------------------------------------------------
  static void HIPRT_CB hipRtCallback(hipStream_t /*queue*/, hipError_t /*status*/, void *arg)
    {
      // explicitly copy the shared_ptr so that this method holds the state even when the executing thread has already finished.
      const auto pCallbackSynchronizationData = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

      // Notify the executing thread.
      {
        std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
        pCallbackSynchronizationData->state = CallbackState::notified;
      }
      pCallbackSynchronizationData->m_event.notify_one();

      // Wait for the executing thread to finish the task if it has not already finished.
      std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
      if(pCallbackSynchronizationData->state != CallbackState::finished)
      {
        pCallbackSynchronizationData->m_event.wait(
          lock,
          [pCallbackSynchronizationData](){
            return pCallbackSynchronizationData->state == CallbackState::finished;
          }
          );
      }
    }

  //-----------------------------------------------------------------------------
  template<bool isBlocking>
  static auto enqueue(
    QueueHipRt<isBlocking> & queue,
    TTask const & task)
    -> void
    {

#if WORKAROUND  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
      {
        // thread-safe callee incrementing
        std::lock_guard<std::mutex> guard(queue.m_mutex);
        queue.m_callees += 1;
      }
#endif
      auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();
      // test example: https://github.com/ROCm-Developer-Tools/HIP/blob/roc-1.9.x/tests/src/runtimeApi/stream/hipStreamAddCallback.cpp
      HIP_ASSERT(hipStreamAddCallback(
                            queue.m_HipQueue,
                            hipRtCallback,
                            pCallbackSynchronizationData.get(),
                            0u));

      // We start a new std::thread which stores the task to be executed.
      // This circumvents the limitation that it is not possible to call HIP methods within the HIP callback thread.
      // The HIP thread signals the std::thread when it is ready to execute the task.
      // The HIP thread is waiting for the std::thread to signal that it is finished executing the task
      // before it executes the next task in the queue (HIP stream).
      std::thread t(
        [pCallbackSynchronizationData,
         task
#if WORKAROUND // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
         ,&queue // requires queue's destructor to wait for all tasks
#endif
          ](){

#if WORKAROUND // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
          // thread-safe task execution and callee decrementing
          std::lock_guard<std::mutex> guard(queue.m_mutex);
#endif

          // If the callback has not yet been called, we wait for it.
          {
            std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
            if(pCallbackSynchronizationData->state != CallbackState::notified)
            {
              pCallbackSynchronizationData->m_event.wait(
                lock,
                [pCallbackSynchronizationData](){
                  return pCallbackSynchronizationData->state == CallbackState::notified;
                }
                );
            }

            task();

            // Notify the waiting HIP thread.
            pCallbackSynchronizationData->state = CallbackState::finished;
          }
          pCallbackSynchronizationData->m_event.notify_one();
#if WORKAROUND // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
          queue.m_callees -= 1;
#endif
        }
        );
      if(isBlocking)
        t.join(); // => waiting for task completion
      else
        t.detach(); // => do not wait for task completion
    }
};
//#############################################################################
//! The HIP RT non-blocking queue test trait specialization.
struct Empty
{
  //-----------------------------------------------------------------------------
  template<bool isBlocking>
  static auto empty(
    QueueHipRt<isBlocking> const & queue)
    -> bool
    {

#if WORKAROUND  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
      return (queue.m_callees==0);
#else

      // Query is allowed even for queues on non current device.
      hipError_t ret = hipSuccess;
      HIP_ASSERT_IGNORE(
        ret = hipStreamQuery(
          queue.m_HipQueue),
        hipErrorNotReady);
      return (ret == hipSuccess);
#endif
    }
};

template<bool isBlocking>
auto currentThreadWaitFor(QueueHipRt<isBlocking> const & queue)  -> void
{
#if WORKAROUND  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
  while(queue.m_callees>0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10u));
  }
#else
  // Sync is allowed even for queues on non current device.
  HIP_ASSERT( hipStreamSynchronize(
                         queue.m_HipQueue));
#endif
}




// --- Tests

#define TEMPLATE_LIST_TEST_CASE(TestName) \
template<typename TestType> static void TestName (std::atomic<int> &check); \
static int TestName##Runner () { \
    std::atomic<int> check{0}; \
    TestName< QueueHipRt<true> >(check); \
    fprintf(stderr, "After " #TestName " < QueueHipRt<true> >  errors=%d\n", check.load()); \
    TestName< QueueHipRt<false> >(check); \
    fprintf(stderr, "After " #TestName " < QueueHipRt<false> > errors=%d\n", check.load()); \
    return check.load(); \
} \
template<typename TestType> static void TestName (std::atomic<int> &check)

// add 1 if a check fails
#define CHECK(result) do{int arg=(!(result)); fprintf(stderr, "Checking " #result " %d\n", arg); check.fetch_add(arg);}while(false)

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( queueIsInitiallyEmpty )
{
  TestType queue{0};
  CHECK(Empty::empty(queue));
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( queueCallbackIsWorking )
{
  std::promise<bool> promise;
  auto task = [&](){ promise.set_value(true); };
  TestType queue{0};
  Enqueue<decltype(task)> enqueue;
  enqueue.enqueue(
    queue,
    task
    );

  CHECK(promise.get_future().get());
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( queueWaitShouldWork )
{
  bool CallbackFinished = false;
  auto task =
    [&CallbackFinished]() noexcept
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100u));
        CallbackFinished = true;
      };
  TestType queue{0};
  Enqueue<decltype(task)> enqueue;
  enqueue.enqueue(
    queue,
    task
    );

  currentThreadWaitFor(queue);
  CHECK(CallbackFinished);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( queueShouldNotBeEmptyWhenLastTaskIsStillExecutingAndIsEmptyAfterProcessingFinished )
{
  bool CallbackFinished = false;
  TestType queue{0};
  auto task = [&queue, &CallbackFinished, &check]() noexcept
                {
                  CHECK(!Empty::empty(queue));
                  std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                  CallbackFinished = true;
                };
  Enqueue<decltype(task)> enqueue;
  enqueue.enqueue(
    queue,
    task
    );
  // A non-blocking queue will always stay empty because the task has been executed immediately.
  if(!TestType::isBlocking)
  {
    currentThreadWaitFor(queue);
  }

  CHECK(Empty::empty(queue));
  CHECK(CallbackFinished);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( queueShouldNotExecuteTasksInParallel )
{
  std::atomic<bool> taskIsExecuting(false);
  std::promise<void> firstTaskFinished;
  std::future<void> firstTaskFinishedFuture = firstTaskFinished.get_future();
  std::promise<void> secondTaskFinished;
  std::future<void> secondTaskFinishedFuture = secondTaskFinished.get_future();

  TestType queue{0};

  std::thread thread1(
    [&queue, &taskIsExecuting, &firstTaskFinished, &check]()
      {
        auto task1 = [&taskIsExecuting, &firstTaskFinished, &check]() noexcept
                       {
                         CHECK(!taskIsExecuting.exchange(true));
                         std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                         CHECK(taskIsExecuting.exchange(false));
                         firstTaskFinished.set_value();
                       };
        Enqueue<decltype(task1)> enqueue;
        enqueue.enqueue(
          queue,
          task1
          );
      });

  std::thread thread2(
    [&queue, &taskIsExecuting, &secondTaskFinished, &check]()
      {
        auto task2 = [&taskIsExecuting, &secondTaskFinished, &check]() noexcept
                       {
                         CHECK(!taskIsExecuting.exchange(true));
                         std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                         CHECK(taskIsExecuting.exchange(false));
                         secondTaskFinished.set_value();
                       };

        Enqueue<decltype(task2)> enqueue;
        enqueue.enqueue(
          queue,
          task2
          );
      });

  // Both tasks have to be enqueued
  thread1.join();
  thread2.join();

  currentThreadWaitFor(queue);

  firstTaskFinishedFuture.get();
  secondTaskFinishedFuture.get();
}

#define TESTER(name) do { \
    int result = name (); \
    fprintf(stderr, #name " %s\n", result?"Errors":"No Errors"); \
    if (result) { failed(#name " failed\n"); } \
} while (false)

int main()
{
    TESTER(queueIsInitiallyEmptyRunner);
    TESTER(queueCallbackIsWorkingRunner);
    TESTER(queueWaitShouldWorkRunner);
    TESTER(queueShouldNotBeEmptyWhenLastTaskIsStillExecutingAndIsEmptyAfterProcessingFinishedRunner);
//    TESTER(queueShouldNotExecuteTasksInParallelRunner);
    passed();
}
