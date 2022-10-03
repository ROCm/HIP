/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>

/*
Guarantees total ordering between parent and child thread
PARENT      CHILD
THREAD      THREAD
TestPart1
         \
          \
           \
            TestPart2
           /
          /
         /
TestPart3
         \
          \
           \
            TestPart4
Usage:
Define a derived class which inherits from ThreadedZigZagTest instantiated with that selfsame class,
which implements the appropriate test methods
class DerivedTestClass : public ThreadedZigZagTest<DerivedTestClass> {
  void TestPart1() {...}
  void TestPart2() {...}
  void TestPart3() {...}
  void TestPart4() {...}
};
The derived class can contain state that the test requires.
*/

template <typename T> class ThreadedZigZagTest {
 public:
  void run() {
    // 1.
    static_cast<T*>(this)->TestPart1();

    auto t = std::thread([this] {
      // 2.
      static_cast<T*>(this)->TestPart2();

      {
        std::lock_guard<std::mutex> lock(mtx_);
        ready_ = true;
      }
      cv_.notify_one();

      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !ready_; });
      }

      // 4.
      static_cast<T*>(this)->TestPart4();
    });

    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this] { return ready_; });
    }

    // 3.
    static_cast<T*>(this)->TestPart3();

    {
      std::lock_guard<std::mutex> lock(mtx_);
      ready_ = false;
    }
    cv_.notify_one();

    // Finalize
    t.join();
    HIP_CHECK_THREAD_FINALIZE();
  }

  void TestPart1() const {}
  void TestPart2() const {}
  void TestPart3() const {}
  void TestPart4() const {}

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool ready_ = false;
};