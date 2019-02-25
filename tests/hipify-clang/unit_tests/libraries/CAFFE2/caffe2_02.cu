// RUN: %run_test hipify "%s" "%t" %hipify_args "-roc" %clang_args

// NOTE: Nonworking code just for conversion testing

// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

namespace caffe2 {

// Operator Definition.
struct OperatorDef {
  int input = 1;
  int output = 2;
  int name = 3;
};

class OperatorBase;
class Workspace;

template <class T>
class Observable {
 public:
  Observable() = default;

  Observable(Observable&&) = default;
  Observable& operator =(Observable&&) = default;

  virtual ~Observable() = default;
};

template <class T>
class ObserverBase {
 public:
  explicit ObserverBase(T* subject) : subject_(subject) {}

  virtual void Start() {}
  virtual void Stop() {}

  virtual std::string debugInfo() {
    return "Not implemented.";
  }

  virtual ~ObserverBase() noexcept {};

  T* subject() const {
    return subject_;
  }

 protected:
  T* subject_;
};

typedef ObserverBase<OperatorBase> OperatorObserver;

class OperatorBase : public Observable<OperatorBase> {
 public:
  explicit OperatorBase(const OperatorDef& operator_def, Workspace* ws);
  virtual ~OperatorBase() noexcept {}
};

template <class Context>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {
  }
  ~Operator() noexcept override {}
};

template <class Context>
class DummyEmptyOp : public Operator<Context> {
 public:
  DummyEmptyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() final { return true; }
};


class CUDAContext {
public:
  CUDAContext();
  virtual ~CUDAContext() noexcept {}
};

#define REGISTER_CUDA_OPERATOR(name, ...)                           \
  void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();              \
  static void CAFFE_ANONYMOUS_VARIABLE_CUDA##name() { \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                 \
  }

#define REGISTER_CUDA_OPERATOR_CREATOR(key, ...)

// CHECK: REGISTER_HIP_OPERATOR(Operator, DummyEmptyOp<HIPContext>);
REGISTER_CUDA_OPERATOR(Operator, DummyEmptyOp<CUDAContext>);
// CHECK: REGISTER_HIP_OPERATOR_CREATOR(Operator, DummyEmptyOp<HIPContext>);
REGISTER_CUDA_OPERATOR_CREATOR(Operator, DummyEmptyOp<CUDAContext>);

}
