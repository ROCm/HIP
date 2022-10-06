# HIP C++ Feature

## C++ 11

### Rvalue References

```cpp
struct Y {
  int x;
};

__device__ void do_something(Y &&val) { val.x += 1; }

__global__ void kernel() {
  Y y{10};
  // do_something(y); // does not compile since the argument is an lvalue
  do_something(std::move(y));
}

int main() { kernel<<<1, 1>>>(); }
```

### Rvalue References for `*this`

```cpp
struct Sample {
  __host__ __device__ void callMe() & { printf("Lval Func\n"); }
  __host__ __device__ void callMe() && { printf("Rval Func\n"); }
};

__global__ void kernel() {
  Sample s;
  s.callMe();        // prints Lval Func
  Sample().callMe(); // prints Rval Func
}

int main() { kernel<<<1, 1>>>(); }
```

### Variadic templates, Static Assertions, `auto` Variables

```cpp
template <typename T> __host__ __device__ T add(T val) { return val; }

template <typename T, typename... Targs>
__host__ __device__ T add(T val, Targs... pVal) {
  static_assert(std::is_arithmetic<T>::value, "Not a valid type");
  return val + add(pVal...);
}

template <typename T, typename... Targs>
__global__ void kernel(T *ptr, Targs... args) {
  auto &&sum = add(args...);
  *ptr = sum;
}

// Or something like

__device__ int &getX(int &x) { return ++x; }
__device__ int getY(int &x) { return x + 10; }

__global__ void kernel() {
  int X = 0;
  auto &&x = getX(X);
  auto &&y = getY(X);

  // Init with value or initializer list
  auto val{10};
  auto list = {10};
}

int main() { kernel<<<1, 1>>>(); }
```

### Non-static Data Member Initialization

```cpp
struct S {
  int a = 1;
  int b = 2;
};

__global__ void kernel() {
  S s; // s.a == 1 and s.b == 2
}
int main() { kernel<<<1, 1>>>(); }
```

### Lambda Device Functions

```cpp
template <typename T> __global__ void kernel(T f) { f(); }

int main() {
  auto func = [=] __device__() { printf("In Kernel\n"); };
  kernel<<<1, 1>>>(func);
  hipDeviceSynchronize();
}
```

### `decltype` Usage

```cpp
template <typename T> __device__ T ret() {
  T x{0};
  return x;
}

template <typename T> __global__ void kernel() {
  decltype(ret<T>()) a;
  int i = 0;
  decltype(i) j = i + 1;
}
int main() { kernel<float><<<1, 1>>>(); }
```

### Default Template Arguments

```cpp
template <int N = 5> __global__ void kernel(int x) { x += N; }
int main() {
  kernel<<<1, 1>>>(1);
  kernel<-2><<<1, 1>>>(1);
}
```

### Template Alias

```cpp
template <typename T> struct Alloc {};

template <typename T, typename U> struct Vector {};

template <typename T> using V = Vector<T, Alloc<T>>;

template <typename T> __global__ void kernel(T x) { V<T> v; }

int main() { kernel<<<1, 1>>>(5); }
```

### Extern Template

```cpp
template <typename T> __global__ void kernel(T x) {}

extern template __global__ void kernel(long x);

int main() {
  kernel<<<1, 1, 0, 0>>>(10); // will create a template specialization
  // kernel<<<1,1,0,0>>>(10l);  // looks for existing kernel<long>, causing
  // linking to fail
}
```

### `nullptr` as a Keyword in Device Compiler

```cpp
__global__ void kernel() {
  int *ptr = nullptr;
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### Strongly Typed Enums

```cpp
enum class EnumVals { Red, Blue, Green };
__global__ void kernel() {
  auto val = EnumVals::Red;
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### Standardized Attribute Syntax

```cpp
[[deprecated]] __global__ void kernel() {
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### `constexpr`

```cpp
struct S {
  constexpr __device__ S(double v) : val(v) {}
  constexpr __device__ double value() const { return val; }

private:
  double val;
};

constexpr __device__ int factorial(int n) {
  return n <= 1 ? 1 : (n * factorial(n - 1));
}

__global__ void kernel() {
  constexpr S s(factorial(5));
  constexpr double d = s.value();
  // ...
}
int main() { kernel<<<1, 1>>>(); }
```

### `alignas` with Struct

```cpp
struct alignas(alignof(int)) S {
  //...
};

__global__ void kernel() {
  S s;
  static_assert(alignof(S) == alignof(int), "they have the same alignment");
  // check the alignment
}
int main() { kernel<<<1, 1>>>(); }
```

### Delegating Constructors

```cpp
struct S {
private:
  int val;

public:
  __device__ S(int v) : val(v) {}
  __device__ S() : S(42) {}
};
__global__ void kernel() { S s{}; }
int main() { kernel<<<1, 1>>>(); }
```

### Explicit Conversion Functions

```cpp
struct S {
private:
  int val;

public:
  __device__ S(int val) : val(val) {}
  __device__ explicit operator int *() { return &val; }
};

__global__ void kernel() {
  S s{0};
  // if (s) { // compile error
  // without the explicit function specifier then s would be converted to the
  // pointer to s.val, which would be non-zero so always true.
  //}
  if ((int *)(s)) {
    // this compiles but is likely not what the user intended
  }
}
int main() { kernel<<<1, 1>>>(); }
```

### Unicode Character Types, Unicode String, Universal Character Literal

```cpp
__global__ void kernel() {
  // cant print it since printf(gpu) doesnot support unicode char arguments
  char16_t a = u'y';
  char32_t l = U'猫';
  auto *string = U"इस अनुवाद को करने से आपको क्या मिला?";
}
int main() { kernel<<<1, 1>>>(); }
```

### User Defined Literals

```cpp
__device__ long double operator"" _w(long double a) { return a; }
__device__ unsigned operator"" _w(char const *c) { return *c - '0'; }

__global__ void kernel() {
  auto ld = 1.2_w; // calls operator "" _w(1.2L)
  auto val = 2_w;  // calls operator "" _w("2")
}
int main() { kernel<<<1, 1>>>(); }
```

### `default`/`delete` Functions

```cpp
struct S {
  __device__ S() = default;
  __device__ S &operator=(const S &) = delete;
};
__global__ void kernel() {
  S s, other; // fine
  // other = s; // compile error, function deleted
}
int main() { kernel<<<1, 1>>>(); }
```

### Friend Declaration

```cpp
struct Y {};

struct A {
  __device__ A() = default;
  friend Y;
  // friend Z; // compile error since class or struct Z doesn't exist
  friend class Z;        // this is fine
  friend void asdf(int); // functions can be declared without a definition
};
__global__ void kernel() { A a; }
int main() { kernel<<<1, 1>>>(); }
```

### Extended `sizeof`

```cpp
template <typename... Ts> __global__ void kernel(Ts... ts) {
  auto size = sizeof...(ts);
  // ...
}
int main() { kernel<<<1, 1>>>(); }
```

### Unrestricted Unions

```cpp
struct Point {
  __device__ Point() {}
  __device__ Point(int x, int y) : x_(x), y_(y) {}
  int x_, y_;
};

union U {
  int z;
  double w;
  Point p;
  __device__ U() {}
  __device__ U(const Point &pt) : p(pt) {}
  __device__ U &operator=(const Point &pt) {
    new (&p) Point(pt);
    return *this;
  }
};

__global__ void kernel() {
  U u;
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### Inline Namespaces

```cpp
namespace XX {
inline namespace YY {
struct Y {
  int x;
};
} // namespace YY
struct X {
  int a;
};
} // namespace XX

__global__ void kernel() {
  XX::X x{};
  XX::Y y{};
}

int main() { kernel<<<1, 1>>>(); }
```

### Range Based For-loop

```cpp
__global__ void kernel() {
  for (auto &x : {1, 2, 3, 4, 5}) {
    // ...
  }
}
int main() { kernel<<<1, 1>>>(); }
```

### `override` Specifier

```cpp
struct Base {
  int n;
  __device__ Base(int v) : n(v + 1) {}
  __device__ Base() : Base(10) {}
  __device__ virtual ~Base() {}
  __device__ virtual int get() { return n; }
};

struct Derived : public Base {
  int n;
  __device__ Derived(int v) : n(v) {}
  __device__ int get() override { return n; }
  __device__ ~Derived() {}
};

__global__ void kernel() {
  Derived d(10);
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### `noexcept` Keyword

```cpp
__global__ void kernel() noexcept {
  int n;
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### Consecutive Right Angle Brackets in Templates

```cpp
template <typename T> struct A { T a; };

template <typename T> struct B { T b; };

__global__ void kernel() {
  A<B<int>> ab;
  //...
}
int main() { kernel<<<1, 1>>>(); }
```

### Not Yet Documented 

* Right Angled Brackets : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html
* Initializer List :  http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm
* Solving SFINAE problem for expression :  http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html
* Forward Declaration of Enum : http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206 http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf
* Conditionally supported behavior : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf
* Inheriting Constructors : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm
* Standard layout types : https://en.cppreference.com/w/cpp/named_req/StandardLayoutType
* Local and unnamed types as template arguments : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm
* Minimal support for Garbage Collection : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm
* Move special functions : https://en.cppreference.com/w/cpp/language/rule_of_three
* long long int


## C++14

### Not Yet Documented

* Tweak C++ contextual conversions : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf
* Binary literals : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf
* Functions to deduce return type : https://isocpp.org/files/papers/N3638.html
* Lambda capture changes : https://isocpp.org/files/papers/N3648.html
* Polymorphic lambda : https://isocpp.org/files/papers/N3649.html
* Variable template : https://en.cppreference.com/w/cpp/language/variable_template
* constexpr changes : https://isocpp.org/files/papers/N3652.html
* struct member initializer : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html
* clarifying mem allocation : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html
* sized dealloc : https://isocpp.org/files/papers/n3778.html
* deprecated attribute : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html
* digit separator : http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf