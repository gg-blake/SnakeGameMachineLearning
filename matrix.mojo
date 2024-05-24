from algorithm import Static2DTileUnitFunc as Tile2DFunc
from random import rand
from algorithm import parallelize, vectorize
import benchmark
from time import sleep
from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from python import Python
from math import exp

alias M = 3  # rows of A and C
alias N = 4  # cols of B and C
alias K = 5  # cols of A and rows of B
alias type = DType.float32

# simdwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simdwidthof[type]() * 2


trait MatrixStruct:
    #var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):...

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, owned data: DTypePointer[type]):...

    fn __str__(self) -> String:...

    fn __moveinit__(inout self, owned existing: Self):...

    fn __add__(self, other: Self) -> Self:...

    fn to_tensor(self) -> Tensor[type]:...

    fn to_array(self) raises -> PythonObject:...
    # Initialize with random values
    @staticmethod
    fn rand() -> Self:...

    fn __getitem__(borrowed self, y: Int, x: Int) -> Scalar[type]:...

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):...

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:...

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):...


struct Matrix[rows: Int, cols: Int](MatrixStruct):
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, owned data: DTypePointer[type]):
        self.data = data

    fn __str__(self) -> String:
        return str(self.to_tensor())

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data

    fn __add__(self, other: Self) -> Self:
        var result = Self()
        for r in range(rows):
            for c in range(cols):
                result[r, c] = self[r, c] + other[r, c]

        return result^

    fn to_tensor(self) -> Tensor[type]:
        var t = Tensor[type](TensorShape(rows, cols))
        for r in range(rows):
            for c in range(cols):
                var i = Index(r, c)
                t[i] = self[r, c]
        
        return t

    fn to_array(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var arr = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                var val = self[r, c]
                arr[r][c] = val

        return arr^

    

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(borrowed self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

fn matmul[M: Int, K: Int, N: Int](A: Matrix[M, K], B: Matrix[K, N]) -> Matrix[M, N]:
    var C = Matrix[M, N]()
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.cols]()

    return C^

fn benchmark_matmul():
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    @always_inline
    @parameter
    fn test_fn():
        _ = matmul[M, K, N](A, B)

    var secs = benchmark.run[test_fn]().mean()

    A.data.free()
    B.data.free()

    print("Mojo: ", str(secs))

fn benchmark_matmul_py() raises:
    var np = Python.import_module("numpy")
    var A = np.random.rand(M, K)
    var B = np.random.rand(K, N)

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = np.matmul(A, B)
        except:
            pass

    var secs = benchmark.run[test_fn]().mean()

    print("Python: ", str(secs))

fn expit_vec(A: Matrix) -> Matrix[A.rows, A.cols]:
    var B = Matrix[A.rows, A.cols]()
    var x = A.load[nelts](0, 0)
    var result = 1 / (1 + exp[type, nelts](-x))
    B.store[nelts](0, 0, result)
    return B^

fn expit(A: Matrix) -> Matrix[A.rows, A.cols]:
    var B = Matrix[A.rows, A.cols]()
    for r in range(A.rows):
        for c in range(A.cols):
            B[r, c] = 1 / (1 + exp[type, 1](-A[r, c]))
    return B^

fn expit_py[M: Int, N: Int](A: PythonObject) -> PythonObject:
    try:
        var np = Python.import_module("numpy")
        var B = np.zeros(M, N)
        B = 1 / (1 + np.exp(-A))
        return B
    except:
        return Python.none()





fn main() raises:
    var A = Matrix[M, N].rand()
    var B = Matrix[M, N]()
    var C = Matrix[M, N]()

    @parameter
    fn list[x: VariadicList[Int], i: Int]():
        if i+1 == len(x) - 1:
            return VariadicList[MatrixStruct](Matrix[x[i], x[i+1]])

        return VariadicList[MatrixStruct](Matrix[x[i], x[i+1]], list[x, i+1]())
    
    alias matricies = list[VariadicList[Int](2, 3, 4, 5), 0]()