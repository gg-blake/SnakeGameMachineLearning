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
alias N = 7  # cols of B and C
alias K = 5  # cols of A and rows of B
alias type = DType.float32

# simdwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simdwidthof[type]() * 2


trait MatrixStruct:
    #var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):...

    fn __str__(self) -> String:...

    fn __moveinit__(inout self, owned existing: Self):...

    fn __add__(self, other: Self) -> Self:...

    fn __mul__(self, other: Self) -> Self:...

    fn to_tensor(self) -> Tensor[type]:...

    fn to_array(self) raises -> PythonObject:...

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:...

    fn __getitem__(borrowed self, y: Int, x: Int) -> Scalar[type]:...

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):...

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:...

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):...

struct Matrix(CollectionElement):
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize zeroeing all values
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        self.rows = rows
        self.cols = cols
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, rows: Int, cols: Int, owned data: DTypePointer[type]):
        self.data = data
        self.rows = rows
        self.cols = cols

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data
        self.rows = existing.rows
        self.cols = existing.cols

    fn __copyinit__(inout self, borrowed existing: Self):
        self.data = DTypePointer[type].alloc(existing.rows * existing.cols)
        memset_zero(self.data, existing.rows * existing.cols)
        self.rows = existing.rows
        self.cols = existing.cols
        for i in range(existing.rows):
            for j in range(existing.cols):
                self[i, j] = existing[i, j]

    fn __str__(self) -> String:
        var content: String = "["
        for i in range(self.rows):
            var substring: String = "["
            for j in range(self.cols):
                substring += str(self[i, j])
                if j != self.rows - 1 and self.cols != 1:
                    substring += ", "
            substring += "]"
            if i != self.rows - 1:
                substring += ",\n"
            content += substring
        content += "], shape=" + str(self.rows) + "x" + str(self.cols) + "]"
        return content^

    fn __add__(self, other: Self) -> Self:
        var result = Self(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                result[r, c] = self[r, c] + other[r, c]

        return result^

    fn __add__(self, val: Scalar[type]) -> Self:
        var result = Self(self.rows, self.cols)
        var vec = self.load[nelts](0, 0)
        result.store[nelts](0, 0, vec + val)
        return result^

    fn __sub__(self, val: Scalar[type]) -> Self:
        var result = Self(self.rows, self.cols)
        var vec = self.load[nelts](0, 0)
        result.store[nelts](0, 0, vec - val)
        return result^

    fn __mul__(self, other: Self) -> Self:
        var result = Self(self.rows, self.cols)
        var vec_a = self.load[nelts](0, 0)
        var vec_b = other.load[nelts](0, 0)
        result.store[nelts](0, 0, vec_a * vec_b)
        return result^

    fn __mul__(self, val: Scalar[type]) -> Self:
        var result = Self(self.rows, self.cols)
        var vec = self.load[nelts](0, 0)
        result.store[nelts](0, 0, vec * val)
        return result^

    # Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(borrowed self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

fn matmul[N: Int](A: Matrix, B: Matrix) -> Matrix:
    # Note: N must be equal to B's # of columns
    var C = Matrix(A.rows, B.cols)
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = N]()

    return C^
    

fn benchmark_matmul():
    var A = Matrix.rand(M, K)
    var B = Matrix.rand(K, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = matmul[N](A, B)

    var secs = benchmark.run[test_fn]().mean()

    A.data.free()
    B.data.free()

    print("Mojo: ", str(secs))

fn benchmark_matmul_py() raises:
    var np = Python.import_module("numpy")
    

    @always_inline
    @parameter
    fn test_fn():
        try:
            var A = np.random.rand(M, K)
            var B = np.random.rand(K, N)
            _ = np.matmul(A, B)
        except:
            pass

    var secs = benchmark.run[test_fn]().mean()

    print("Python: ", str(secs))

fn expit_vec(A: Matrix) -> Matrix:
    var B = Matrix(A.rows, A.cols)
    var x = A.load[nelts](0, 0)
    var result = 1 / (1 + exp[type, nelts](-x))
    B.store[nelts](0, 0, result)
    return B^

fn expit(A: Matrix) -> Matrix:
    var B = Matrix(A.rows, A.cols)
    for r in range(A.rows):
        for c in range(A.cols):
            B[r, c] = 1 / (1 + exp[type, 1](-A[r, c]))
    return B^

fn expit_py[M: Int, N: Int](A: PythonObject) -> PythonObject:
    try:
        var np = Python.import_module("numpy")
        var B = np.zeros(M, N)
        B = 1 / (1 + np.exp(-A))
        return B^
    except:
        return Python.none()

fn mutate(A: Matrix, strength: Scalar[type]) -> Matrix:
    return A + Matrix.rand(A.rows, A.cols) * strength

fn feed_forward(A: Matrix, B: Matrix, C: Matrix) -> Matrix:
    return expit(matmul[1](A, B) + C)

fn main() raises:
    var A = Matrix.rand(4, 5)
    var B = Matrix.rand(5, 1)
    var C = Matrix.rand(5, 1)
    
    @parameter
    @always_inline
    fn test_fn():
        var D = feed_forward(A, B, C)

    var secs = benchmark.run[test_fn]().mean()
    print("Time:", secs)
    


    