from python import Python
from tensor import Tensor, TensorSpec, TensorShape
from collections.vector import InlinedFixedVector
from matrix import Matrix, matmul,type, expit_vec, mutate, feed_forward, nelts
import benchmark


struct NeuralNetwork:
    var data: List[Matrix]
    
    fn __init__(inout self, *layers: Tuple[Int, Int]):
        self.data = List[Matrix]()
        for i in range(len(layers)):
            self.data.append(Matrix.rand(layers[i].get[0, Int](), layers[i].get[1, Int]()))

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data^

    fn __copyinit__(inout self, existing: Self):
        self.data = List[Matrix]()
        for i in range(len(existing)):
            var data_copy = existing.data[i]
            self.data.append(data_copy^)

    fn __len__(self) -> Int:
        return len(self.data)

    fn __str__(self) -> String:
        var result: String = ""
        for i in range(len(self)):
            result += str(self.data[i]) + "\n"
        return result

    fn __repr__(self) -> String:
        var result: String = "NeuralNetwork-"
        for i in range(0, len(self), 2):
            result += str(self.data[i].rows)
        return result

    fn feed(self, input_array: Matrix) -> Matrix:
        var result = input_array
        for i in range(len(self)):
            var result = feed_forward(self.data[2*i], result, self.data[2*i+1])
        return result^

    fn mutate(self, strength: Float32) -> Self:
        var result = self
        for i in range(len(result)):
            var result = mutate(result.data[i], strength)

        return result^

fn main() raises:
    
    
    @parameter
    fn test_fn():
        var neural = NeuralNetwork((20, 12), (20, 1), (20, 20), (20, 1), (4, 20), (4, 1))
        var inputs = Matrix.rand(12, 1)
        _ = neural.feed(inputs)
        inputs.data.free()
        
    var secs = benchmark.run[test_fn]().mean()
    print("Time:", secs)