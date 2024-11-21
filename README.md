# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1
Diagnostics output:

        MAP
        
        ================================================================================
        Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (163)  
        ================================================================================


        Parallel loop listing for  Function tensor_map.<locals>._map, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (163) 
        -----------------------------------------------------------------------------------------------------------------|loop #ID
        def _map(                                                                                                    | 
                out: Storage,                                                                                            | 
                out_shape: Shape,                                                                                        | 
                out_strides: Strides,                                                                                    | 
                in_storage: Storage,                                                                                     | 
                in_shape: Shape,                                                                                         | 
                in_strides: Strides,                                                                                     | 
        ) -> None:                                                                                                   | 
                # TODO: Implement for Task 3.1.                                                                          | 
                """Optimized tensor_map function using Numba.                                                            | 
                                                                                                                        | 
                Optimizations:                                                                                           | 
                1. **Stride Alignment Check**: If `out` and `in_storage` are stride-aligned (same shape and strides),    | 
                they are treated as 1D arrays for direct element-wise mapping, reducing indexing overhead.               | 
                                                                                                                        | 
                2. **Parallel Execution**: Uses Numba’s `prange` for parallel processing, improving performance          | 
                for large tensors in both aligned and non-aligned cases.                                                 | 
                                                                                                                        | 
                3. **Efficient Indexing for Non-Aligned Tensors**: For non-aligned tensors, multi-dimensional            | 
                indexing with `to_index`, `broadcast_index`, and `index_to_position` maintains compatibility             | 
                across shapes, while benefiting from Numba's optimizations.                                              | 
                """                                                                                                      | 
                in_index = np.zeros(len(in_shape), dtype=np.int32) ------------------------------------------------------| #0
                out_index = np.zeros(len(out_shape), dtype=np.int32)-----------------------------------------------------| #1
                aligned = np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape)                | 
                if aligned:                                                                                              | 
                for i in prange(len(out)):---------------------------------------------------------------------------| #2
                        out[i] = fn(in_storage[i])                                                                       | 
                else:                                                                                                    | 
                for ordinal in prange(len(out)):---------------------------------------------------------------------| #3
                        to_index(ordinal, out_shape, out_index)                                                          | 
                        broadcast_index(out_index, out_shape, in_shape, in_index)                                        | 
                        in_position = index_to_position(in_index, in_strides)                                            | 
                        out_position = index_to_position(out_index, out_strides)                                         | 
                        out[out_position] = fn(in_storage[in_position])                                                  | 
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 4 parallel for-
        loop(s) (originating from loops labelled: #0, #1, #2, #3).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------
        
        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None
        ZIP
        
        ================================================================================
        Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (224)  
        ================================================================================


        Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (224) 
        ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
        def _zip(                                                                                                                                                                    | 
                out: Storage,                                                                                                                                                            | 
                out_shape: Shape,                                                                                                                                                        | 
                out_strides: Strides,                                                                                                                                                    | 
                a_storage: Storage,                                                                                                                                                      | 
                a_shape: Shape,                                                                                                                                                          | 
                a_strides: Strides,                                                                                                                                                      | 
                b_storage: Storage,                                                                                                                                                      | 
                b_shape: Shape,                                                                                                                                                          | 
                b_strides: Strides,                                                                                                                                                      | 
        ) -> None:                                                                                                                                                                   | 
                # TODO: Implement for Task 3.1.                                                                                                                                          | 
                a_index = np.zeros(len(a_shape), dtype=np.int32)-------------------------------------------------------------------------------------------------------------------------| #4
                b_index = np.zeros(len(b_shape), dtype=np.int32)-------------------------------------------------------------------------------------------------------------------------| #5
                out_index = np.zeros(len(out_shape), dtype=np.int32)---------------------------------------------------------------------------------------------------------------------| #6
                aligned = np.array_equal(out_strides, a_strides) and np.array_equal(a_strides, b_strides) and np.array_equal(out_shape, a_shape) and np.array_equal(a_shape, b_shape)    | 
                if aligned:                                                                                                                                                              | 
                for i in prange(len(out)):-------------------------------------------------------------------------------------------------------------------------------------------| #7
                        out[i] = fn(a_storage[i], b_storage[i])                                                                                                                          | 
                else:                                                                                                                                                                    | 
                for ordinal in prange(len(out)):-------------------------------------------------------------------------------------------------------------------------------------| #8
                        to_index(ordinal, out_shape, out_index)                                                                                                                          | 
                        broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                          | 
                        broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                          | 
                        a_position = index_to_position(a_index, a_strides)                                                                                                               | 
                        b_position = index_to_position(b_index, b_strides)                                                                                                               | 
                        out_position = index_to_position(out_index, out_strides)                                                                                                         | 
                        out[out_position] = fn(a_storage[a_position], b_storage[b_position])                                                                                             | 
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 5 parallel for-
        loop(s) (originating from loops labelled: #4, #5, #6, #7, #8).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------
        
        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None
        REDUCE
        
        ================================================================================
        Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (277)  
        ================================================================================


        Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (277) 
        -----------------------------------------------------------------------------------------------------|loop #ID
        def _reduce(                                                                                     | 
                out: Storage,                                                                                | 
                out_shape: Shape,                                                                            | 
                out_strides: Strides,                                                                        | 
                a_storage: Storage,                                                                          | 
                a_shape: Shape,                                                                              | 
                a_strides: Strides,                                                                          | 
                reduce_dim: int,                                                                             | 
        ) -> None:                                                                                       | 
                # TODO: Implement for Task 3.1.                                                              | 
                """Optimized tensor_reduce function using Numba.                                             | 
                                                                                                        | 
                Optimizations:                                                                               | 
                1. **Parallel Main Loop**: The outer loop uses `prange` for parallel execution.              | 
                2. **Numpy Buffers for Indices**: Efficiently handles indexing with pre-computed strides.    | 
                3. **Position Increment in Inner Loop**: Avoids repeated `index_to_position` calls by        | 
                incrementing `a_position` directly along the reduction dimension.                            | 
                """                                                                                          | 
                a_index = np.zeros(len(a_shape), dtype=np.int32)---------------------------------------------| #9
                out_index = np.zeros(len(out_shape), dtype=np.int32)-----------------------------------------| #10
                                                                                                        | 
                reduce_stride = a_strides[reduce_dim]                                                        | 
                reduce_size = a_shape[reduce_dim]                                                            | 
                for ordinal in prange(len(out)):-------------------------------------------------------------| #11
                to_index(ordinal, out_shape, out_index)                                                  | 
                for i in range(len(out_shape)):                                                          | 
                        a_index[i] = out_index[i]                                                            | 
                a_index[reduce_dim] = 0                                                                  | 
                a_position = index_to_position(a_index, a_strides)                                       | 
                result = a_storage[a_position]                                                           | 
                for i in range(1, reduce_size):                                                          | 
                        a_position += reduce_stride                                                          | 
                        result = fn(result, a_storage[int(a_position)])                                      | 
                out[ordinal] = result                                                                    | 
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 3 parallel for-
        loop(s) (originating from loops labelled: #9, #10, #11).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------
        
        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None
## Task 3.2
Diagnostics output:
       
        MATRIX MULTIPLY
        
        ================================================================================
        Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (315)  
        ================================================================================


        Parallel loop listing for  Function _tensor_matrix_multiply, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (315) 
        -----------------------------------------------------------------------|loop #ID
        def _tensor_matrix_multiply(                                           | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
        ) -> None:                                                             | 
        """NUMBA tensor matrix multiply function.                          | 
                                                                        | 
        Should work for any tensor shapes that broadcast as long as        | 
                                                                        | 
        ```                                                                | 
        assert a_shape[-1] == b_shape[-2]                                  | 
        ```                                                                | 
                                                                        | 
        Optimizations:                                                     | 
                                                                        | 
        * Outer loop in parallel                                           | 
        * No index buffers or function calls                               | 
        * Inner loop should have no global writes, 1 multiply.             | 
                                                                        | 
                                                                        | 
        Args:                                                              | 
        ----                                                               | 
                out (Storage): storage for `out` tensor                        | 
                out_shape (Shape): shape for `out` tensor                      | 
                out_strides (Strides): strides for `out` tensor                | 
                a_storage (Storage): storage for `a` tensor                    | 
                a_shape (Shape): shape for `a` tensor                          | 
                a_strides (Strides): strides for `a` tensor                    | 
                b_storage (Storage): storage for `b` tensor                    | 
                b_shape (Shape): shape for `b` tensor                          | 
                b_strides (Strides): strides for `b` tensor                    | 
                                                                        | 
        Returns:                                                           | 
        -------                                                            | 
                None : Fills in `out`                                          | 
                                                                        | 
        """                                                                | 
        # TODO: Implement for Task 3.2.                                    | 
        a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0             | 
        b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0             | 
        out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0       | 
                                                                        | 
        # Extract dimensions                                               | 
        batch_size = out_shape[0]                                          | 
        out_rows = out_shape[1]                                            | 
        out_cols = out_shape[2]                                            | 
        inner_dim = a_shape[-1]                                            | 
                                                                        | 
        # Parallelize outer loop over batches and rows                     | 
        for batch in prange(batch_size):-----------------------------------| #12
                a_batch_offset = batch * a_batch_stride                        | 
                b_batch_offset = batch * b_batch_stride                        | 
                out_batch_offset = batch * out_batch_stride                    | 
                                                                        | 
                for i in range(out_rows):                                      | 
                a_row_offset = a_batch_offset + i * a_strides[1]           | 
                out_row_offset = out_batch_offset + i * out_strides[1]     | 
                for j in range(out_cols):                                  | 
                        result = 0.0                                           | 
                        b_column_offset = b_batch_offset + j * b_strides[2]    | 
                        for k in range(inner_dim):                             | 
                        a_idx = a_row_offset + k * a_strides[2]            | 
                        b_idx = k * b_strides[1] + b_column_offset         | 
                        result += a_storage[a_idx] * b_storage[b_idx]      | 
                                                                        | 
                        out_idx = out_row_offset + j * out_strides[2]          | 
                        out[out_idx] = result                                  | 
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 1 parallel for-
        loop(s) (originating from loops labelled: #12).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------
        
        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None
## Task 3.5
### Split
#### CPU
#### GPU
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

```
Epoch 0, Loss: 7.067772703878541, Correct: 29, Time per epoch: 6.319 seconds
Epoch 10, Loss: 7.2926700049842585, Correct: 36, Time per epoch: 1.286 seconds
Epoch 20, Loss: 7.084129401759604, Correct: 37, Time per epoch: 1.188 seconds
Epoch 30, Loss: 5.612738472199355, Correct: 39, Time per epoch: 1.206 seconds
Epoch 40, Loss: 3.490015741391367, Correct: 46, Time per epoch: 1.251 seconds
Epoch 50, Loss: 4.422211986613442, Correct: 44, Time per epoch: 1.249 seconds
Epoch 60, Loss: 4.662944251875238, Correct: 42, Time per epoch: 1.183 seconds
Epoch 70, Loss: 2.2640910062884125, Correct: 46, Time per epoch: 1.212 seconds
Epoch 80, Loss: 2.438541405279595, Correct: 49, Time per epoch: 1.195 seconds
Epoch 90, Loss: 1.4078174683022526, Correct: 49, Time per epoch: 1.195 seconds
Epoch 100, Loss: 2.1541947551519565, Correct: 47, Time per epoch: 1.189 seconds
Epoch 110, Loss: 1.3060080447621671, Correct: 50, Time per epoch: 1.225 seconds
Epoch 120, Loss: 1.063000014217251, Correct: 49, Time per epoch: 1.202 seconds
Epoch 130, Loss: 0.8039157548650032, Correct: 49, Time per epoch: 1.270 seconds
Epoch 140, Loss: 0.41892217457352426, Correct: 50, Time per epoch: 1.185 seconds
Epoch 150, Loss: 2.3043434223143127, Correct: 50, Time per epoch: 1.203 seconds
Epoch 160, Loss: 1.1124970187666758, Correct: 50, Time per epoch: 1.232 seconds
Epoch 170, Loss: 0.9298985336196348, Correct: 50, Time per epoch: 1.517 seconds
Epoch 180, Loss: 0.9929461045450704, Correct: 49, Time per epoch: 1.562 seconds
Epoch 190, Loss: 1.2616782642694135, Correct: 49, Time per epoch: 1.762 seconds
Epoch 200, Loss: 0.7844707612712554, Correct: 50, Time per epoch: 1.732 seconds
Epoch 210, Loss: 0.6384536121555635, Correct: 50, Time per epoch: 1.541 seconds
Epoch 220, Loss: 0.33261134381848356, Correct: 48, Time per epoch: 1.193 seconds
Epoch 230, Loss: 0.7631318578759074, Correct: 50, Time per epoch: 1.268 seconds
Epoch 240, Loss: 0.6198236040705906, Correct: 50, Time per epoch: 1.180 seconds
Epoch 250, Loss: 1.5108712446497898, Correct: 49, Time per epoch: 1.193 seconds
Epoch 260, Loss: 0.969339278419332, Correct: 50, Time per epoch: 1.184 seconds
Epoch 270, Loss: 1.3933164441072294, Correct: 49, Time per epoch: 1.178 seconds
Epoch 280, Loss: 1.160750282542737, Correct: 50, Time per epoch: 1.181 seconds
Epoch 290, Loss: 1.1177012823558286, Correct: 50, Time per epoch: 1.181 seconds
Epoch 300, Loss: 1.841222021312694, Correct: 49, Time per epoch: 1.200 seconds
Epoch 310, Loss: 0.9310236613278304, Correct: 50, Time per epoch: 1.193 seconds
Epoch 320, Loss: 0.5199611110294511, Correct: 49, Time per epoch: 1.192 seconds
Epoch 330, Loss: 0.6129734637820158, Correct: 49, Time per epoch: 1.198 seconds
Epoch 340, Loss: 1.0081116463281883, Correct: 50, Time per epoch: 1.199 seconds
Epoch 350, Loss: 0.6182741281846832, Correct: 49, Time per epoch: 1.197 seconds
Epoch 360, Loss: 0.6559963958028397, Correct: 50, Time per epoch: 1.175 seconds
Epoch 370, Loss: 0.04885996490561376, Correct: 49, Time per epoch: 1.191 seconds
Epoch 380, Loss: 1.917830523652118, Correct: 49, Time per epoch: 1.219 seconds
Epoch 390, Loss: 0.4543201913522107, Correct: 50, Time per epoch: 1.184 seconds
Epoch 400, Loss: 1.5621380269886684, Correct: 49, Time per epoch: 1.185 seconds
Epoch 410, Loss: 0.3011307650305586, Correct: 49, Time per epoch: 1.210 seconds
Epoch 420, Loss: 0.8592109314575497, Correct: 50, Time per epoch: 1.180 seconds
Epoch 430, Loss: 1.4490130695922874, Correct: 49, Time per epoch: 1.315 seconds
Epoch 440, Loss: 1.8092770889441656, Correct: 49, Time per epoch: 1.430 seconds
Epoch 450, Loss: 1.092658029371688, Correct: 48, Time per epoch: 1.615 seconds
Epoch 460, Loss: 0.02005439233309631, Correct: 50, Time per epoch: 1.799 seconds
Epoch 470, Loss: 0.27518018539961664, Correct: 49, Time per epoch: 1.833 seconds
Epoch 480, Loss: 1.1804852719071184, Correct: 49, Time per epoch: 1.621 seconds
Epoch 490, Loss: 0.6614519753953101, Correct: 50, Time per epoch: 1.338 seconds
```