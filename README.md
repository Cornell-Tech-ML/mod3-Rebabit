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
                aligned = np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape)                | 
                size = int(np.prod(out_shape))---------------------------------------------------------------------------| #3
                if aligned:                                                                                              | 
                for i in prange(len(out)):---------------------------------------------------------------------------| #2
                        out[i] = fn(in_storage[i])                                                                       | 
                else:                                                                                                    | 
                in_index = np.zeros(len(in_shape), dtype=np.int32) --------------------------------------------------| #0
                out_index = np.zeros(len(out_shape), dtype=np.int32)-------------------------------------------------| #1
                for ordinal in prange(size):-------------------------------------------------------------------------| #4
                        to_index(ordinal, out_shape, out_index)                                                          | 
                        broadcast_index(out_index, out_shape, in_shape, in_index)                                        | 
                        in_position = index_to_position(in_index, in_strides)                                            | 
                        out_position = index_to_position(out_index, out_strides)                                         | 
                        out[out_position] = fn(in_storage[in_position])                                                  | 
                # in_index = np.zeros(len(in_shape), dtype=np.int32)                                                     | 
                # out_index = np.zeros(len(out_shape), dtype=np.int32)                                                   | 
                # for i in range(len(out)):                                                                              | 
                #     to_index(i, out_shape, out_index)                                                                  | 
                #     broadcast_index(out_index, out_shape, in_shape, in_index)                                          | 
                #     o = index_to_position(out_index, out_strides)                                                      | 
                #     j = index_to_position(in_index, in_strides)                                                        | 
                #     out[o] = fn(in_storage[j])                                                                         | 
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 5 parallel for-
        loop(s) (originating from loops labelled: #3, #2, #0, #1, #4).
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
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (233)  
        ================================================================================


        Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (233) 
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
                aligned = np.array_equal(out_strides, a_strides) and np.array_equal(a_strides, b_strides) and np.array_equal(out_shape, a_shape) and np.array_equal(a_shape, b_shape)    | 
                size = int(np.prod(out_shape))-------------------------------------------------------------------------------------------------------------------------------------------| #8
                if aligned:                                                                                                                                                              | 
                for i in prange(len(out)):-------------------------------------------------------------------------------------------------------------------------------------------| #9
                        out[i] = fn(a_storage[i], b_storage[i])                                                                                                                          | 
                else:                                                                                                                                                                    | 
                a_index = np.zeros(len(a_shape), dtype=np.int32)---------------------------------------------------------------------------------------------------------------------| #5
                b_index = np.zeros(len(b_shape), dtype=np.int32)---------------------------------------------------------------------------------------------------------------------| #6
                out_index = np.zeros(len(out_shape), dtype=np.int32)-----------------------------------------------------------------------------------------------------------------| #7
                for ordinal in prange(size):-----------------------------------------------------------------------------------------------------------------------------------------| #10
                        to_index(ordinal, out_shape, out_index)                                                                                                                          | 
                        broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                          | 
                        broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                          | 
                        a_position = index_to_position(a_index, a_strides)                                                                                                               | 
                        b_position = index_to_position(b_index, b_strides)                                                                                                               | 
                        out_position = index_to_position(out_index, out_strides)                                                                                                         | 
                        out[out_position] = fn(a_storage[a_position], b_storage[b_position])                                                                                             | 
                # out_index: Index = np.zeros(MAX_DIMS, np.int32)                                                                                                                        | 
                # a_index: Index = np.zeros(MAX_DIMS, np.int32)                                                                                                                          | 
                # b_index: Index = np.zeros(MAX_DIMS, np.int32)                                                                                                                          | 
                # for i in range(len(out)):                                                                                                                                              | 
                #     to_index(i, out_shape, out_index)                                                                                                                                  | 
                #     o = index_to_position(out_index, out_strides)                                                                                                                      | 
                #     broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                            | 
                #     j = index_to_position(a_index, a_strides)                                                                                                                          | 
                #     broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                            | 
                #     k = index_to_position(b_index, b_strides)                                                                                                                          | 
                #     out[o] = fn(a_storage[j], b_storage[k])                                                                                                                            | 
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 6 parallel for-
        loop(s) (originating from loops labelled: #8, #9, #5, #6, #7, #10).
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
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (297)  
        ================================================================================


        Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (297) 
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
                a_index = np.zeros(len(a_shape), dtype=np.int32)---------------------------------------------| #11
                out_index = np.zeros(len(out_shape), dtype=np.int32)-----------------------------------------| #12
                reduce_stride = a_strides[reduce_dim]                                                        | 
                reduce_size = a_shape[reduce_dim]                                                            | 
                size = 1                                                                                     | 
                for dim in out_shape:                                                                        | 
                size *= dim                                                                              | 
                for ordinal in prange(size):-----------------------------------------------------------------| #13
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
        loop(s) (originating from loops labelled: #11, #12, #13).
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
        /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (337)  
        ================================================================================


        Parallel loop listing for  Function _tensor_matrix_multiply, /Users/yaxuan/Documents/GitHub/mod3-Rebabit/minitorch/fast_ops.py (337) 
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
        for batch in prange(batch_size):-----------------------------------| #14
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
        loop(s) (originating from loops labelled: #14).
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