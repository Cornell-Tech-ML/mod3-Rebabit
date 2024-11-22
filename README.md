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
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
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
### XOR
#### CPU
#### GPU
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```
Epoch 0, Loss: 7.375727233876458, Correct: 37, Time per epoch: 3.467 seconds
Epoch 10, Loss: 6.078468495745869, Correct: 42, Time per epoch: 1.290 seconds
Epoch 20, Loss: 5.0646373228255035, Correct: 43, Time per epoch: 1.181 seconds
Epoch 30, Loss: 3.1335867135519146, Correct: 43, Time per epoch: 1.183 seconds
Epoch 40, Loss: 2.3192992878280054, Correct: 44, Time per epoch: 1.177 seconds
Epoch 50, Loss: 2.7021346941655935, Correct: 46, Time per epoch: 1.272 seconds
Epoch 60, Loss: 3.134365626397329, Correct: 46, Time per epoch: 1.193 seconds
Epoch 70, Loss: 4.474306976090135, Correct: 46, Time per epoch: 1.185 seconds
Epoch 80, Loss: 1.7999009029555308, Correct: 46, Time per epoch: 1.207 seconds
Epoch 90, Loss: 2.64376661701622, Correct: 46, Time per epoch: 1.183 seconds
Epoch 100, Loss: 1.0131499484315725, Correct: 46, Time per epoch: 1.192 seconds
Epoch 110, Loss: 3.1973933669039747, Correct: 46, Time per epoch: 1.396 seconds
Epoch 120, Loss: 1.870347276206653, Correct: 46, Time per epoch: 1.562 seconds
Epoch 130, Loss: 2.415895391021435, Correct: 47, Time per epoch: 1.849 seconds
Epoch 140, Loss: 1.0647787499360262, Correct: 47, Time per epoch: 1.430 seconds
Epoch 150, Loss: 1.0085280062022943, Correct: 47, Time per epoch: 1.269 seconds
Epoch 160, Loss: 0.49059614201030655, Correct: 48, Time per epoch: 1.169 seconds
Epoch 170, Loss: 1.193722988113516, Correct: 47, Time per epoch: 1.263 seconds
Epoch 180, Loss: 1.6479213211416721, Correct: 47, Time per epoch: 1.185 seconds
Epoch 190, Loss: 2.447796771151575, Correct: 48, Time per epoch: 1.178 seconds
Epoch 200, Loss: 0.5206323554580065, Correct: 47, Time per epoch: 1.214 seconds
Epoch 210, Loss: 1.3434292908874752, Correct: 48, Time per epoch: 1.195 seconds
Epoch 220, Loss: 1.1153735228366566, Correct: 49, Time per epoch: 1.236 seconds
Epoch 230, Loss: 0.6517211960485579, Correct: 48, Time per epoch: 1.256 seconds
Epoch 240, Loss: 3.1175413765449793, Correct: 47, Time per epoch: 1.170 seconds
Epoch 250, Loss: 1.569436273795383, Correct: 48, Time per epoch: 1.188 seconds
Epoch 260, Loss: 0.7419021828838371, Correct: 47, Time per epoch: 1.173 seconds
Epoch 270, Loss: 1.762097622918224, Correct: 47, Time per epoch: 1.193 seconds
Epoch 280, Loss: 0.2764679168320554, Correct: 48, Time per epoch: 1.192 seconds
Epoch 290, Loss: 2.7496005210389725, Correct: 49, Time per epoch: 1.216 seconds
Epoch 300, Loss: 0.6071833543246169, Correct: 48, Time per epoch: 1.190 seconds
Epoch 310, Loss: 1.0755428139517442, Correct: 48, Time per epoch: 1.184 seconds
Epoch 320, Loss: 0.35697002180825077, Correct: 48, Time per epoch: 1.179 seconds
Epoch 330, Loss: 0.4591876802646595, Correct: 47, Time per epoch: 1.177 seconds
Epoch 340, Loss: 2.29921622042483, Correct: 48, Time per epoch: 1.186 seconds
Epoch 350, Loss: 0.5867516651926417, Correct: 47, Time per epoch: 1.184 seconds
Epoch 360, Loss: 0.7037143919501218, Correct: 48, Time per epoch: 1.253 seconds
Epoch 370, Loss: 0.815886625193198, Correct: 48, Time per epoch: 1.316 seconds
Epoch 380, Loss: 3.3363360630781838, Correct: 48, Time per epoch: 1.460 seconds
Epoch 390, Loss: 2.949316830913541, Correct: 46, Time per epoch: 1.607 seconds
Epoch 400, Loss: 2.3944326448421083, Correct: 48, Time per epoch: 1.752 seconds
Epoch 410, Loss: 0.4414295416735795, Correct: 48, Time per epoch: 1.735 seconds
Epoch 420, Loss: 2.5963895938518693, Correct: 48, Time per epoch: 1.738 seconds
Epoch 430, Loss: 0.7596716377720492, Correct: 48, Time per epoch: 1.640 seconds
Epoch 440, Loss: 0.3570401897132005, Correct: 49, Time per epoch: 1.404 seconds
Epoch 450, Loss: 0.34938688061288814, Correct: 49, Time per epoch: 1.233 seconds
Epoch 460, Loss: 0.3871982243752342, Correct: 49, Time per epoch: 1.467 seconds
Epoch 470, Loss: 3.032922273454699, Correct: 46, Time per epoch: 1.229 seconds
Epoch 480, Loss: 1.8559262294990493, Correct: 48, Time per epoch: 1.214 seconds
Epoch 490, Loss: 1.771049543842385, Correct: 48, Time per epoch: 1.169 seconds
```
### Simple
#### CPU
#### GPU
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```
Epoch 0, Loss: 5.516414809597006, Correct: 43, Time per epoch: 5.370 seconds
Epoch 10, Loss: 3.2840791377364766, Correct: 43, Time per epoch: 1.419 seconds
Epoch 20, Loss: 1.719520819354006, Correct: 48, Time per epoch: 1.192 seconds
Epoch 30, Loss: 1.741085353388282, Correct: 48, Time per epoch: 1.170 seconds
Epoch 40, Loss: 0.8296526387491836, Correct: 48, Time per epoch: 1.188 seconds
Epoch 50, Loss: 0.776976861068333, Correct: 48, Time per epoch: 1.245 seconds
Epoch 60, Loss: 0.3737786210356906, Correct: 49, Time per epoch: 1.164 seconds
Epoch 70, Loss: 0.6907664199543707, Correct: 50, Time per epoch: 1.189 seconds
Epoch 80, Loss: 0.5970645750387666, Correct: 49, Time per epoch: 1.182 seconds
Epoch 90, Loss: 1.2910743530976487, Correct: 49, Time per epoch: 1.155 seconds
Epoch 100, Loss: 0.6824187517501987, Correct: 49, Time per epoch: 1.160 seconds
Epoch 110, Loss: 2.497185992932521, Correct: 47, Time per epoch: 1.183 seconds
Epoch 120, Loss: 0.5403922437544426, Correct: 50, Time per epoch: 1.156 seconds
Epoch 130, Loss: 0.20989983924725047, Correct: 50, Time per epoch: 1.260 seconds
Epoch 140, Loss: 1.4480787853074095, Correct: 49, Time per epoch: 1.169 seconds
Epoch 150, Loss: 0.7400994954415159, Correct: 50, Time per epoch: 1.208 seconds
Epoch 160, Loss: 0.04205827990501291, Correct: 50, Time per epoch: 1.185 seconds
Epoch 170, Loss: 0.8516159252561262, Correct: 49, Time per epoch: 1.235 seconds
Epoch 180, Loss: 0.1981211812175629, Correct: 49, Time per epoch: 1.171 seconds
Epoch 190, Loss: 0.34183129905653736, Correct: 50, Time per epoch: 1.166 seconds
Epoch 200, Loss: 0.21131916247899465, Correct: 50, Time per epoch: 1.156 seconds
Epoch 210, Loss: 0.9831900775459502, Correct: 49, Time per epoch: 1.168 seconds
Epoch 220, Loss: 0.07853105449301111, Correct: 50, Time per epoch: 1.175 seconds
Epoch 230, Loss: 0.3458752774555975, Correct: 50, Time per epoch: 1.823 seconds
Epoch 240, Loss: 0.31041821445218737, Correct: 49, Time per epoch: 1.161 seconds
Epoch 250, Loss: 0.40002873776429815, Correct: 49, Time per epoch: 1.165 seconds
Epoch 260, Loss: 0.013535451999680419, Correct: 49, Time per epoch: 1.164 seconds
Epoch 270, Loss: 0.46422659673289174, Correct: 50, Time per epoch: 1.160 seconds
Epoch 280, Loss: 0.1256354550930416, Correct: 49, Time per epoch: 1.155 seconds
Epoch 290, Loss: 0.012190079347211053, Correct: 49, Time per epoch: 1.192 seconds
Epoch 300, Loss: 0.04249106864351862, Correct: 50, Time per epoch: 1.181 seconds
Epoch 310, Loss: 0.07524304631005484, Correct: 50, Time per epoch: 1.183 seconds
Epoch 320, Loss: 0.2539364260511513, Correct: 49, Time per epoch: 1.182 seconds
Epoch 330, Loss: 0.31064760015468595, Correct: 49, Time per epoch: 1.169 seconds
Epoch 340, Loss: 0.4977676326675397, Correct: 50, Time per epoch: 1.161 seconds
Epoch 350, Loss: 0.05428129472279494, Correct: 50, Time per epoch: 1.170 seconds
Epoch 360, Loss: 0.0010800771820646275, Correct: 49, Time per epoch: 1.290 seconds
Epoch 370, Loss: 0.059895020748592265, Correct: 49, Time per epoch: 1.412 seconds
Epoch 380, Loss: 0.05585965705895073, Correct: 49, Time per epoch: 1.504 seconds
Epoch 390, Loss: 0.08486584613705234, Correct: 50, Time per epoch: 1.614 seconds
Epoch 400, Loss: 0.019778194213418837, Correct: 49, Time per epoch: 1.702 seconds
Epoch 410, Loss: 1.1790824229415242, Correct: 50, Time per epoch: 1.666 seconds
Epoch 420, Loss: 0.018504504990679325, Correct: 50, Time per epoch: 1.630 seconds
Epoch 430, Loss: 0.31317788702632593, Correct: 50, Time per epoch: 1.583 seconds
Epoch 440, Loss: 1.0760362690615017, Correct: 49, Time per epoch: 1.515 seconds
Epoch 450, Loss: 0.03468783202938342, Correct: 50, Time per epoch: 1.413 seconds
Epoch 460, Loss: 0.003279631506889776, Correct: 49, Time per epoch: 1.372 seconds
Epoch 470, Loss: 0.714393135110035, Correct: 49, Time per epoch: 1.367 seconds
Epoch 480, Loss: 0.5032178628839438, Correct: 50, Time per epoch: 1.217 seconds
Epoch 490, Loss: 0.06725681764846303, Correct: 50, Time per epoch: 1.155 seconds
```
### Simple - Larger
#### CPU
#### GPU
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05
```
```
Epoch 0, Loss: 2.842611828139131, Correct: 42, Time per epoch: 3.372 seconds
Epoch 10, Loss: 1.446976690397959, Correct: 47, Time per epoch: 1.379 seconds
Epoch 20, Loss: 0.9729621726889107, Correct: 50, Time per epoch: 1.246 seconds
Epoch 30, Loss: 1.0674065111215396, Correct: 47, Time per epoch: 1.252 seconds
Epoch 40, Loss: 1.956570549330924, Correct: 50, Time per epoch: 1.258 seconds
Epoch 50, Loss: 0.44134685838264687, Correct: 50, Time per epoch: 1.348 seconds
Epoch 60, Loss: 0.21118096031713401, Correct: 49, Time per epoch: 1.820 seconds
Epoch 70, Loss: 0.37301909691062307, Correct: 50, Time per epoch: 1.297 seconds
Epoch 80, Loss: 0.7073115246342402, Correct: 49, Time per epoch: 1.261 seconds
Epoch 90, Loss: 0.13708083988924424, Correct: 50, Time per epoch: 1.271 seconds
Epoch 100, Loss: 0.16716774590273273, Correct: 50, Time per epoch: 1.258 seconds
Epoch 110, Loss: 0.4821472128714319, Correct: 50, Time per epoch: 1.252 seconds
Epoch 120, Loss: 0.14898917455953828, Correct: 50, Time per epoch: 1.257 seconds
Epoch 130, Loss: 0.02650156443060781, Correct: 50, Time per epoch: 1.301 seconds
Epoch 140, Loss: 0.0318175187767811, Correct: 50, Time per epoch: 1.235 seconds
Epoch 150, Loss: 0.35852999520893214, Correct: 50, Time per epoch: 1.238 seconds
Epoch 160, Loss: 0.4928834336314164, Correct: 50, Time per epoch: 1.446 seconds
Epoch 170, Loss: 0.4735952903906252, Correct: 50, Time per epoch: 1.987 seconds
Epoch 180, Loss: 0.07518131230511854, Correct: 50, Time per epoch: 1.577 seconds
Epoch 190, Loss: 0.05829570508891557, Correct: 50, Time per epoch: 1.234 seconds
Epoch 200, Loss: 0.5453004636655746, Correct: 50, Time per epoch: 1.267 seconds
Epoch 210, Loss: 0.0002213889733699635, Correct: 50, Time per epoch: 1.590 seconds
Epoch 220, Loss: 0.021520866646429205, Correct: 50, Time per epoch: 1.255 seconds
Epoch 230, Loss: 0.39425127362598156, Correct: 50, Time per epoch: 1.308 seconds
Epoch 240, Loss: 0.0009194384819498124, Correct: 50, Time per epoch: 1.245 seconds
Epoch 250, Loss: 0.018950900036993037, Correct: 50, Time per epoch: 1.239 seconds
Epoch 260, Loss: 0.05070554775392251, Correct: 50, Time per epoch: 1.261 seconds
Epoch 270, Loss: 0.19320026352322967, Correct: 50, Time per epoch: 1.679 seconds
Epoch 280, Loss: 0.22904571305211852, Correct: 50, Time per epoch: 1.807 seconds
Epoch 290, Loss: 0.10257458576599873, Correct: 50, Time per epoch: 1.348 seconds
Epoch 300, Loss: 0.05554651272271164, Correct: 50, Time per epoch: 1.245 seconds
Epoch 310, Loss: 0.012029553779957928, Correct: 50, Time per epoch: 1.262 seconds
Epoch 320, Loss: 0.36346957446467043, Correct: 50, Time per epoch: 1.263 seconds
Epoch 330, Loss: 0.07189030303465155, Correct: 50, Time per epoch: 1.253 seconds
Epoch 340, Loss: 0.05840808603394396, Correct: 50, Time per epoch: 1.257 seconds
Epoch 350, Loss: 0.16463039831767914, Correct: 50, Time per epoch: 1.235 seconds
Epoch 360, Loss: 0.0034316777355818356, Correct: 50, Time per epoch: 1.249 seconds
Epoch 370, Loss: 0.017198901406464136, Correct: 50, Time per epoch: 1.250 seconds
Epoch 380, Loss: 0.0722852022899308, Correct: 50, Time per epoch: 1.276 seconds
Epoch 390, Loss: 0.0670352680823978, Correct: 50, Time per epoch: 1.617 seconds
Epoch 400, Loss: 0.024650466836482695, Correct: 50, Time per epoch: 1.872 seconds
Epoch 410, Loss: 0.08296411016853746, Correct: 50, Time per epoch: 1.427 seconds
Epoch 420, Loss: 0.23454640355956932, Correct: 50, Time per epoch: 1.239 seconds
Epoch 430, Loss: 5.235515792729864e-05, Correct: 50, Time per epoch: 1.236 seconds
Epoch 440, Loss: 0.04996230507233987, Correct: 50, Time per epoch: 1.241 seconds
Epoch 450, Loss: 0.01755549149210441, Correct: 50, Time per epoch: 1.256 seconds
Epoch 460, Loss: 0.0001060191300252888, Correct: 50, Time per epoch: 1.244 seconds
Epoch 470, Loss: 0.013784829756963823, Correct: 50, Time per epoch: 1.327 seconds
Epoch 480, Loss: 0.09742278960711087, Correct: 50, Time per epoch: 1.272 seconds
Epoch 490, Loss: 0.054498158578108566, Correct: 50, Time per epoch: 1.241 seconds
```