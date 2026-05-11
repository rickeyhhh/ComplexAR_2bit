import triton
import triton.language as tl
import torch


@triton.jit
def _quant_logic(U_re: tl.tensor, U_im: tl.tensor, dtype=tl.bfloat16):
    abs_real = tl.abs(U_re)
    abs_imag = tl.abs(U_im)
    is_real_dominant = abs_real > abs_imag

    out_real = tl.where(is_real_dominant, tl.where(U_re >= 0, 1, -1), 0)
    local_real_count = tl.sum(is_real_dominant).to(tl.float32)
    real_abs_bf16 = tl.where(is_real_dominant, abs_real, 0).to(tl.float32)
    local_real_sum = tl.sum(real_abs_bf16)
    real_scale = tl.where(local_real_count > 0, local_real_sum / local_real_count, 0.0)
    out_real = out_real * real_scale

    out_imag = tl.where(is_real_dominant, 0, tl.where(U_im >= 0, 1, -1))
    is_imag_dominant = (tl.abs(out_imag) > 0)
    local_imag_count = tl.sum(is_imag_dominant).to(tl.float32)
    imag_abs_bf16 = tl.where(is_imag_dominant, abs_imag, 0).to(tl.float32)
    local_imag_sum = tl.sum(imag_abs_bf16)
    imag_scale = tl.where(local_imag_count > 0, local_imag_sum / local_imag_count, 0.0)
    out_imag = out_imag * imag_scale

    return out_real.to(dtype), out_imag.to(dtype)


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE': B}, num_stages=s, num_warps=w)
        for B in [64]
        for s in [1, 2, 3, 4, 5]
        for w in [4, 8, 16]
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N'],
)
@triton.jit
def fairytoi_quant_kernel(
    A_ptr, B_ptr, A_row_stride, A_col_stride, M, N,
    SM_NUM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # A is a 2M x 2N matrix
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    TOTAL_TASK = num_pid_m * num_pid_n

    for pid in range(start_pid, TOTAL_TASK, SM_NUM):
        row_pid = pid // num_pid_n
        col_pid = pid % num_pid_n

        # A11, A12 = A[:n, :m], A[:n, m:]
        # A21, A22 = A[n:, :m], A[n:, m:]
        A11_ptr = tl.make_block_ptr(
            A_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE, col_pid * BLOCK_SIZE],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )
        A12_ptr = tl.make_block_ptr(
            A_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE, col_pid * BLOCK_SIZE + N],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )
        A21_ptr = tl.make_block_ptr(
            A_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE + M, col_pid * BLOCK_SIZE],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )
        A22_ptr = tl.make_block_ptr(
            A_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE + M, col_pid * BLOCK_SIZE + N],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )

        A11 = tl.load(A11_ptr)
        A12 = tl.load(A12_ptr)
        A21 = tl.load(A21_ptr)
        A22 = tl.load(A22_ptr)

        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)

        # Two-step residual quantization for U
        out_real, out_imag = _quant_logic(U_re, U_im)
        res_real, res_imag = _quant_logic(U_re - out_real, U_im - out_imag)
        U_out_real = out_real + res_real
        U_out_imag = out_imag + res_imag

        # Two-step residual quantization for W
        out_real, out_imag = _quant_logic(W_re, W_im)
        res_real, res_imag = _quant_logic(W_re - out_real, W_im - out_imag)
        W_out_real = out_real + res_real
        W_out_imag = out_imag + res_imag

        A11_q = W_out_real + U_out_real
        A12_q = W_out_imag - U_out_imag
        A21_q = W_out_imag + U_out_imag
        A22_q = -W_out_real + U_out_real

        B11_ptr = tl.make_block_ptr(
            B_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE, col_pid * BLOCK_SIZE],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )
        B12_ptr = tl.make_block_ptr(
            B_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE, col_pid * BLOCK_SIZE + N],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )
        B21_ptr = tl.make_block_ptr(
            B_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE + M, col_pid * BLOCK_SIZE],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )
        B22_ptr = tl.make_block_ptr(
            B_ptr, shape=[2 * M, 2 * N],
            strides=[A_row_stride, A_col_stride],
            offsets=[row_pid * BLOCK_SIZE + M, col_pid * BLOCK_SIZE + N],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE], order=[1, 0],
        )

        tl.store(B11_ptr, A11_q)
        tl.store(B22_ptr, A22_q)
        tl.store(B21_ptr, A21_q)
        tl.store(B12_ptr, A12_q)


def fairytoi_quant_block_V2(A: torch.tensor):
    assert A.is_contiguous(), "Input tensor A must be contiguous"
    M, N = A.shape[0] // 2, A.shape[1] // 2

    A_row_stride, A_col_stride = A.stride()
    B = torch.empty_like(A, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS,)

    fairytoi_quant_kernel[grid](A, B, A_row_stride, A_col_stride, M, N, NUM_SMS)
    return B
